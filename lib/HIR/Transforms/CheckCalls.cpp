//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "check-calls"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_CHECKCALLSPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

namespace {
struct CheckCallsPass : public hir::impl::CheckCallsPassBase<CheckCallsPass> {
  void runOnOperation() override;
  LogicalResult checkRegion(Region &region, const SymbolTable &symbolTable);
};
} // namespace

void CheckCallsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Since functions can call other functions in their signature, we need to
  // ensure that we process the functions called in a signature before we
  // process the function itself. We do this by manually setting up a post-order
  // traversal of the call graph in the function signatures, and we detect and
  // report recursions as we find them.
  struct WorklistItem {
    UnifiedFuncOp funcOp;
    Location callLoc;
    SmallVector<UnifiedCallOp, 1> calls;
  };
  SmallVector<WorklistItem> worklist;
  SmallDenseSet<UnifiedFuncOp> callStack;
  DenseSet<UnifiedFuncOp> checkedFuncs;

  auto addToWorklist = [&](UnifiedFuncOp funcOp, Location callLoc) {
    SmallDenseSet<UnifiedFuncOp, 1> seenCallees;
    SmallVector<UnifiedCallOp, 1> calls;
    funcOp.getSignature().walk([&](UnifiedCallOp op) {
      auto callee = symbolTable.lookup<UnifiedFuncOp>(op.getCallee());
      if (callee && !checkedFuncs.contains(callee))
        if (seenCallees.insert(callee).second)
          calls.push_back(op);
    });
    worklist.push_back({funcOp, callLoc, std::move(calls)});
  };

  bool anyErrors = false;
  for (auto funcOp : getOperation().getOps<UnifiedFuncOp>()) {
    if (checkedFuncs.contains(funcOp))
      continue;
    addToWorklist(funcOp, funcOp.getLoc());
    callStack.clear();
    callStack.insert(funcOp);
    while (!worklist.empty()) {
      auto &item = worklist.back();

      // If we have handled all calls in the function's signature, check the
      // function itself and remove it from the worklist.
      if (item.calls.empty()) {
        if (failed(checkRegion(item.funcOp.getSignature(), symbolTable)))
          anyErrors = true;
        checkedFuncs.insert(item.funcOp);
        callStack.erase(item.funcOp);
        worklist.pop_back();
        continue;
      }

      // Add the next call in the function's signature to the worklist. Uses the
      // call stack set to detect recursion.
      auto call = item.calls.pop_back_val();
      auto callee = symbolTable.lookup<UnifiedFuncOp>(call.getCallee());
      if (!callee || checkedFuncs.contains(callee))
        continue;
      if (callStack.insert(callee).second) {
        addToWorklist(callee, call.getLoc());
        continue;
      }

      // If we get here, the call stack already contained a call to this
      // function, which means that the function is on a recursive call chain.
      // Recursions through the function signature are not allowed.
      anyErrors = true;
      emitError(callee.getLoc())
          << "signature of `" << callee.getSymName() << "` cannot call itself";
      bool selfSeen = false;
      for (auto &frame : worklist) {
        if (!selfSeen) {
          selfSeen = frame.funcOp == callee;
          continue;
        }
        emitRemark(frame.callLoc)
            << "called through `" << frame.funcOp.getSymName() << "`";
      }
      emitRemark(call.getLoc())
          << "calling `" << callee.getSymName() << "` itself here";
    }
  }

  // If we have encountered any errors while checking calls in function
  // signatures, stop early here.
  if (anyErrors)
    signalPassFailure();

  // Process the function bodies in parallel.
  std::atomic<bool> anyBodyErrors = false;
  mlir::parallelForEach(
      &getContext(), getOperation().getOps<UnifiedFuncOp>(),
      [&](UnifiedFuncOp funcOp) {
        if (failed(checkRegion(funcOp.getBody(), symbolTable)))
          anyBodyErrors = true;
      });
  if (anyBodyErrors)
    signalPassFailure();
}

/// Inline the blocks in `region` into the `into` block at the given position.
static void inlineRegion(Region &region, Block &into,
                         Block::iterator before = {}) {
  // Inline the first block at the location where we would put the expr op.
  auto &firstBlock = region.front();
  into.getOperations().splice(before, firstBlock.getOperations());
  region.getBlocks().pop_front();

  // If there are any remaining blocks, split the `into` block and insert the
  // blocks in between the split.
  if (region.empty())
    return;
  auto &tailBlock = *into.splitBlock(before);
  auto &intoRegion = *tailBlock.getParent();
  intoRegion.getBlocks().splice(tailBlock.getIterator(), region.getBlocks());
}

/// Resolve a type value from the signature region into the body context.
///
/// If the value is a signature block argument, it is looked up in the mapping.
/// Otherwise, the defining op and its transitive operands are cloned into the
/// body. This handles cases like `uint_type(constant_int 42)` where the type
/// op depends on other ops in the signature region.
static Value resolveTypeIntoBody(OpBuilder &builder, Value sigTypeVal,
                                 IRMapping &sigToBody) {
  if (auto mapped = sigToBody.lookupOrNull(sigTypeVal))
    return mapped;
  auto *defOp = sigTypeVal.getDefiningOp();
  for (Value operand : defOp->getOperands())
    resolveTypeIntoBody(builder, operand, sigToBody);
  auto *cloned = builder.clone(*defOp, sigToBody);
  return cloned->getResult(cast<OpResult>(sigTypeVal).getResultNumber());
}

LogicalResult CheckCallsPass::checkRegion(Region &region,
                                          const SymbolTable &symbolTable) {
  auto funcOp = cast<UnifiedFuncOp>(region.getParentOp());
  bool isBody = &funcOp.getBody() == &region;
  LLVM_DEBUG({
    llvm::dbgs() << "Checking " << (isBody ? "body" : "signature") << " of "
                 << funcOp.getSymNameAttr() << "\n";
  });

  // Insert coerce_type annotations on body block args to connect them with
  // the declared types from the signature. This allows type_of(coerce_type(x,
  // T)) to fold to T during canonicalization, resolving type_of chains early.
  IRMapping sigToBody;
  if (isBody) {
    auto sigOp = funcOp.getSignatureOp();
    auto &sigBlock = funcOp.getSignature().front();
    auto &bodyBlock = funcOp.getBody().front();

    for (auto [sigArg, bodyArg] :
         llvm::zip(sigBlock.getArguments(), bodyBlock.getArguments()))
      sigToBody.map(sigArg, bodyArg);

    OpBuilder insertBuilder(&bodyBlock, bodyBlock.begin());
    for (auto [idx, bodyArg] : llvm::enumerate(bodyBlock.getArguments())) {
      if (idx >= sigOp.getTypeOfArgs().size())
        break;
      Value sigTypeVal = sigOp.getTypeOfArgs()[idx];
      Value resolvedType =
          resolveTypeIntoBody(insertBuilder, sigTypeVal, sigToBody);

      auto coerceOp = CoerceTypeOp::create(insertBuilder, bodyArg.getLoc(),
                                           bodyArg, resolvedType);
      bodyArg.replaceUsesWithIf(coerceOp.getResult(), [&](OpOperand &use) {
        return use.getOwner() != coerceOp;
      });
    }
  }

  // Process calls by inlining callee signatures and unifying types.
  region.walk([&](UnifiedCallOp callOp) {
    LLVM_DEBUG(llvm::dbgs() << "- " << callOp << "\n");

    // Clone the signature of the called function.
    auto *callee = symbolTable.lookup(callOp.getCallee());
    Region signature;
    IRMapping mapper;
    callee->getRegion(0).cloneInto(&signature, mapper);

    // Replace function arguments in the signature with the concrete values
    // provided by the call, and insert type unification as necessary.
    //
    // The cloned signature region has block arguments (one per function arg)
    // that stand in for the argument values in dependent-type expressions. We
    // replace those block arguments with the actual call arguments before
    // unifying the declared argument types with the call argument types.
    OpBuilder builder(callOp);
    auto terminatorOp =
        cast<UnifiedSignatureOp>(signature.back().getTerminator());

    // Replace the signature's block arguments with the actual call arguments.
    auto &sigBlock = signature.front();
    for (auto [blockArg, callArg] :
         llvm::zip(sigBlock.getArguments(), callOp.getArguments()))
      blockArg.replaceAllUsesWith(callArg);
    sigBlock.eraseArguments(0, sigBlock.getNumArguments());

    // Unify the declared argument types with the call's type-of-arg operands.
    // Inferrable placeholders are replaced directly with the signature type.
    // Concrete type operands are unified with the signature type.
    SmallVector<Value> unifiedArgTypes;
    unifiedArgTypes.reserve(callOp.getArguments().size());
    auto callTypeOfArgs = callOp.getTypeOfArgs();
    for (auto [idx, sigArgType] :
         llvm::enumerate(terminatorOp.getTypeOfArgs())) {
      builder.setInsertionPoint(terminatorOp);
      Value unified = sigArgType;
      if (idx < callTypeOfArgs.size()) {
        if (auto inferrable =
                callTypeOfArgs[idx].getDefiningOp<InferrableOp>()) {
          inferrable.replaceAllUsesWith(unified);
          inferrable.erase();
        } else {
          unified = UnifyOp::create(builder, callOp.getLoc(), sigArgType,
                                    callTypeOfArgs[idx]);
        }
      }
      unifiedArgTypes.push_back(unified);
    }

    // Unify the declared result types. If the call already carries
    // type-of-result operands, we also unify those. Inferrable placeholders
    // are replaced directly.
    SmallVector<Value> unifiedResultTypes;
    unifiedResultTypes.reserve(terminatorOp.getTypeOfResults().size());
    auto callTypeOfResults = callOp.getTypeOfResults();
    for (auto [idx, sigResultType] :
         llvm::enumerate(terminatorOp.getTypeOfResults())) {
      builder.setInsertionPoint(terminatorOp);
      Value unified = sigResultType;
      if (idx < callTypeOfResults.size()) {
        if (auto inferrable =
                callTypeOfResults[idx].getDefiningOp<InferrableOp>()) {
          inferrable.replaceAllUsesWith(unified);
          inferrable.erase();
        } else {
          unified = UnifyOp::create(builder, callOp.getLoc(), sigResultType,
                                    callTypeOfResults[idx]);
        }
      }
      unifiedResultTypes.push_back(unified);
    }

    // Update the call's type operands in-place.
    callOp.getTypeOfArgsMutable().assign(unifiedArgTypes);
    callOp.getTypeOfResultsMutable().assign(unifiedResultTypes);

    // Erase the signature terminator and inline the signature region.
    terminatorOp.erase();
    inlineRegion(signature, *callOp->getBlock(), callOp->getIterator());
  });

  // Unify body return types with the declared return types from the signature.
  // This connects the body's inferred return type to the declared type,
  // enabling InferTypes to resolve them. Also unify the return's typeOfArgs
  // with the declared argument types from the signature.
  if (isBody) {
    auto sigOp = funcOp.getSignatureOp();
    auto sigRetTypes = sigOp.getTypeOfResults();
    auto sigArgTypes = sigOp.getTypeOfArgs();

    funcOp.getBody().walk([&](UnifiedReturnOp returnOp) {
      OpBuilder builder(returnOp);

      // Unify return value types with the declared return types.
      if (!returnOp.getTypeOfValues().empty() && !sigRetTypes.empty()) {
        SmallVector<Value> newTypeOfValues;
        for (auto [idx, retTypeVal] :
             llvm::enumerate(returnOp.getTypeOfValues())) {
          if (idx >= sigRetTypes.size()) {
            newTypeOfValues.push_back(retTypeVal);
            continue;
          }
          Value resolvedSigRetType =
              resolveTypeIntoBody(builder, sigRetTypes[idx], sigToBody);
          Value unified = UnifyOp::create(builder, returnOp.getLoc(),
                                          retTypeVal, resolvedSigRetType);
          newTypeOfValues.push_back(unified);
        }
        returnOp.getTypeOfValuesMutable().assign(newTypeOfValues);
      }

      // Populate typeOfArgs from the declared argument types in the signature.
      // This threads the argument type information through to the return op so
      // it survives into hir.return after SplitPhases.
      SmallVector<Value> newTypeOfArgs;
      for (auto [idx, sigArgType] : llvm::enumerate(sigArgTypes)) {
        Value resolvedSigArgType =
            resolveTypeIntoBody(builder, sigArgType, sigToBody);
        newTypeOfArgs.push_back(resolvedSigArgType);
      }
      returnOp.getTypeOfArgsMutable().assign(newTypeOfArgs);
    });
  }

  return success();
}
