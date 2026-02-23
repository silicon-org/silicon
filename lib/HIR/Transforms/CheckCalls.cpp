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

LogicalResult CheckCallsPass::checkRegion(Region &region,
                                          const SymbolTable &symbolTable) {
  auto funcOp = cast<UnifiedFuncOp>(region.getParentOp());
  LLVM_DEBUG({
    llvm::dbgs() << "Checking "
                 << (&funcOp.getSignature() == &region ? "signature" : "body")
                 << " of " << funcOp.getSymNameAttr() << "\n";
  });
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
    SmallVector<Value> argTypes;
    SmallVector<int32_t> constnessOfArgs;
    SmallVector<int32_t> constnessOfResults;
    argTypes.reserve(callOp.getArguments().size());
    constnessOfArgs.reserve(callOp.getArguments().size());
    constnessOfResults.reserve(callOp.getResults().size());

    // Replace the signature's block arguments with the actual call arguments.
    auto &sigBlock = signature.front();
    for (auto [blockArg, callArg] :
         llvm::zip(sigBlock.getArguments(), callOp.getArguments()))
      blockArg.replaceAllUsesWith(callArg);
    sigBlock.eraseArguments(0, sigBlock.getNumArguments());

    // Unify the declared argument types with the call argument types.
    auto argPhasesFromCall = callOp.getArgPhases();
    unsigned argIdx = 0;
    for (auto [argType, callArg] :
         llvm::zip(terminatorOp.getTypeOfArgs(), callOp.getArguments())) {
      builder.setInsertionPoint(terminatorOp);
      auto callArgType = TypeOfOp::create(builder, callArg.getLoc(), callArg);
      auto unifiedType =
          UnifyOp::create(builder, callOp.getLoc(), argType, callArgType);
      argTypes.push_back(unifiedType);
      constnessOfArgs.push_back(argPhasesFromCall[argIdx++]);
    }

    // Results carry phase annotations from the callee's resultPhases.
    for (auto phase : callOp.getResultPhases())
      constnessOfResults.push_back(phase);

    // Replace the call op with a variant that encodes the exact argument and
    // result types.
    builder.setInsertionPoint(callOp);
    // With !hir.any, result types are the same as the type operand types.
    SmallVector<Type> resultTypes(terminatorOp.getTypeOfResults().size(),
                                  AnyType::get(callOp.getContext()));
    auto newCallOp = CheckedCallOp::create(
        builder, callOp.getLoc(), resultTypes, callOp.getCallee(),
        callOp.getArguments(), argTypes, terminatorOp.getTypeOfResults(),
        constnessOfArgs, constnessOfResults);
    callOp.replaceAllUsesWith(newCallOp);
    callOp.erase();
    terminatorOp.erase();

    // Inline the signature into the call site.
    inlineRegion(signature, *newCallOp->getBlock(), newCallOp->getIterator());
  });
  return success();
}
