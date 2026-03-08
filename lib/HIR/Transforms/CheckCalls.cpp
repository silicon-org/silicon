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

/// Ensure there is exactly one `UnifiedSignatureOp` terminator in the cloned
/// signature region. If multiple terminators exist, they are consolidated into
/// a single exit block whose block arguments carry the typeOfArgs and
/// typeOfResults values.
static void consolidateSignatureTerminators(Region &clonedSig) {
  SmallVector<UnifiedSignatureOp> sigTerminators;
  for (auto &block : clonedSig)
    if (auto sigOp = dyn_cast<UnifiedSignatureOp>(block.getTerminator()))
      sigTerminators.push_back(sigOp);
  if (sigTerminators.size() <= 1)
    return;

  auto firstSigOp = sigTerminators.front();
  unsigned numArgTypes = firstSigOp.getTypeOfArgs().size();
  unsigned numResultTypes = firstSigOp.getTypeOfResults().size();

  // Create the exit block with one block arg per typeOfArgs + typeOfResults.
  auto *exitBlock = new Block();
  clonedSig.push_back(exitBlock);
  SmallVector<Type> blockArgTypes;
  SmallVector<Location> blockArgLocs;
  for (unsigned i = 0; i < numArgTypes; ++i) {
    blockArgTypes.push_back(firstSigOp.getTypeOfArgs()[i].getType());
    blockArgLocs.push_back(firstSigOp.getTypeOfArgs()[i].getLoc());
  }
  for (unsigned i = 0; i < numResultTypes; ++i) {
    blockArgTypes.push_back(firstSigOp.getTypeOfResults()[i].getType());
    blockArgLocs.push_back(firstSigOp.getTypeOfResults()[i].getLoc());
  }
  exitBlock->addArguments(blockArgTypes, blockArgLocs);

  // Create the consolidated signature terminator in the exit block.
  OpBuilder exitBuilder(firstSigOp->getContext());
  exitBuilder.setInsertionPointToStart(exitBlock);
  auto exitArgTypes = exitBlock->getArguments().take_front(numArgTypes);
  auto exitResultTypes = exitBlock->getArguments().drop_front(numArgTypes);
  UnifiedSignatureOp::create(exitBuilder, firstSigOp.getLoc(),
                             SmallVector<Value>(exitArgTypes),
                             SmallVector<Value>(exitResultTypes));

  // Replace each original terminator with a branch to the exit block.
  for (auto sigOp : sigTerminators) {
    OpBuilder builder(firstSigOp->getContext());
    builder.setInsertionPoint(sigOp);
    SmallVector<Value> branchArgs;
    branchArgs.append(sigOp.getTypeOfArgs().begin(),
                      sigOp.getTypeOfArgs().end());
    branchArgs.append(sigOp.getTypeOfResults().begin(),
                      sigOp.getTypeOfResults().end());
    ConstBranchOp::create(builder, sigOp.getLoc(), branchArgs, exitBlock);
    sigOp.erase();
  }
}

/// Clone the signature region's blocks into the body region and wire up
/// coerce_type and return type operands.
///
/// Instead of selectively cloning individual ops from the signature, this
/// clones all signature blocks into the body region. If there are multiple
/// `UnifiedSignatureOp` terminators (from multi-block signatures), they are
/// first consolidated into a single exit block. The cloned blocks are placed
/// before the body's entry block, making the cloned signature entry block the
/// new entry of the body region. The signature terminator's operands feed into
/// coerce_type ops on the block arguments and into the body return's type
/// operands.
static void cloneSignatureIntoBody(UnifiedFuncOp funcOp) {
  auto &sigRegion = funcOp.getSignature();
  auto &bodyRegion = funcOp.getBody();
  auto &bodyBlock = bodyRegion.front();

  // Clone the signature region into a temporary region and consolidate
  // multiple signature terminators if needed.
  Region clonedSig;
  IRMapping sigToBody;
  sigRegion.cloneInto(&clonedSig, sigToBody);
  consolidateSignatureTerminators(clonedSig);

  // Find the single signature terminator and extract its operands.
  auto &clonedEntry = clonedSig.front();
  UnifiedSignatureOp clonedSigOp = nullptr;
  for (auto &block : clonedSig)
    if (auto op = dyn_cast<UnifiedSignatureOp>(block.getTerminator()))
      clonedSigOp = op;
  assert(clonedSigOp && "expected a single signature terminator");
  SmallVector<Value> clonedArgTypes(clonedSigOp.getTypeOfArgs());
  SmallVector<Value> clonedResultTypes(clonedSigOp.getTypeOfResults());

  // Move all cloned blocks before the body block, making the cloned entry
  // block the new entry of the body region. For single-block signatures, we
  // merge the blocks to keep the body as a single block (which downstream
  // passes like SplitPhases expect). For multi-block signatures, the cloned
  // blocks form a preamble that branches into the body block, passing the type
  // values as block arguments.
  // Track the insertion point for coerce_type ops, which must go after the
  // inlined sig ops but before the original body ops.
  Block::iterator coerceInsertPt = bodyBlock.begin();

  if (clonedSig.hasOneBlock()) {
    // Single block: replace cloned entry args with body block args, update the
    // type vectors to point at body block args, erase the terminator, and
    // splice ops directly into the body block at the beginning.
    for (auto [sigArg, bodyArg] :
         llvm::zip(clonedEntry.getArguments(), bodyBlock.getArguments())) {
      // Update clonedArgTypes/clonedResultTypes if they reference this arg.
      for (auto &v : clonedArgTypes)
        if (v == sigArg)
          v = bodyArg;
      for (auto &v : clonedResultTypes)
        if (v == sigArg)
          v = bodyArg;
      sigArg.replaceAllUsesWith(bodyArg);
    }
    clonedEntry.eraseArguments(0, clonedEntry.getNumArguments());
    clonedSigOp.erase();
    bodyBlock.getOperations().splice(coerceInsertPt,
                                     clonedEntry.getOperations());
    // coerceInsertPt still points to the first original body op.
  } else {
    // Multi-block: the cloned entry block keeps its args (the function args).
    // Replace body block arg uses with cloned entry block args, then erase
    // the body block args.
    for (auto [bodyArg, sigArg] :
         llvm::zip(bodyBlock.getArguments(), clonedEntry.getArguments()))
      bodyArg.replaceAllUsesWith(sigArg);
    bodyBlock.eraseArguments(0, bodyBlock.getNumArguments());

    // Pass the type values through the branch as block arguments so they
    // dominate their uses in the body block.
    SmallVector<Value> branchArgs;
    branchArgs.append(clonedArgTypes.begin(), clonedArgTypes.end());
    branchArgs.append(clonedResultTypes.begin(), clonedResultTypes.end());

    SmallVector<Type> branchArgTypes;
    SmallVector<Location> branchArgLocs;
    for (auto v : branchArgs) {
      branchArgTypes.push_back(v.getType());
      branchArgLocs.push_back(v.getLoc());
    }
    bodyBlock.addArguments(branchArgTypes, branchArgLocs);

    // Update clonedArgTypes/clonedResultTypes to point at the new body block
    // arguments instead of the values in the cloned region.
    for (unsigned i = 0; i < clonedArgTypes.size(); ++i)
      clonedArgTypes[i] = bodyBlock.getArgument(i);
    for (unsigned i = 0; i < clonedResultTypes.size(); ++i)
      clonedResultTypes[i] = bodyBlock.getArgument(clonedArgTypes.size() + i);

    // Erase the signature terminator. If the terminator block is now empty
    // (e.g., a consolidated exit block or a single-terminator exit block with
    // no other ops), eliminate it by redirecting predecessor branches directly
    // to the body block instead of creating a pass-through block.
    Block *terminatorBlock = clonedSigOp->getBlock();
    auto sigLoc = clonedSigOp.getLoc();
    clonedSigOp.erase();
    if (terminatorBlock->empty()) {
      for (auto it = terminatorBlock->pred_begin();
           it != terminatorBlock->pred_end();) {
        Block *pred = *it++;
        auto *term = pred->getTerminator();
        for (unsigned i = 0; i < term->getNumSuccessors(); ++i)
          if (term->getSuccessor(i) == terminatorBlock)
            term->setSuccessor(&bodyBlock, i);
      }
      terminatorBlock->erase();
    } else {
      OpBuilder branchBuilder(terminatorBlock, terminatorBlock->end());
      ConstBranchOp::create(branchBuilder, sigLoc, branchArgs, &bodyBlock);
    }

    // Move the cloned blocks before the body block.
    bodyRegion.getBlocks().splice(bodyRegion.begin(), clonedSig.getBlocks());
  }

  // Insert coerce_type ops right before the original body ops. For single-
  // block signatures, this is after the inlined sig ops. For multi-block,
  // this is at the top of the body block (types come through block args).
  auto &entryBlock = bodyRegion.front();
  OpBuilder insertBuilder(&bodyBlock, coerceInsertPt);
  for (auto [idx, entryArg] : llvm::enumerate(entryBlock.getArguments())) {
    if (idx >= clonedArgTypes.size())
      break;
    auto coerceOp = CoerceTypeOp::create(insertBuilder, entryArg.getLoc(),
                                         entryArg, clonedArgTypes[idx]);
    entryArg.replaceUsesWithIf(coerceOp.getResult(), [&](OpOperand &use) {
      return use.getOwner() != coerceOp;
    });
  }

  // Unify body return types with the declared return types from the signature.
  // Also populate typeOfArgs from the declared argument types. Since the
  // signature ops are now part of the body region, we can use them directly.
  bodyRegion.walk([&](ReturnOp returnOp) {
    OpBuilder builder(returnOp);

    // Unify return value types with the declared return types.
    if (!returnOp.getTypeOfValues().empty() && !clonedResultTypes.empty()) {
      SmallVector<Value> newTypeOfValues;
      for (auto [idx, retTypeVal] :
           llvm::enumerate(returnOp.getTypeOfValues())) {
        if (idx >= clonedResultTypes.size()) {
          newTypeOfValues.push_back(retTypeVal);
          continue;
        }
        Value unified = UnifyOp::create(builder, returnOp.getLoc(), retTypeVal,
                                        clonedResultTypes[idx]);
        newTypeOfValues.push_back(unified);
      }
      returnOp.getTypeOfValuesMutable().assign(newTypeOfValues);
    }

    // Populate typeOfArgs from the declared argument types.
    SmallVector<Value> newTypeOfArgs;
    for (Value argType : clonedArgTypes)
      newTypeOfArgs.push_back(argType);
    returnOp.getTypeOfArgsMutable().assign(newTypeOfArgs);
  });
}

LogicalResult CheckCallsPass::checkRegion(Region &region,
                                          const SymbolTable &symbolTable) {
  auto funcOp = cast<UnifiedFuncOp>(region.getParentOp());
  bool isBody = &funcOp.getBody() == &region;
  LLVM_DEBUG({
    llvm::dbgs() << "Checking " << (isBody ? "body" : "signature") << " of "
                 << funcOp.getSymNameAttr() << "\n";
  });

  // Process calls by inlining callee signatures and unifying types. The
  // callee's signature region is cloned, multiple terminators are consolidated
  // into a single exit block, and the signature blocks are inlined ahead of
  // the call. For single-block signatures, the ops are spliced directly before
  // the call. For multi-block signatures, the call's block is split and the
  // signature blocks form a preamble, with type values threaded as block
  // arguments to the continuation block containing the call.
  //
  // Collect call ops upfront since multi-block inlining splits blocks, which
  // invalidates the walk iterator.
  SmallVector<UnifiedCallOp> callOps;
  region.walk([&](UnifiedCallOp op) { callOps.push_back(op); });
  for (auto callOp : callOps) {
    LLVM_DEBUG(llvm::dbgs() << "- " << callOp << "\n");

    // Clone the signature of the called function and consolidate multiple
    // signature terminators into a single exit block.
    auto *callee = symbolTable.lookup(callOp.getCallee());
    Region signature;
    IRMapping mapper;
    callee->getRegion(0).cloneInto(&signature, mapper);
    consolidateSignatureTerminators(signature);

    // Find the single signature terminator.
    UnifiedSignatureOp terminatorOp = nullptr;
    for (auto &block : signature)
      if (auto op = dyn_cast<UnifiedSignatureOp>(block.getTerminator()))
        terminatorOp = op;
    assert(terminatorOp && "expected a single signature terminator");

    // Replace the signature's entry block arguments with the actual call
    // arguments. These stand in for function parameters in dependent-type
    // expressions.
    auto &sigEntry = signature.front();
    for (auto [blockArg, callArg] :
         llvm::zip(sigEntry.getArguments(), callOp.getArguments()))
      blockArg.replaceAllUsesWith(callArg);
    sigEntry.eraseArguments(0, sigEntry.getNumArguments());

    // Unify the declared argument types with the call's type-of-arg operands.
    // Track inferrable placeholders for later replacement (after inlining,
    // when the final type values are known).
    struct InferrableReplacement {
      InferrableOp op;
      bool isResult;
      unsigned index;
    };
    SmallVector<InferrableReplacement> inferrables;

    OpBuilder builder(terminatorOp);
    SmallVector<Value> unifiedArgTypes;
    auto callTypeOfArgs = callOp.getTypeOfArgs();
    for (auto [idx, sigArgType] :
         llvm::enumerate(terminatorOp.getTypeOfArgs())) {
      Value unified = sigArgType;
      if (idx < callTypeOfArgs.size()) {
        if (auto inferrable =
                callTypeOfArgs[idx].getDefiningOp<InferrableOp>()) {
          inferrables.push_back({inferrable, false, (unsigned)idx});
        } else {
          unified = UnifyOp::create(builder, callOp.getLoc(), sigArgType,
                                    callTypeOfArgs[idx]);
        }
      }
      unifiedArgTypes.push_back(unified);
    }

    // Unify the declared result types.
    SmallVector<Value> unifiedResultTypes;
    auto callTypeOfResults = callOp.getTypeOfResults();
    for (auto [idx, sigResultType] :
         llvm::enumerate(terminatorOp.getTypeOfResults())) {
      Value unified = sigResultType;
      if (idx < callTypeOfResults.size()) {
        if (auto inferrable =
                callTypeOfResults[idx].getDefiningOp<InferrableOp>()) {
          inferrables.push_back({inferrable, true, (unsigned)idx});
        } else {
          unified = UnifyOp::create(builder, callOp.getLoc(), sigResultType,
                                    callTypeOfResults[idx]);
        }
      }
      unifiedResultTypes.push_back(unified);
    }

    // Erase the signature terminator and inline the signature region.
    Block *terminatorBlock = terminatorOp->getBlock();
    terminatorOp.erase();

    if (signature.hasOneBlock()) {
      // Single block: splice ops directly before the call in the same block.
      callOp->getBlock()->getOperations().splice(callOp->getIterator(),
                                                 sigEntry.getOperations());
    } else {
      // Multi-block: split the call's block and insert signature blocks as a
      // preamble. Type values are threaded as block arguments to the
      // continuation block containing the call.
      Block *callBlock = callOp->getBlock();
      Block *continuation = callBlock->splitBlock(callOp);

      // Add block arguments for the type values to the continuation block.
      SmallVector<Value> typeValues;
      typeValues.append(unifiedArgTypes.begin(), unifiedArgTypes.end());
      typeValues.append(unifiedResultTypes.begin(), unifiedResultTypes.end());

      SmallVector<Type> tvTypes;
      SmallVector<Location> tvLocs;
      for (auto v : typeValues) {
        tvTypes.push_back(v.getType());
        tvLocs.push_back(v.getLoc());
      }
      continuation->addArguments(tvTypes, tvLocs);

      // Update unified type vectors to point at continuation block args.
      for (unsigned i = 0; i < unifiedArgTypes.size(); ++i)
        unifiedArgTypes[i] = continuation->getArgument(i);
      for (unsigned i = 0; i < unifiedResultTypes.size(); ++i)
        unifiedResultTypes[i] =
            continuation->getArgument(unifiedArgTypes.size() + i);

      // Splice the entry block's ops into the call block (before the split
      // point), then move remaining signature blocks between callBlock and
      // continuation. This must happen before the empty-block elimination
      // below, since that redirects branches to the continuation block which
      // is in the parent region.
      callBlock->getOperations().splice(callBlock->end(),
                                        sigEntry.getOperations());
      signature.getBlocks().pop_front();
      if (!signature.empty()) {
        auto &parentRegion = *continuation->getParent();
        parentRegion.getBlocks().splice(continuation->getIterator(),
                                        signature.getBlocks());
      }

      // If the terminator block is now empty (e.g., a consolidated exit
      // block), eliminate it by redirecting predecessor branches directly to
      // the continuation block. Otherwise, add a branch to the continuation.
      if (terminatorBlock->empty()) {
        for (auto it = terminatorBlock->pred_begin();
             it != terminatorBlock->pred_end();) {
          Block *pred = *it++;
          auto *term = pred->getTerminator();
          for (unsigned i = 0; i < term->getNumSuccessors(); ++i)
            if (term->getSuccessor(i) == terminatorBlock)
              term->setSuccessor(continuation, i);
        }
        terminatorBlock->erase();
      } else {
        OpBuilder branchBuilder(callOp->getContext());
        branchBuilder.setInsertionPointToEnd(terminatorBlock);
        ConstBranchOp::create(branchBuilder, callOp.getLoc(), typeValues,
                              continuation);
      }
    }

    // Replace inferrable placeholders with the final unified type values.
    for (auto &r : inferrables) {
      Value replacement =
          r.isResult ? unifiedResultTypes[r.index] : unifiedArgTypes[r.index];
      r.op.replaceAllUsesWith(replacement);
      r.op.erase();
    }

    // Update the call's type operands in-place.
    callOp.getTypeOfArgsMutable().assign(unifiedArgTypes);
    callOp.getTypeOfResultsMutable().assign(unifiedResultTypes);
  }

  // Clone the signature blocks into the body region. This replaces the body
  // block's arguments with the signature entry block's arguments, inserts
  // coerce_type ops, and wires up the return type operands.
  if (isBody)
    cloneSignatureIntoBody(funcOp);

  return success();
}
