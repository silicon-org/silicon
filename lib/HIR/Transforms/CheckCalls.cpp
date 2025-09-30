//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/HIR/Types.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/IRMapping.h>

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
    UncheckedFuncOp funcOp;
    Location callLoc;
    SmallVector<UncheckedCallOp, 1> calls;
  };
  SmallVector<WorklistItem> worklist;
  SmallDenseSet<UncheckedFuncOp> callStack;
  DenseSet<UncheckedFuncOp> checkedFuncs;

  auto addToWorklist = [&](UncheckedFuncOp funcOp, Location callLoc) {
    SmallDenseSet<UncheckedFuncOp, 1> seenCallees;
    SmallVector<UncheckedCallOp, 1> calls;
    funcOp.getSignature().walk([&](UncheckedCallOp op) {
      auto callee = symbolTable.lookup<UncheckedFuncOp>(op.getCallee());
      if (callee && !checkedFuncs.contains(callee))
        if (seenCallees.insert(callee).second)
          calls.push_back(op);
    });
    worklist.push_back({funcOp, callLoc, std::move(calls)});
  };

  bool anyErrors = false;
  for (auto funcOp : getOperation().getOps<UncheckedFuncOp>()) {
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
      auto callee = symbolTable.lookup<UncheckedFuncOp>(call.getCallee());
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
      &getContext(), getOperation().getOps<UncheckedFuncOp>(),
      [&](UncheckedFuncOp funcOp) {
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
  auto funcOp = cast<UncheckedFuncOp>(region.getParentOp());
  LLVM_DEBUG({
    llvm::dbgs() << "Checking "
                 << (&funcOp.getSignature() == &region ? "signature" : "body")
                 << " of " << funcOp.getSymNameAttr() << "\n";
  });
  region.walk([&](UncheckedCallOp callOp) {
    LLVM_DEBUG(llvm::dbgs() << "- " << callOp << "\n");

    // Clone the signature of the called function.
    auto *callee = symbolTable.lookup(callOp.getCallee());
    Region signature;
    IRMapping mapper;
    callee->getRegion(0).cloneInto(&signature, mapper);

    // Replace function arguments in the signature with the concrete values
    // provided by the call, and insert type unification as necessary.
    OpBuilder builder(callOp);
    auto terminatorOp =
        cast<UncheckedSignatureOp>(signature.back().getTerminator());
    SmallVector<Value> argTypes;
    argTypes.reserve(callOp.getArguments().size());
    for (auto [fnArg, callArg] :
         llvm::zip(terminatorOp.getArgValues(), callOp.getArguments())) {
      auto fnArgOp = cast<UncheckedArgOp>(fnArg.getDefiningOp());
      builder.setInsertionPoint(fnArgOp);
      auto callArgType = TypeOfOp::create(builder, callArg.getLoc(), callArg);
      auto unifiedType = UnifyOp::create(builder, callOp.getLoc(),
                                         fnArgOp.getTypeOperand(), callArgType);
      argTypes.push_back(unifiedType);
      fnArgOp.replaceAllUsesWith(callArg);
      fnArgOp.erase();
    }

    // Replace the call op with a variant that encodes the exact argument and
    // result types.
    builder.setInsertionPoint(callOp);
    auto newCallOp = CheckedCallOp::create(
        builder, callOp.getLoc(),
        getLowerKindRange(terminatorOp.getTypeOfResults().getTypes()),
        callOp.getCallee(), callOp.getArguments(), argTypes,
        terminatorOp.getTypeOfResults());
    callOp.replaceAllUsesWith(newCallOp);
    callOp.erase();
    terminatorOp.erase();

    // Inline the signature into the call site.
    inlineRegion(signature, *newCallOp->getBlock(), newCallOp->getIterator());
  });
  return success();
}
