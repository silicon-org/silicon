//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # FlattenCF: Lower Structured UIR Control Flow to Block-Based CF
//
// Converts uir.if, uir.loop, and their terminators (uir.yield, uir.break,
// uir.continue, uir.return, uir.unreachable) into cf.br, cf.cond_br, and
// hir.return. Processes ops innermost-first (post-order) so that each
// lowering step handles only one level of structured CF.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "flatten-cf"

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_FLATTENCFPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

//===----------------------------------------------------------------------===//
// uir.if lowering
//===----------------------------------------------------------------------===//

/// Lower a uir.if op into cf.cond_br + then/else/merge blocks.
static LogicalResult lowerIfOp(IfOp ifOp) {
  auto loc = ifOp.getLoc();
  auto anyTy = hir::AnyType::get(ifOp.getContext());
  auto *parentRegion = ifOp->getBlock()->getParent();

  // Gather info before we start mutating. After inner lowerings, regions
  // may have multiple blocks. The UIR terminator is in the last block.
  unsigned numResults = ifOp.getNumResults();
  bool hasElse = !ifOp.getElseRegion().empty();
  auto *thenTerminator = ifOp.getThenRegion().back().getTerminator();
  bool thenYields = isa<YieldOp>(thenTerminator);
  Operation *elseTerminator =
      hasElse ? ifOp.getElseRegion().back().getTerminator() : nullptr;
  bool elseYields = elseTerminator && isa<YieldOp>(elseTerminator);
  bool needsMerge = thenYields || elseYields;

  // Split the current block right after the if op. Everything after the
  // if goes into a continuation block.
  auto *currentBlock = ifOp->getBlock();
  auto *contBlock = currentBlock->splitBlock(ifOp->getNextNode());

  // Create merge block if needed (at least one branch yields).
  // Only value results need block args — the if's resultTypes operands
  // are already SSA values visible outside the if.
  Block *mergeBlock = nullptr;
  if (needsMerge) {
    mergeBlock = new Block();
    for (unsigned i = 0; i < numResults; ++i)
      mergeBlock->addArgument(anyTy, loc);
  }

  // Move the then region's block into the parent region.
  auto *thenBlock = &ifOp.getThenRegion().front();
  parentRegion->getBlocks().splice(Region::iterator(contBlock),
                                   ifOp.getThenRegion().getBlocks());

  // Move the else region's block into the parent region (if present).
  Block *elseBlock = nullptr;
  if (hasElse) {
    elseBlock = &ifOp.getElseRegion().front();
    parentRegion->getBlocks().splice(Region::iterator(contBlock),
                                     ifOp.getElseRegion().getBlocks());
  }

  // Now lower the terminators. After inner lowerings, the UIR terminator
  // is in the last block of what was the region (now inlined into parent).
  // We saved the terminator pointer before splicing.
  OpBuilder builder(ifOp.getContext());

  // Helper to lower a UIR terminator in an if-branch.
  auto lowerBranchTerminator = [&](Operation *term) {
    builder.setInsertionPoint(term);
    if (auto yieldOp = dyn_cast<YieldOp>(term)) {
      // Only branch with values — type operands from the yield are not
      // needed; the if's resultTypes operands serve that role outside.
      cf::BranchOp::create(builder, loc, mergeBlock, yieldOp.getValues());
      term->erase();
    } else if (auto returnOp = dyn_cast<ReturnOp>(term)) {
      hir::ReturnOp::create(builder, loc, returnOp.getValues(),
                            returnOp.getTypeOfValues());
      term->erase();
    }
    // uir.unreachable: left in place. If the block is truly unreachable
    // (no predecessors), dead block cleanup removes it. If it survives,
    // the post-lowering check reports a compiler bug.
  };

  // Lower then and else terminators.
  lowerBranchTerminator(thenTerminator);
  if (elseBlock)
    lowerBranchTerminator(elseTerminator);

  // Insert merge block before continuation.
  if (mergeBlock) {
    parentRegion->getBlocks().insert(Region::iterator(contBlock), mergeBlock);

    // Replace uses of the if op's results with merge block arguments.
    for (unsigned i = 0; i < numResults; ++i)
      ifOp.getResult(i).replaceAllUsesWith(mergeBlock->getArgument(i));

    // Merge block falls through to continuation.
    builder.setInsertionPointToEnd(mergeBlock);
    cf::BranchOp::create(builder, loc, contBlock);
  }

  // Emit the conditional branch at the end of the original block.
  builder.setInsertionPointToEnd(currentBlock);
  auto i1Cond = hir::CoerceToI1Op::create(builder, loc, ifOp.getCondition());
  Block *elseTarget = elseBlock ? elseBlock : contBlock;
  cf::CondBranchOp::create(builder, loc, i1Cond, thenBlock, elseTarget);

  // Erase the if op (regions are now empty).
  ifOp.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Dead block cleanup
//===----------------------------------------------------------------------===//

/// Remove blocks with no predecessors (except the entry block).
static void cleanupDeadBlocks(Region &region) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &block : llvm::make_early_inc_range(region.getBlocks())) {
      if (&block == &region.front())
        continue;
      if (block.hasNoPredecessors()) {
        block.dropAllDefinedValueUses();
        block.erase();
        changed = true;
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

namespace {
struct FlattenCFPass : uir::impl::FlattenCFPassBase<FlattenCFPass> {
  using FlattenCFPassBase::FlattenCFPassBase;

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Collect all uir.if ops in post-order (innermost first).
    SmallVector<IfOp> ifOps;
    moduleOp->walk([&](IfOp op) { ifOps.push_back(op); });

    // Lower each if op.
    for (auto ifOp : ifOps) {
      if (failed(lowerIfOp(ifOp))) {
        signalPassFailure();
        return;
      }
    }

    // Clean up dead blocks (from uir.unreachable after all-early-exit ifs).
    moduleOp->walk([](Region *region) { cleanupDeadBlocks(*region); });

    // Any surviving uir.unreachable ops indicate a compiler bug.
    bool hasError = false;
    moduleOp->walk(
        [&](UnreachableOp op) {
          emitBug(op.getLoc())
              << "uir.unreachable survived FlattenCF (the block should have "
                 "been unreachable after lowering structured CF)";
          hasError = true;
        });
    if (hasError) {
      signalPassFailure();
      return;
    }
  }
};
} // namespace
