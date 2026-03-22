//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # FlattenCF: Lower Structured UIR Ops to Block-Based CF
//
// Converts structured UIR ops into flat block-based IR:
// - uir.if, uir.loop, and their terminators (uir.yield, uir.break,
//   uir.continue, uir.return, uir.unreachable) into cf.br, cf.cond_br,
//   and hir.return.
// - uir.expr is inlined into the parent block.
// - uir.pin is replaced by its inputs (identity removal).
//
// Control flow ops are processed innermost-first (post-order) so that each
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
// uir.loop lowering
//===----------------------------------------------------------------------===//

/// Lower a uir.loop op into a loop header block + exit block.
///
/// The loop body is spliced into the parent region. All `uir.yield` and
/// `uir.continue` terminators become `cf.br ^header`. All `uir.break`
/// terminators become `cf.br ^exit(values)`. `uir.return` terminators
/// become `hir.return`.
static LogicalResult lowerLoopOp(LoopOp loopOp) {
  auto loc = loopOp.getLoc();
  auto anyTy = hir::AnyType::get(loopOp.getContext());
  auto *parentRegion = loopOp->getBlock()->getParent();
  unsigned numResults = loopOp.getNumResults();

  // Split the current block after the loop op.
  auto *currentBlock = loopOp->getBlock();
  auto *contBlock = currentBlock->splitBlock(loopOp->getNextNode());

  // Create the exit block with args for the loop's value results.
  auto *exitBlock = new Block();
  for (unsigned i = 0; i < numResults; ++i)
    exitBlock->addArgument(anyTy, loc);

  // Splice the loop body into the parent region. The first block of the
  // body becomes the loop header.
  auto *headerBlock = &loopOp.getBody().front();
  parentRegion->getBlocks().splice(Region::iterator(contBlock),
                                   loopOp.getBody().getBlocks());

  // Branch from the current block to the loop header.
  OpBuilder builder(loopOp.getContext());
  builder.setInsertionPointToEnd(currentBlock);
  cf::BranchOp::create(builder, loc, headerBlock);

  // Walk all blocks that came from the body and replace UIR terminators.
  // We iterate from headerBlock to contBlock (exclusive).
  for (auto it = Region::iterator(headerBlock),
            end = Region::iterator(contBlock);
       it != end; ++it) {
    auto *term = it->getTerminator();

    if (auto yieldOp = dyn_cast<YieldOp>(term)) {
      // yield = continue to next iteration.
      builder.setInsertionPoint(term);
      cf::BranchOp::create(builder, loc, headerBlock);
      term->erase();
    } else if (auto continueOp = dyn_cast<ContinueOp>(term)) {
      builder.setInsertionPoint(term);
      cf::BranchOp::create(builder, loc, headerBlock);
      term->erase();
    } else if (auto breakOp = dyn_cast<BreakOp>(term)) {
      // break = exit loop with values.
      builder.setInsertionPoint(term);
      cf::BranchOp::create(builder, loc, exitBlock, breakOp.getValues());
      term->erase();
    } else if (auto returnOp = dyn_cast<ReturnOp>(term)) {
      builder.setInsertionPoint(term);
      hir::ReturnOp::create(builder, loc, returnOp.getValues(),
                            returnOp.getTypeOfValues());
      term->erase();
    }
  }

  // Insert exit block before continuation.
  parentRegion->getBlocks().insert(Region::iterator(contBlock), exitBlock);

  // Replace uses of the loop's results with exit block arguments.
  for (unsigned i = 0; i < numResults; ++i)
    loopOp.getResult(i).replaceAllUsesWith(exitBlock->getArgument(i));

  // Exit block falls through to continuation.
  builder.setInsertionPointToEnd(exitBlock);
  cf::BranchOp::create(builder, loc, contBlock);

  // Erase the loop op (body region is now empty).
  loopOp.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// uir.expr lowering
//===----------------------------------------------------------------------===//

/// Lower a uir.expr op by inlining its body into the parent block.
///
/// The expr's results are replaced with the yield's values. The yield and
/// expr ops are erased.
static LogicalResult lowerExprOp(ExprOp exprOp) {
  auto &body = exprOp.getBody();
  assert(body.hasOneBlock() && "expr body must have a single block");

  auto *parentBlock = exprOp->getBlock();
  auto &bodyBlock = body.front();
  auto yieldOp = cast<YieldOp>(bodyBlock.getTerminator());

  // Replace expr results with yield values.
  for (auto [result, value] :
       llvm::zip(exprOp.getResults(), yieldOp.getValues()))
    result.replaceAllUsesWith(value);

  // Inline the body ops (except the yield) before the expr op.
  yieldOp->erase();
  parentBlock->getOperations().splice(exprOp->getIterator(),
                                      bodyBlock.getOperations());

  exprOp->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// uir.pin lowering
//===----------------------------------------------------------------------===//

/// Lower a uir.pin op by replacing its outputs with its inputs.
static LogicalResult lowerPinOp(PinOp pinOp) {
  for (auto [output, input] : llvm::zip(pinOp.getOutputs(), pinOp.getInputs()))
    output.replaceAllUsesWith(input);
  pinOp->erase();
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

    // Dissolve phase grouping ops first. These are structural markers that
    // SplitPhases used for phase analysis; after splitting they have no
    // runtime effect and just need to be inlined/removed.
    SmallVector<ExprOp> exprOps;
    moduleOp->walk([&](ExprOp op) { exprOps.push_back(op); });
    for (auto exprOp : exprOps) {
      if (failed(lowerExprOp(exprOp))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<PinOp> pinOps;
    moduleOp->walk([&](PinOp op) { pinOps.push_back(op); });
    for (auto pinOp : pinOps) {
      if (failed(lowerPinOp(pinOp))) {
        signalPassFailure();
        return;
      }
    }

    // Lower ifs first (innermost first via walk's post-order), then loops.
    // Ifs inside loop bodies get lowered before the enclosing loop.
    SmallVector<IfOp> ifOps;
    moduleOp->walk([&](IfOp op) { ifOps.push_back(op); });
    for (auto ifOp : ifOps) {
      if (failed(lowerIfOp(ifOp))) {
        signalPassFailure();
        return;
      }
    }

    SmallVector<LoopOp> loopOps;
    moduleOp->walk([&](LoopOp op) { loopOps.push_back(op); });
    for (auto loopOp : loopOps) {
      if (failed(lowerLoopOp(loopOp))) {
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
