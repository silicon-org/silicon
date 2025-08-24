//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "eval-consts"

using namespace mlir;
using namespace silicon;
using namespace hir;
using llvm::MapVector;

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_EVALCONSTSPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

static Block *cloneBlock(Block *block, PatternRewriter &rewriter) {
  IRMapping mapper;
  auto *clonedBlock = rewriter.createBlock(block);
  for (auto oldArg : block->getArguments()) {
    auto newArg = clonedBlock->addArgument(oldArg.getType(), oldArg.getLoc());
    mapper.map(oldArg, newArg);
  }
  for (auto &op : *block)
    rewriter.clone(op, mapper);
  return clonedBlock;
}

namespace {
struct ConstBranchOpPattern : public OpRewritePattern<ConstBranchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstBranchOp op,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(llvm::dbgs() << "Rewriting " << op << "\n");

    // Clone the destination block if this branch isn't its only use.
    auto *destBlock = op.getDest();
    if (!destBlock->hasOneUse())
      destBlock = cloneBlock(destBlock, rewriter);

    // Remove the branch and append the destination block.
    auto *thisBlock = op->getBlock();
    SmallVector<Value> destOperands = op.getDestOperands();
    rewriter.eraseOp(op);
    rewriter.mergeBlocks(destBlock, thisBlock, destOperands);
    return success();
  }
};

struct ConstCondBranchOpPattern : public OpRewritePattern<ConstCondBranchOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ConstCondBranchOp op,
                                PatternRewriter &rewriter) const override {
    // Extract the condition value if it is a constant.
    APInt condition;
    if (auto *defOp = op.getCondition().getDefiningOp();
        !defOp || !m_ConstantInt(&condition).match(defOp))
      return failure();

    LLVM_DEBUG(llvm::dbgs()
               << "Rewriting " << op << " (condition = " << condition << ")\n");

    // Clone the destination block if this branch isn't its only use.
    auto *destBlock = condition.isOne() ? op.getTrueDest() : op.getFalseDest();
    if (!destBlock->hasOneUse())
      destBlock = cloneBlock(destBlock, rewriter);

    // Remove the branch and append the destination block.
    auto *thisBlock = op->getBlock();
    SmallVector<Value> destOperands =
        condition.isOne() ? op.getTrueOperands() : op.getFalseOperands();
    rewriter.eraseOp(op);
    rewriter.mergeBlocks(destBlock, thisBlock, destOperands);
    return success();
  }
};
} // namespace

namespace {
struct EvalConstsPass : public hir::impl::EvalConstsPassBase<EvalConstsPass> {
  void runOnOperation() override;
  void promoteToBlockArguments(Region &region, Liveness &liveness,
                               DominanceInfo &dominance);
  void promoteToBlockArguments(Region &region, Value value,
                               ArrayRef<Operation *> branches,
                               Liveness &liveness, DominanceInfo &dominance);
};
} // namespace

void EvalConstsPass::runOnOperation() {
  auto *context = &getContext();

  // Promote values that are live across constant branches to block arguments.
  // This prevents control flow unrolling from creating dominance violations.
  auto &dominance = getAnalysis<DominanceInfo>();
  auto &liveness = getAnalysis<Liveness>();
  for (auto &region : getOperation()->getRegions())
    promoteToBlockArguments(region, liveness, dominance);

  // Unroll constant branches.
  RewritePatternSet patterns(context);
  for (auto *dialect : context->getLoadedDialects())
    dialect->getCanonicalizationPatterns(patterns);
  for (auto op : context->getRegisteredOperations())
    op.getCanonicalizationPatterns(patterns, context);
  patterns.add<ConstBranchOpPattern>(context);
  patterns.add<ConstCondBranchOpPattern>(context);

  GreedyRewriteConfig config;
  config.setRegionSimplificationLevel(GreedySimplifyRegionLevel::Normal);

  if (failed(
          applyPatternsGreedily(getOperation(), std::move(patterns), config)))
    return signalPassFailure();
}

void EvalConstsPass::promoteToBlockArguments(Region &region, Liveness &liveness,
                                             DominanceInfo &dominance) {
  if (region.hasOneBlock())
    return;

  // Collect all values in blocks targeted by constant branches which escape the
  // block they are defined in. These will have to become block arguments to
  // allow for branch unrolling to replicate these blocks and values.
  MapVector<Value, SmallVector<Operation *, 4>> promotableValues;
  for (auto &block : region) {
    if (!llvm::any_of(block.getPredecessors(), [&](Block *predecessor) {
          return isa<ConstBranchOp, ConstCondBranchOp>(
              predecessor->getTerminator());
        }))
      continue;
    auto *blockLiveness = liveness.getLiveness(&block);
    for (auto value : block.getArguments())
      if (blockLiveness->isLiveOut(value))
        promotableValues[value].push_back(block.getTerminator());
    for (auto &op : block)
      for (auto result : op.getResults())
        if (blockLiveness->isLiveOut(result))
          promotableValues[result].push_back(block.getTerminator());
  }

  LLVM_DEBUG(llvm::dbgs() << "Promoting " << promotableValues.size()
                          << " values to block arguments\n");
  for (auto &[value, branches] : promotableValues)
    promoteToBlockArguments(region, value, branches, liveness, dominance);
}

void EvalConstsPass::promoteToBlockArguments(Region &region, Value value,
                                             ArrayRef<Operation *> branches,
                                             Liveness &liveness,
                                             DominanceInfo &dominance) {
  LLVM_DEBUG({
    llvm::dbgs() << "Capture " << value << "\n";
    for (auto *branch : branches)
      llvm::dbgs() << "- Across " << *branch << "\n";
  });

  // Calculate the merge points for this value once it gets promoted to block
  // arguments across the branches.
  auto &domTree = dominance.getDomTree(&region);
  llvm::IDFCalculatorBase<Block, false> idfCalculator(domTree);

  // Calculate the set of blocks which will contain a distinct definition of
  // this value.
  SmallPtrSet<Block *, 4> definingBlocks;
  definingBlocks.insert(value.getParentBlock());
  for (auto *branch : branches)
    for (auto *dest : branch->getSuccessors())
      if (liveness.getLiveIn(dest).contains(value))
        definingBlocks.insert(dest);
  idfCalculator.setDefiningBlocks(definingBlocks);

  // Calculate where the value is live.
  SmallPtrSet<Block *, 16> liveInBlocks;
  for (auto &block : region)
    if (liveness.getLiveness(&block)->isLiveIn(value))
      liveInBlocks.insert(&block);
  idfCalculator.setLiveInBlocks(liveInBlocks);

  // Calculate the merge points where we will have to insert block arguments for
  // this value.
  SmallVector<Block *> mergePointsVec;
  idfCalculator.calculate(mergePointsVec);
  SmallPtrSet<Block *, 16> mergePoints(mergePointsVec.begin(),
                                       mergePointsVec.end());
  for (auto *block : definingBlocks)
    if (block != value.getParentBlock())
      mergePoints.insert(block);
  LLVM_DEBUG(llvm::dbgs() << "- " << mergePoints.size() << " merge points\n");

  // Perform a depth-first search starting at the block containing the value,
  // which dominates all its uses. When we encounter a block that is a merge
  // point, insert a block argument.
  struct WorklistItem {
    DominanceInfoNode *domNode;
    Value reachingDef;
  };
  SmallVector<WorklistItem> worklist;
  worklist.push_back({domTree.getNode(value.getParentBlock()), value});

  while (!worklist.empty()) {
    auto item = worklist.pop_back_val();
    auto *block = item.domNode->getBlock();

    // If this block is a merge point, insert a block argument for the value.
    if (mergePoints.contains(block))
      item.reachingDef = block->addArgument(value.getType(), value.getLoc());

    // Replace any uses of the value in this block with the current reaching
    // definition.
    for (auto &op : *block)
      op.replaceUsesOfWith(value, item.reachingDef);

    // If the terminator of this block branches to a merge point, add the
    // current reaching definition as a destination operand.
    if (auto branchOp = dyn_cast<BranchOpInterface>(block->getTerminator()))
      for (auto &blockOperand : branchOp->getBlockOperands())
        if (mergePoints.contains(blockOperand.get()))
          branchOp.getSuccessorOperands(blockOperand.getOperandNumber())
              .append(item.reachingDef);

    for (auto *child : item.domNode->children())
      worklist.push_back({child, item.reachingDef});
  }
}
