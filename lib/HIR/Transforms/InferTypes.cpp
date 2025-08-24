//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "mlir/IR/Dominance.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "infer-types"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_INFERTYPESPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

namespace {
struct InferTypesPass : public hir::impl::InferTypesPassBase<InferTypesPass> {
  void runOnOperation() override;
};
} // namespace

void InferTypesPass::runOnOperation() {
  SmallVector<UnifyOp> worklist;
  getOperation()->walk([&](UnifyOp op) { worklist.push_back(op); });
  std::reverse(worklist.begin(), worklist.end());
  LLVM_DEBUG(llvm::dbgs() << "Starting with " << worklist.size()
                          << " initial unify ops\n");

  auto &domInfo = getAnalysis<DominanceInfo>();

  while (!worklist.empty()) {
    auto unifyOp = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Processing " << unifyOp << "\n");

    auto inferrableLhs = unifyOp.getLhs().getDefiningOp<InferrableOp>();
    auto inferrableRhs = unifyOp.getRhs().getDefiningOp<InferrableOp>();

    // In case both operands are inferrable, we pick the earlier one in the IR
    // and replace all uses of the later one.
    if (inferrableLhs && inferrableRhs) {
      auto keepOp = inferrableLhs;
      auto eraseOp = inferrableRhs;
      if (domInfo.properlyDominates(eraseOp.getOperation(), keepOp)) {
        keepOp = inferrableRhs;
        eraseOp = inferrableLhs;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Keeping " << keepOp << ", erasing " << eraseOp << "\n");
      eraseOp.replaceAllUsesWith(keepOp.getResult());
      eraseOp.erase();
      unifyOp.replaceAllUsesWith(keepOp.getResult());
      unifyOp.erase();
      continue;
    }

    // In case one of the operands is inferrable, try to replace all uses of it
    // with the other operand.
    if (inferrableLhs || inferrableRhs) {
      auto inferrable = inferrableLhs ? inferrableLhs : inferrableRhs;
      auto concrete = inferrableLhs ? unifyOp.getRhs() : unifyOp.getLhs();
      if (!domInfo.properlyDominates(concrete, inferrable)) {
        LLVM_DEBUG(llvm::dbgs() << "Skipping " << inferrable
                                << " (appears before " << concrete << ")\n");
        continue;
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "Replacing " << inferrable << " with " << concrete << "\n");
      inferrable.replaceAllUsesWith(concrete);
      inferrable.erase();
      unifyOp.replaceAllUsesWith(concrete);
      unifyOp.erase();
      continue;
    }

    // If both operands are defined by the same kind of operation, replace the
    // later with the earlier one and unify their operands.
    auto *lhsOp = unifyOp.getLhs().getDefiningOp();
    auto *rhsOp = unifyOp.getRhs().getDefiningOp();
    if (!lhsOp || !rhsOp)
      continue;
    if (lhsOp->getNumRegions() != 0 || lhsOp->getNumSuccessors() != 0 ||
        lhsOp->getNumResults() != 1)
      continue;
    if (!isMemoryEffectFree(lhsOp))
      continue;
    if (!OperationEquivalence::isEquivalentTo(
            lhsOp, rhsOp, OperationEquivalence::ignoreValueEquivalence, nullptr,
            OperationEquivalence::IgnoreLocations))
      continue;

    // Pick which one of the ops to keep.
    auto *keepOp = lhsOp;
    auto *eraseOp = rhsOp;
    if (domInfo.properlyDominates(eraseOp, keepOp)) {
      keepOp = inferrableRhs;
      eraseOp = inferrableLhs;
    }

    // Ensure that all operands of the erased op dominate the op we keep, such
    // that we can introduce unification ops for the operands in front of the op
    // we keep.
    if (!llvm::all_of(eraseOp->getOperands(), [&](auto value) {
          return domInfo.properlyDominates(value, keepOp);
        }))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Unifying " << *keepOp << " (kept) and "
                            << *eraseOp << " (erased)\n");

    // Create unify ops for each of the operands.
    OpBuilder builder(keepOp);
    for (auto [keepArg, eraseArg] :
         llvm::zip(keepOp->getOpOperands(), eraseOp->getOperands())) {
      auto unified = builder.create<UnifyOp>(
          unifyOp.getLoc(), eraseArg.getType(), keepArg.get(), eraseArg);
      keepArg.set(unified);
      worklist.push_back(unified);
    }
    eraseOp->replaceAllUsesWith(keepOp);
    eraseOp->erase();
    unifyOp.replaceAllUsesWith(keepOp);
    unifyOp.erase();
  }
}
