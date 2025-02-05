//===- InferTypes.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIROps.h"
#include "silicon/Dialect/HIR/HIRPasses.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "infer-types"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_INFERTYPESPASS
#include "silicon/Dialect/HIR/HIRPasses.h.inc"
} // namespace hir
} // namespace silicon

namespace {
struct InferTypesPass : public hir::impl::InferTypesPassBase<InferTypesPass> {
  void runOnOperation() override;
};
} // namespace

void InferTypesPass::runOnOperation() {
  SmallVector<UnifyTypeOp> worklist;
  getOperation()->walk([&](UnifyTypeOp op) { worklist.push_back(op); });
  std::reverse(worklist.begin(), worklist.end());
  LLVM_DEBUG(llvm::dbgs() << "Starting with " << worklist.size()
                          << " initial unify ops\n");

  while (!worklist.empty()) {
    auto unifyOp = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Processing " << unifyOp << "\n");

    auto inferrableLhs = unifyOp.getLhs().getDefiningOp<InferrableTypeOp>();
    auto inferrableRhs = unifyOp.getRhs().getDefiningOp<InferrableTypeOp>();

    // In case both operands are inferrable, we pick the earlier one in the IR
    // and replace all uses of the later one.
    if (inferrableLhs && inferrableRhs) {
      auto keepOp = inferrableLhs;
      auto eraseOp = inferrableRhs;
      if (!keepOp->isBeforeInBlock(eraseOp)) {
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
      auto *concreteOp = concrete.getDefiningOp();
      if (concreteOp && !concreteOp->isBeforeInBlock(inferrable)) {
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
  }
}
