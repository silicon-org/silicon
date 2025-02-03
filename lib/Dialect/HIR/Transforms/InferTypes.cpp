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
  LLVM_DEBUG(llvm::dbgs() << "Starting with " << worklist.size()
                          << " initial unify ops\n");

  while (!worklist.empty()) {
    auto unifyOp = worklist.pop_back_val();
    LLVM_DEBUG(llvm::dbgs() << "Processing " << unifyOp << "\n");

    auto lhs = unifyOp.getLhs().getDefiningOp<InferrableTypeOp>();
    auto rhs = unifyOp.getRhs().getDefiningOp<InferrableTypeOp>();
    if (!lhs || !rhs)
      continue;

    auto keepOp = lhs;
    auto eraseOp = rhs;
    if (!keepOp->isBeforeInBlock(eraseOp)) {
      keepOp = rhs;
      eraseOp = lhs;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "Keeping " << keepOp << ", erasing " << eraseOp << "\n");
    eraseOp.replaceAllUsesWith(keepOp.getResult());
    eraseOp.erase();
    unifyOp.replaceAllUsesWith(keepOp.getResult());
    unifyOp.erase();
  }
}
