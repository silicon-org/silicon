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
#include "mlir/IR/IRMapping.h"
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

/// Materialize a value at an insertion point by cloning its defining tree.
///
/// Walks the value's use-def tree and clones every op that does not already
/// properly dominate `point`, until reaching values that do. Only
/// memory-effect-free, region-free, single-result ops are cloned; such ops are
/// location-free, so materializing a copy at a dominating position preserves
/// semantics. Returns the materialized value, or null if the tree contains a
/// non-cloneable op (or block argument) that does not dominate `point`.
///
/// This makes type constraints from nested regions (structured control flow
/// arms, loop bodies, expression blocks) or later program points available at
/// the position of an inferrable: the type expression tree is hoisted to where
/// the inferrable is defined. Trees rooted in values that are genuinely local
/// to a region (e.g. loop iteration arguments) refuse to materialize and leave
/// the unification unresolved.
static Value materializeAtPoint(Value value, Operation *point,
                                DominanceInfo &domInfo) {
  if (domInfo.properlyDominates(value, point))
    return value;

  // Collect the ops that need cloning in post-order, such that cloning them in
  // order visits defs before uses.
  SmallVector<Operation *> toClone;
  SmallPtrSet<Operation *, 8> visited;
  std::function<bool(Value)> collect = [&](Value collectValue) {
    if (domInfo.properlyDominates(collectValue, point))
      return true;
    auto *defOp = collectValue.getDefiningOp();
    if (!defOp || !isMemoryEffectFree(defOp) || defOp->getNumResults() != 1 ||
        defOp->getNumRegions() != 0)
      return false;
    if (!visited.insert(defOp).second)
      return true;
    for (auto operand : defOp->getOperands())
      if (!collect(operand))
        return false;
    toClone.push_back(defOp);
    return true;
  };
  if (!collect(value))
    return {};

  OpBuilder builder(point);
  IRMapping mapping;
  for (auto *op : toClone)
    builder.clone(*op, mapping);
  return mapping.lookup(value);
}

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

    // If both operands are the same value, the unify is trivially resolved.
    if (unifyOp.getLhs() == unifyOp.getRhs()) {
      unifyOp.replaceAllUsesWith(unifyOp.getLhs());
      unifyOp.erase();
      continue;
    }

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
      if (domInfo.properlyDominates(concrete, inferrable)) {
        LLVM_DEBUG(llvm::dbgs() << "Replacing " << inferrable << " with "
                                << concrete << "\n");
        inferrable.replaceAllUsesWith(concrete);
        inferrable.erase();
        unifyOp.replaceAllUsesWith(concrete);
        unifyOp.erase();
        continue;
      }

      // If the concrete doesn't dominate the inferrable, try to materialize
      // its defining tree at the inferrable's location. This handles the
      // pattern where type constructors after the inferrable or inside nested
      // regions (branch arms, loop bodies, expression blocks) constrain an
      // inferrable in a dominating position.
      if (auto cloned = materializeAtPoint(concrete, inferrable, domInfo)) {
        LLVM_DEBUG(llvm::dbgs() << "Materialized " << cloned << " to resolve "
                                << inferrable << "\n");
        inferrable.replaceAllUsesWith(cloned);
        inferrable.erase();
        unifyOp.replaceAllUsesWith(cloned);
        unifyOp.erase();
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Skipping " << inferrable
                              << " (appears before " << concrete << ")\n");
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
    if (domInfo.properlyDominates(eraseOp, keepOp))
      std::swap(keepOp, eraseOp);

    // If all operands of the erased op dominate the op we keep, we can merge
    // the two ops in place: insert operand unifications before the kept op,
    // replace the erased op with the kept one.
    if (llvm::all_of(eraseOp->getOperands(), [&](auto value) {
          return domInfo.properlyDominates(value, keepOp);
        })) {
      LLVM_DEBUG(llvm::dbgs() << "Unifying " << *keepOp << " (kept) and "
                              << *eraseOp << " (erased)\n");

      OpBuilder builder(keepOp);
      for (auto [keepArg, eraseArg] :
           llvm::zip(keepOp->getOpOperands(), eraseOp->getOperands())) {
        if (keepArg.get() == eraseArg)
          continue;
        auto unified =
            UnifyOp::create(builder, unifyOp.getLoc(), eraseArg.getType(),
                            keepArg.get(), eraseArg);
        keepArg.set(unified);
        worklist.push_back(unified);
      }
      eraseOp->replaceAllUsesWith(keepOp);
      eraseOp->erase();
      unifyOp.replaceAllUsesWith(keepOp);
      unifyOp.erase();
      continue;
    }

    // Otherwise, the erased op's operands are defined after the kept op (e.g.,
    // a const block computation feeding a uint_type). We can't merge in place,
    // but we can decompose the type-level unification into operand-level
    // unifications at the unify op's location, where both ops' operands are
    // available. This turns `unify(T<a>, T<b>)` into `T<unify(a, b)>`.
    LLVM_DEBUG(llvm::dbgs()
               << "Decomposing " << unifyOp << " into operand unifications\n");

    OpBuilder builder(unifyOp);
    SmallVector<Value> newOperands;
    for (auto [keepArg, eraseArg] :
         llvm::zip(keepOp->getOperands(), eraseOp->getOperands())) {
      if (keepArg == eraseArg) {
        newOperands.push_back(keepArg);
        continue;
      }
      auto unified = UnifyOp::create(builder, unifyOp.getLoc(),
                                     keepArg.getType(), keepArg, eraseArg);
      newOperands.push_back(unified);
      worklist.push_back(unified);
    }
    auto *newOp = builder.clone(*keepOp);
    for (auto [newArg, newVal] : llvm::zip(newOp->getOpOperands(), newOperands))
      newArg.set(newVal);
    unifyOp.replaceAllUsesWith(newOp->getResult(0));
    unifyOp.erase();
  }

  // Fall back to int_type for any constant_int ops whose type operand is still
  // an unconstrained inferrable. This handles literals like `42` that appear
  // without any context to constrain their type.
  getOperation()->walk([&](ConstantIntOp op) {
    if (auto inferrable = op.getTypeOperand().getDefiningOp<InferrableOp>()) {
      OpBuilder builder(inferrable);
      auto intType = IntTypeOp::create(builder, inferrable.getLoc());
      inferrable.replaceAllUsesWith(intType.getResult());
      inferrable.erase();
    }
  });
}
