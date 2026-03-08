//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "check-types"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_CHECKTYPESPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

//===----------------------------------------------------------------------===//
// Type Constructor Classification
//===----------------------------------------------------------------------===//

/// Check whether a value is produced by a concrete type constructor op, i.e.,
/// an op that directly represents a specific Silicon type (like `int_type`,
/// `unit_type`, etc.). Values produced by other ops (like `coerce_type`,
/// `type_of`, or block arguments) are not considered concrete type constructors
/// because they may be resolved to a specific type later in the pipeline.
static bool isConcreteTypeConstructor(Value typeValue) {
  auto *defOp = typeValue.getDefiningOp();
  if (!defOp)
    return false;
  return isa<UnitTypeOp, IntTypeOp, TypeTypeOp, UIntTypeOp, AnyfuncTypeOp,
             RefTypeOp, FuncTypeOp, OpaqueTypeOp>(defOp);
}

/// Return a user-friendly name for a type value based on its defining op. For
/// example, `hir.int_type` becomes "`int`" and `hir.unit_type` becomes "`()`".
static std::string describeTypeValue(Value typeValue) {
  auto *defOp = typeValue.getDefiningOp();
  if (!defOp)
    return "(unknown)";

  if (isa<UnitTypeOp>(defOp))
    return "`()`";
  if (isa<IntTypeOp>(defOp))
    return "`int`";
  if (isa<TypeTypeOp>(defOp))
    return "`type`";
  if (isa<UIntTypeOp>(defOp))
    return "`uint`";
  if (isa<AnyfuncTypeOp>(defOp))
    return "`anyfunc`";
  if (isa<RefTypeOp>(defOp))
    return "`ref`";
  if (isa<FuncTypeOp>(defOp))
    return "a function type";

  return "(unknown)";
}

namespace {
//===----------------------------------------------------------------------===//
// CheckTypes Pass
//===----------------------------------------------------------------------===//

/// Check for remaining `hir.unify` ops with two different concrete type
/// constructor operands. These represent type mismatches in the source program
/// that type inference could not resolve.
struct CheckTypesPass : public hir::impl::CheckTypesPassBase<CheckTypesPass> {
  void runOnOperation() override;
};
} // namespace

void CheckTypesPass::runOnOperation() {
  bool anyErrors = false;

  getOperation()->walk([&](UnifyOp unifyOp) {
    auto lhs = unifyOp.getLhs();
    auto rhs = unifyOp.getRhs();

    // Only flag mismatches between two known, different concrete type
    // constructors. Skip unify ops involving inferrables, coerce_type,
    // type_of, block arguments, or any other non-constructor value, since
    // those may be resolved by later pipeline stages.
    if (lhs == rhs)
      return;
    if (!isConcreteTypeConstructor(lhs) || !isConcreteTypeConstructor(rhs))
      return;

    auto lhsDesc = describeTypeValue(lhs);
    auto rhsDesc = describeTypeValue(rhs);
    emitError(unifyOp.getLoc()) << "type mismatch: " << lhsDesc
                                << " is not compatible with " << rhsDesc;
    anyErrors = true;
  });

  if (anyErrors)
    signalPassFailure();
}
