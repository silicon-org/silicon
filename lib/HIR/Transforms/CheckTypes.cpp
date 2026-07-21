//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "llvm/ADT/TypeSwitch.h"
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
// Type Constructor and Constant Classification
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
  return isa<BoolTypeOp, UnitTypeOp, IntTypeOp, TypeTypeOp, UIntTypeOp,
             AnyfuncTypeOp, RefTypeOp, FuncTypeOp, OpaqueTypeOp>(defOp);
}

/// Check whether a value is produced by a constant op, i.e., an op whose
/// result value is fully determined by its attributes (`constant_int`,
/// `constant_bool`, `constant_unit`, `mir_constant`). Two such ops of the same
/// kind with different attributes can never unify.
static bool isConstantValueOp(Value value) {
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return false;
  return isa<ConstantIntOp, ConstantBoolOp, ConstantUnitOp, MIRConstantOp>(
      defOp);
}

/// Return a user-friendly name for a type value based on its defining op. For
/// example, `hir.int_type` becomes "`int`" and `hir.unit_type` becomes "`()`".
static std::string describeTypeValue(Value typeValue) {
  auto *defOp = typeValue.getDefiningOp();
  if (!defOp)
    return "(unknown)";

  if (isa<BoolTypeOp>(defOp))
    return "`bool`";
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
  if (isa<OpaqueTypeOp>(defOp))
    return "`opaque`";

  return "(unknown)";
}

/// Return a user-friendly rendering of a constant op's value for diagnostics,
/// e.g. "`8`" for `hir.constant_int 8` and "`true`" for `hir.constant_bool
/// <true>`. Other constant attributes print in their generic form.
static std::string describeConstantValue(Value value) {
  auto attr =
      llvm::TypeSwitch<Operation *, Attribute>(value.getDefiningOp())
          .Case<ConstantIntOp>([](auto op) { return op.getValueAttr(); })
          .Case<ConstantBoolOp>([](auto op) { return op.getValueAttr(); })
          .Case<MIRConstantOp>([](auto op) { return op.getValue(); })
          .Default([](auto) { return Attribute(); });

  std::string str;
  llvm::raw_string_ostream os(str);
  os << "`";
  if (auto intAttr = dyn_cast_or_null<base::IntAttr>(attr))
    os << intAttr.getValue();
  else if (auto boolAttr = dyn_cast_or_null<base::BoolAttr>(attr))
    os << (boolAttr.getValue() ? "true" : "false");
  else if (attr)
    os << attr;
  else
    os << "()";
  os << "`";
  return str;
}

namespace {
//===----------------------------------------------------------------------===//
// CheckTypes Pass
//===----------------------------------------------------------------------===//

/// Check for remaining `hir.unify` ops whose operands are provably unequal.
/// These represent type mismatches in the source program that type inference
/// could not resolve.
struct CheckTypesPass : public hir::impl::CheckTypesPassBase<CheckTypesPass> {
  void runOnOperation() override;
};
} // namespace

void CheckTypesPass::runOnOperation() {
  bool anyErrors = false;

  // Check whether two ops are structurally equivalent modulo their operands:
  // same op kind, same attributes. Operand mismatches are not compared here;
  // InferTypes decomposes them into their own operand-level unify ops.
  auto equivalentModuloOperands = [](Operation *lhsOp, Operation *rhsOp) {
    return OperationEquivalence::isEquivalentTo(
        lhsOp, rhsOp, OperationEquivalence::ignoreValueEquivalence, nullptr,
        OperationEquivalence::IgnoreLocations);
  };

  // Flag unify ops whose operands are provably unequal. Only provable
  // mismatches are reported; unify ops involving inferrables, coerce_type,
  // type_of, block arguments, or any other unresolved value are skipped,
  // since those may be resolved by later pipeline stages.
  getOperation()->walk([&](UnifyOp unifyOp) {
    auto lhs = unifyOp.getLhs();
    auto rhs = unifyOp.getRhs();
    if (lhs == rhs)
      return;
    auto *lhsOp = lhs.getDefiningOp();
    auto *rhsOp = rhs.getDefiningOp();

    // Two concrete type constructors are provably unequal if they are
    // structurally incompatible: different constructor kinds denote disjoint
    // types, and same-kind constructors with different attributes differ in
    // structure (e.g. function types of different arity). Same-kind
    // constructors with identical attributes are not flagged: their operand
    // pairs may still unify.
    if (isConcreteTypeConstructor(lhs) && isConcreteTypeConstructor(rhs)) {
      if (equivalentModuloOperands(lhsOp, rhsOp))
        return;
      emitError(unifyOp.getLoc())
          << "type mismatch: " << describeTypeValue(lhs)
          << " is not compatible with " << describeTypeValue(rhs);
      anyErrors = true;
      return;
    }

    // Two constants of the same kind with different value attributes are
    // provably unequal, e.g. mismatched `uint` widths after InferTypes has
    // decomposed the type unification down to the width operands. Constants
    // of different op kinds are skipped: different spellings may denote the
    // same value.
    if (isConstantValueOp(lhs) && isConstantValueOp(rhs) &&
        lhsOp->getName() == rhsOp->getName()) {
      if (equivalentModuloOperands(lhsOp, rhsOp))
        return;
      emitError(unifyOp.getLoc())
          << "type mismatch: constant " << describeConstantValue(lhs)
          << " is not compatible with constant " << describeConstantValue(rhs);
      anyErrors = true;
    }
  });

  // Check that integer literals have not been unified with non-integer types.
  // This catches cases like `fn foo() -> unit { 42 }` where context constrains
  // the literal to an incompatible type.
  getOperation()->walk([&](ConstantIntOp op) {
    auto typeVal = op.getTypeOperand();
    if (!isConcreteTypeConstructor(typeVal))
      return;
    if (isa<IntTypeOp, UIntTypeOp>(typeVal.getDefiningOp()))
      return;
    auto desc = describeTypeValue(typeVal);
    emitError(op.getLoc()) << "integer literal is not compatible with type "
                           << desc;
    anyErrors = true;
  });

  if (anyErrors)
    signalPassFailure();
}
