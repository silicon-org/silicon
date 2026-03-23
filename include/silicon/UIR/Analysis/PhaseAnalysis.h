//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/UIR/Ops.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace silicon {
namespace uir {

/// Phase analysis for unified functions (`uir.func`).
///
/// The analysis pushes "latest" phase constraints top-down from consumers to
/// producers via a use-def DFS. All op-level logic (constraint pushing,
/// tightening, pinned op handling, region entry, pure op scheduling) lives in
/// `resolveOp`. The `resolveValue` function is a thin wrapper that translates
/// value-level concerns (block args, call result offsets) to op-level
/// `resolveOp` calls, then checks the resolved phase against the consumer's
/// constraint.
///
/// See `docs/design/phase-inference.md` for the full specification.
///
/// Usage:
///   PhaseAnalysis analysis(funcOp);
///   if (failed(analysis.run()))
///     return signalPassFailure();  // diagnostics already emitted
///   // use analysis.opPhases / analysis.actualPhase
struct PhaseAnalysis {
  PhaseAnalysis(FuncOp funcOp) : funcOp(funcOp) {}

  /// Run the phase analysis. Returns failure if phase errors are detected
  /// (diagnostics already emitted via `mlir::emitError`).
  mlir::LogicalResult run();

  /// Look up the resolved phase for an op. Asserts if not found.
  int16_t getPhase(mlir::Operation *op) const;

  /// Look up the resolved phase for a value. Asserts if not found.
  int16_t getValuePhase(mlir::Value value) const;

  FuncOp funcOp;

  /// The resolved phase for each op (for the test pass annotation).
  mlir::DenseMap<mlir::Operation *, int16_t> opPhases;

  /// The resolved actual phase for each value.
  mlir::DenseMap<mlir::Value, int16_t> actualPhase;

private:
  bool anyErrors = false;

  /// Map from region-bearing op to the latest phase constraint for each of its
  /// block results. Populated before entering a region in `resolveOp`,
  /// consumed by terminators in `processBlock`.
  mlir::DenseMap<mlir::Operation *, llvm::SmallVector<int16_t>>
      resultConstraints;

  /// Process all ops in a block uniformly. Identifies roots (pins, pinned
  /// exprs, zero-use statements) and resolves them in block order, then
  /// dispatches on the block's terminator.
  void processBlock(mlir::Block &block, int16_t blockPhase);

  /// Thin value-level wrapper. Translates value constraints to op constraints
  /// (e.g., call result offset), delegates to `resolveOp`, then checks the
  /// resolved actual phase against `latest`. Returns the actual phase, or
  /// failure if the constraint is violated (error already emitted).
  mlir::FailureOr<int16_t> resolveValue(mlir::Value value, int16_t latest);

  /// Resolve an op at the given phase. Handles tightening, pinned ops,
  /// constant ops, type_of +1 shift, call arg offsets, type operand -1,
  /// region entry, and pure op earliest scheduling. The single place for all
  /// op-level phase resolution logic.
  void resolveOp(mlir::Operation *op, int16_t phase);

  /// Find the nearest enclosing op of a given type by walking up the parent
  /// chain. Asserts if not found.
  template <typename OpTy>
  OpTy findEnclosing(mlir::Operation *from);
};

} // namespace uir
} // namespace silicon
