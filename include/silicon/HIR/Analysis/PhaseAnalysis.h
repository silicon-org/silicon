//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/HIR/Ops.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseMap.h"

namespace silicon {
namespace hir {

/// Compute the earliest available phase for each op and value in a unified
/// function. This uses a forward PreOrder walk: each op's phase is determined
/// by its parent's phase and its operands' phases (for pure ops), or just the
/// parent's phase (for side-effecting ops). ExprOps with a non-zero
/// `phaseShift` attribute shift relative to their parent.
///
/// Constants and other pure ops with no operands get INT16_MIN, meaning they
/// float to whatever phase needs them. This is clipped to minPhase during
/// splitting.
struct PhaseAnalysis {
  PhaseAnalysis(UnifiedFuncOp funcOp) : funcOp(funcOp) {}

  /// Run the forward phase analysis, computing phases for all ops and values.
  void analyze();

  /// Back-propagate phases: pull values to earlier phases when required by
  /// call argument constraints and all transitive dependencies allow it.
  void pullPhases();

  /// Re-run forward phase computation to propagate pulled phases to users.
  /// Takes the min of the existing and recomputed phase, preserving any
  /// pull-induced phases while propagating their effects forward.
  void refreshPhases();

  /// Look up the earliest phase at which a value is available.
  int16_t getValuePhase(mlir::Value value) const;

  /// Check that unified_call arguments are available at their required phases.
  mlir::LogicalResult checkCallArgPhases();

  UnifiedFuncOp funcOp;
  mlir::DenseMap<mlir::Operation *, int16_t> opPhases;

  /// Phases for all values: body block args and op results.
  mlir::DenseMap<mlir::Value, int16_t> valuePhases;

private:
  /// Try to pull a value (and its transitive dependencies) to an earlier phase.
  /// Returns true if the pull succeeded.
  bool pullValueToPhase(mlir::Value value, int16_t targetPhase);

  /// Re-compute phases for all ops inside a region, using `floor` as the
  /// minimum phase. This mirrors the forward analysis logic but with an updated
  /// floor.
  void recomputeRegionPhases(mlir::Region &region, int16_t floor);
};

} // namespace hir
} // namespace silicon
