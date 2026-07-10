//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/Support/MLIR.h"
#include "silicon/UIR/Ops.h"

namespace silicon {
namespace uir {

/// Phase analysis for unified functions (`uir.func`).
///
/// The analysis assigns a phase to every op and value in a `uir.func` using
/// three DFS flow patterns:
///
/// 1. **Pure ops:** demand flows down through operands, then the op schedules
///    at `earliest = max(operand phases)`.
/// 2. **Side-effecting ops:** demand goes UP to the parent block, the block
///    adjusts its phase, then pushes DOWN to all ops.
/// 3. **Region op results:** demand passes through the yield transparently
///    to the inner value. When a merging CF op (if/loop) selects between
///    distinct values, the result phase is the max over the contributing
///    values, floored at the op's phase. A result phase *later* than the op
///    phase is a choice-ification obligation: SortPhases derives those
///    directly from the phase map and reconstructs the selection with
///    `hir.pick` in the receiving phase.
///
/// Two sentinel phases sit below all real phases and satisfy any demand:
/// `kFloatPhase` marks pure values that can materialize at any phase, and
/// `kNeverPhase` (bottom, like Rust's `!`) marks CF results that no
/// control-flow path provides — such a value never exists at runtime, so
/// every consumer is unreachable and no phase requirement can bind it.
///
/// The algorithm has five steps:
/// 1. Pre-walk: collect terminators (break/continue per loop,
///    return/signature ops).
/// 2. Constrain blocks containing function-interface terminators to phase ≤ 0.
/// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
/// 4. Demand-driven constraints: push from declared arg/result phases. Then
///    default undemanded floating exprs with anchored content (side effects,
///    pins) to their block phase — effects execute where they stand when no
///    consumer pulls them elsewhere; dead pure exprs stay phaseless.
/// 5. Final recompute: reschedule pure ops to max(operand phases) and refresh
///    recorded region result phases from the final value and op phases (the
///    DFS records demand-time phases, which may be looser).
///
/// Usage:
///   PhaseAnalysis analysis(funcOp);
///   if (failed(analysis.run()))
///     return signalPassFailure();  // diagnostics already emitted
///   // use analysis.opPhases / analysis.actualPhase
struct PhaseAnalysis {
  /// Sentinel phase for pure values that float to any demanded phase.
  static constexpr int16_t kFloatPhase = INT16_MIN;

  /// Sentinel phase (bottom) for values no control-flow path ever produces,
  /// such as the result of a loop without breaks or of an if whose branches
  /// all return. Satisfies any demand; consumers are unreachable at runtime.
  static constexpr int16_t kNeverPhase = INT16_MIN + 1;

  PhaseAnalysis(FuncOp funcOp) : funcOp(funcOp) {}

  /// Run the phase analysis. Returns failure if phase errors are detected
  /// (diagnostics already emitted via `mlir::emitError`).
  LogicalResult run();

  /// Look up the resolved phase for an op. Asserts if not found.
  int16_t getPhase(Operation *op) const;

  /// Look up the resolved phase for a value. Asserts if not found.
  int16_t getValuePhase(Value value) const;

  FuncOp funcOp;

  /// The resolved phase for each op (for the test pass annotation).
  DenseMap<Operation *, int16_t> opPhases;

  /// The resolved actual phase for each value.
  DenseMap<Value, int16_t> actualPhase;

private:
  /// Debug indentation depth for tracing the DFS.
  unsigned depth = 0;

  /// Break ops for each loop, and the reverse mapping.
  DenseMap<LoopOp, SmallVector<BreakOp>> loopBreaks;
  DenseMap<BreakOp, LoopOp> breakToLoop;

  /// Continue ops mapped to their enclosing loop.
  DenseMap<ContinueOp, LoopOp> continueToLoop;

  /// All ReturnOp and SignatureOp terminators in the function.
  SmallVector<Operation *> funcTerminators;

  /// Walk the function IR and populate the terminator mappings above.
  void collectTerminators();

  /// Consumer demands value at phase ≤ latest. Dispatches based on the value's
  /// defining op type (block arg, constant, pure, region, call,
  /// side-effecting). Returns the resolved phase, or failure if the demand
  /// cannot be satisfied.
  FailureOr<int16_t> constrainValue(Value value, int16_t latest);

  /// Someone needs this block at phase ≤ demanded. Dispatches on the parent op
  /// (floating expr, pinned expr, anchored CF, function body). Returns the
  /// resolved block phase.
  FailureOr<int16_t> constrainBlock(Block &block, int16_t demandedPhase);

  /// Push demand through yield/break to a specific result of a region-bearing
  /// op. The constraint forwards through the terminator operands; the result
  /// phase is recorded as the max over the contributing values, floored at
  /// the op phase when the op selects between distinct values. The recorded
  /// phase uses demand-time value phases; the final recompute in `run()`
  /// refreshes it once all phases are settled.
  LogicalResult constrainRegionResult(Operation *regionOp, unsigned resultIdx,
                                      int16_t latest);

  /// Collect the (value, terminator) pairs providing a given result of a
  /// region-bearing op: yields for exprs/ifs, breaks for loops. Terminators
  /// that do not carry the result are omitted.
  SmallVector<std::pair<Value, Operation *>, 2>
  resultProviders(Operation *regionOp, unsigned resultIdx);

  /// Whether a set of result providers selects between distinct SSA values.
  static bool
  selectsDistinctValues(ArrayRef<std::pair<Value, Operation *>> providers);

  /// Push blockPhase down to all ops in a block (including the terminator).
  LogicalResult processBlock(Block &block, int16_t blockPhase);

  /// Assign phase to a single op and push constraints to operands/regions.
  /// Skips floating exprs, pure ops, and constants (demand-driven only).
  /// Validates terminator phase constraints (return/break/continue/signature
  /// must be at the correct phase).
  LogicalResult processOp(Operation *op, int16_t blockPhase);
};

} // namespace uir
} // namespace silicon
