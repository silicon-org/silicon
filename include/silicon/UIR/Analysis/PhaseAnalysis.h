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
///    to the inner value.
///
/// The algorithm has four steps:
/// 1. Pre-walk: collect terminators (break/continue per loop,
///    return/signature ops).
/// 2. Constrain blocks containing function-interface terminators to phase ≤ 0.
/// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
/// 4. Demand-driven constraints: push from declared arg/result phases.
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

  /// Sentinel value for unconstrained phases.
  static constexpr int16_t kUnconstrained = INT16_MAX;

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
  bool anyErrors = false;

  /// Debug indentation depth for tracing the DFS.
  unsigned depth = 0;

  /// Map from region-bearing op to the latest phase constraint for each of its
  /// block results. Populated by constrainRegionResult, consumed by yield/break
  /// handlers.
  DenseMap<Operation *, SmallVector<int16_t>> resultConstraints;

  //===--------------------------------------------------------------------===//
  // Pre-collected Terminator Mappings
  //
  // Built by collectTerminators() in a single manual walk with loop context.
  // Used to jump between terminators and their parent ops without walking the
  // parent chain at each use.
  //===--------------------------------------------------------------------===//

  /// Break ops for each loop, and the reverse mapping.
  DenseMap<LoopOp, SmallVector<BreakOp>> loopBreaks;
  DenseMap<BreakOp, LoopOp> breakToLoop;

  /// Continue ops mapped to their enclosing loop.
  DenseMap<ContinueOp, LoopOp> continueToLoop;

  /// All ReturnOp and SignatureOp terminators in the function.
  SmallVector<Operation *> funcInterfaceTerminators;

  /// Walk the function IR and populate the terminator mappings above. Uses
  /// manual recursion with a `currentLoop` context to avoid repeated parent
  /// chain walks.
  void collectTerminators();

  //===--------------------------------------------------------------------===//
  // Five Core Functions
  //===--------------------------------------------------------------------===//

  /// Consumer demands value at phase ≤ latest. Dispatches based on the value's
  /// defining op type (block arg, constant, pure, region, call,
  /// side-effecting).
  FailureOr<int16_t> constrainValue(Value value, int16_t latest);

  /// Someone needs this block at phase ≤ demanded. Dispatches on the parent op
  /// (floating expr, pinned expr, anchored CF, function body).
  void constrainBlock(Block &block, int16_t demandedPhase);

  /// Push demand through yield/break to a specific result of a region-bearing
  /// op (ExprOp, IfOp, LoopOp). The constraint propagates to the corresponding
  /// yield/break operand. The parent result's actualPhase is updated to match.
  void constrainRegionResult(Operation *regionOp, unsigned resultIdx,
                             int16_t latest);

  /// Push blockPhase down to all ops in the block. Called when block phase is
  /// known or has changed. Validates terminator phase equalities.
  void processBlock(Block &block, int16_t blockPhase);

  /// Called by processBlock for each non-terminator op. Sets the op's phase
  /// and pushes constraints to operands/regions. Skips floating/pure/constant
  /// ops (demand-driven only).
  void processOp(Operation *op, int16_t blockPhase);
};

} // namespace uir
} // namespace silicon
