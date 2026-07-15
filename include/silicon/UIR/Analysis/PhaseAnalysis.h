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
///    at `earliest = max(operand phases)`. A pure op imposes no latest bound
///    on itself: undemanded pure ops have latest = +inf and derive the same
///    earliest phase from their operands in the final recompute, so deleting
///    a value's last use never invalidates a program. Undemanded floating
///    exprs derive exactly like pure ops, with the lower bounds their
///    anchored contents impose (CF conditions and inits consumed at the
///    frame phase, type operands one phase earlier) playing the role of
///    operand phases; the contents then anchor at the derived phase, so a
///    dead expr resolves to the phase its live twin would occupy under the
///    earliest possible demand.
/// 2. **Side-effecting ops:** demand goes UP to the parent block, the block
///    adjusts its phase, then pushes DOWN to all ops.
/// 3. **Region op results:** demand passes through the yield transparently
///    to the inner value. When more than one terminator of a merging CF op
///    (if/loop) provides a result, the op selects between the provided
///    values — regardless of their identity — and the result phase is the
///    max over the contributing values, floored at the op's phase. A result
///    phase *later* than the op phase is a choice-ification obligation:
///    SortPhases derives those directly from the phase map and reconstructs
///    the selection with `hir.pick` in the receiving phase.
///
/// Two sentinel phases sit below all real phases and satisfy any demand:
/// `kFloatPhase` marks pure values that can materialize at any phase —
/// operand-free pure trees such as constants and arithmetic on constants,
/// which consumers rematerialize per use — and `kNeverPhase` (bottom, like
/// Rust's `!`) marks CF results that no control-flow path provides — such a
/// value never exists at runtime, so every consumer is unreachable and no
/// phase requirement can bind it.
///
/// The algorithm has six steps:
/// 1. Pre-walk: collect terminators (break/continue per loop,
///    return/signature ops).
/// 2. Constrain blocks containing function-interface terminators to phase ≤ 0.
/// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
/// 4. Demand-driven constraints: push from declared arg/result phases. Then
///    default undemanded floating exprs with anchored content (side effects,
///    pins) to their block phase — effects execute where they stand when no
///    consumer pulls them elsewhere.
/// 5. Final recompute: reschedule visited pure ops to max(operand phases)
///    (the DFS records demand-time phases, which may be looser), derive the
///    same earliest phase for pure ops no demand ever reached (when some
///    operand contributes a real phase), derive undemanded floating exprs
///    from the bounds their anchored contents impose and anchor those
///    contents at the derived phase, and refresh or fill in region result
///    phases from their providing terminator values — so demand never
///    influences where a pure op, expr, or region result materializes.
/// 6. Coverage stamp: anything still unresolved — operand-free dead pure
///    trees, dead terminators, dead block arguments — gets the float
///    sentinel, in both `opPhases` and `actualPhase`. Afterwards every op,
///    result, and block argument the analysis models has an entry; a
///    missing entry means the op or value sits inside a region the analysis
///    does not model (e.g. `hir.pick`/`hir.replicate` interiors), which
///    consumers must treat as an error.
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

  /// Whether the analysis models the regions of this op. Contents of any
  /// other op's regions receive no phase entries; after the coverage stamp,
  /// a missing entry means exactly that, and consumers must treat it as an
  /// error.
  static bool isModeledRegionOp(Operation *op) {
    return isa<ExprOp, IfOp, LoopOp>(op);
  }

  /// Run the phase analysis. Returns failure if phase errors are detected
  /// (diagnostics already emitted via `mlir::emitError`).
  LogicalResult run();

  /// Look up the resolved phase for an op. Asserts if not found.
  int16_t getPhase(Operation *op) const;

  /// Look up the resolved phase for a value. Asserts if not found.
  int16_t getValuePhase(Value value) const;

  /// Annotate every op in the function with its resolved phase (`pa.phase`)
  /// and the phases of its operands and results (`pa.operands`,
  /// `pa.results`), for lit testing. Sentinel phases print as "float" and
  /// "never"; values without a resolved phase print as "?".
  void annotate() const;

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
  /// the op phase when more than one terminator provides the result. The
  /// recorded phase uses demand-time value phases; the final recompute in
  /// `run()` refreshes it once all phases are settled.
  LogicalResult constrainRegionResult(Operation *regionOp, unsigned resultIdx,
                                      int16_t latest);

  /// Collect the (value, terminator) pairs providing a given result of a
  /// region-bearing op: yields for exprs/ifs, breaks for loops. Terminators
  /// that do not carry the result are omitted.
  SmallVector<std::pair<Value, Operation *>, 2>
  resultProviders(Operation *regionOp, unsigned resultIdx);

  /// Push blockPhase down to all ops in a block (including the terminator).
  LogicalResult processBlock(Block &block, int16_t blockPhase);

  /// Assign phase to a single op and push constraints to operands/regions.
  /// Skips floating exprs, pure ops, and constants (demand-driven only).
  /// Validates terminator phase constraints (return/break/continue/signature
  /// must be at the correct phase).
  LogicalResult processOp(Operation *op, int16_t blockPhase);

  /// Step-5 recompute pass over a block the DFS has processed: tighten pure
  /// ops to their earliest phase, derive undemanded floating exprs, and
  /// refresh or fill in region result phases. Recurses into modeled regions
  /// in post-order, so providers are final before their consumers.
  void recomputeBlock(Block &block);

  /// Compute a pure op's earliest phase = max over operand phases (type
  /// operands +1, `hir.type_of` -1; sentinel and unresolved operands
  /// contribute no bound) and record it. Visited entries tighten, and
  /// unvisited ops adopt the phase when it is real. In `overwrite` mode
  /// (inside floating-expr derivation) existing entries are replaced
  /// instead, since provisional phases may rise once loop iteration
  /// arguments resolve.
  void derivePure(Operation *op, bool overwrite);

  /// Refresh or fill in the result phases of a modeled region op from its
  /// providers: max over the provider value phases (unresolved providers
  /// count as float), floored at the op phase for multi-provider results,
  /// `never` when no path provides them. Outside `overwrite` mode existing
  /// entries may only tighten.
  void fillRegionResults(Operation *op, bool overwrite);

  /// First pass of floating-expr derivation: derive the pure content, scan
  /// nested regions, and accumulate into `bound` the lower bounds the
  /// anchored content imposes on the expr's phase. Bound sources are frame-
  /// consumed channels (CF conditions, loop inits, continue values, type
  /// operands, out-targeting exits) and internal crossings — an interior op
  /// consuming an operand at a phase below its own, which forces the frame
  /// to materialize at that op's phase. Nested floating exprs derive
  /// independently here and impose no bound. `shift` is the pin offset of
  /// the current frame relative to the expr.
  void scanFloatingContent(Block &block, int16_t shift, int16_t &bound);

  /// Second pass of floating-expr derivation: anchor CF ops, terminators,
  /// pinned exprs, and loop iteration arguments at the derived phase, and
  /// re-derive the pure content and region results with the frame resolved.
  void assignDerivedBlock(Block &block, int16_t blockPhase);

  /// Derive the phase of an undemanded floating expr exactly like a pure
  /// op: earliest = max over the bounds imposed by its anchored content and
  /// its result types. With no real bound the expr and its remaining
  /// contents stay unresolved and float in step 6.
  void deriveFloatingExpr(ExprOp exprOp);
};

} // namespace uir
} // namespace silicon
