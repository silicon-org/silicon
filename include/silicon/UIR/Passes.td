//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_PASSES_TD
#define SILICON_UIR_PASSES_TD

include "silicon/UIR/Dialect.td"
include "mlir/Pass/PassBase.td"

def FlattenCFPass : Pass<"flatten-cf", "mlir::ModuleOp"> {
  let summary = "Lower structured UIR ops to block-based CF";
  let description = [{
    Converts structured UIR ops into flat block-based IR:

    - **Control flow:** `uir.if`, `uir.loop`, and their terminators
      (`uir.yield`, `uir.break`, `uir.continue`, `uir.return`,
      `uir.unreachable`) are lowered to `cf.br`, `cf.cond_br`, and
      `hir.return`.
    - **Phase grouping:** `uir.expr` is inlined into the parent block,
      and `uir.pin` is replaced by its inputs (identity removal).

    This pass runs after SplitPhases and before PhaseEvalLoop. After
    this pass, no UIR ops remain in the IR (except `uir.split_func`
    witnesses).
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "silicon::hir::HIRDialect",
  ];
}

def SortPhasesPass : Pass<"sort-phases", "mlir::ModuleOp"> {
  let summary = "Sort uir.func content into per-phase uir.phase ops";
  let description = [{
    Runs phase analysis on each `uir.func` and rewrites the internals of its
    body and signature regions into the sorted phase-op form, leaving the
    function op itself untouched: each region holds only `uir.phase` ops (at
    most one per phase, in increasing phase order), floating ops, and the
    region terminator. Ops anchored at a real phase move into that phase's
    `uir.phase` op in program order; floating ops stay at the region top
    level (or sink after operands that moved); function results move onto
    the `uir.phase_return` of their declared phase and the body's
    `uir.return` becomes `uir.unreachable`. Phase ops are only materialized
    for phases with content, where a phase return carrying function results
    counts as content. `uir.expr` and `uir.pin` phase grouping is dissolved
    first.

    Values computed inside a phase and consumed in a later one cross as
    phase-op results: the defining phase's `uir.phase_return` yields the
    value once, and consumers at any later phase — including function-result
    and type operands on later phase returns and the region terminator — use
    the `uir.phase` op's result directly, at any distance. Values already
    visible at the region top level (function arguments and floating ops)
    are consumed directly, without routing through a phase op.

    Each `uir.call` is decomposed into one `uir.phase_call` per externally
    visible callee phase (the callee's declared argument and result phases
    plus phase 0, in the callee's own frame). The call's own analysis phase
    is the single frame shift shared by all entries; consecutive entries
    chain in ascending phase order through an opaque context value that
    crosses phases like any other value. The entries derive from the callee
    declaration alone, so recursion decomposes like any other call.

    A `uir.if` at phase p whose arms hold later-phase content, or whose
    result selects between distinct later-phase values, is extracted into a
    choice-yielding branch plus a receiving `hir.pick`: the if is rebuilt at
    phase p with a per-arm `hir.choice` (tagging the taken arm and carrying
    the arm-local values the moved content samples) as an extra result, and
    the later-phase content and results move into a `hir.pick` at phase p+1
    that dispatches on the choice. The choice crosses phases as an ordinary
    crossing, and nested branches produce nested choices and picks. Only
    genuine choice-ification is lifted this way; a result diverging from its
    op's phase by identity forwarding or by never being produced is left for
    the divergence rejection.

    A `uir.loop` at phase p whose body holds later-phase content is extracted
    into a hit list of per-activation records plus a receiving `hir.replicate`
    at phase p+1: a `hir.opaque_list_create` is placed before the loop, each
    activation packs and pushes one record of the crossing values it samples,
    and the later-phase content moves into the replicate, which replays the
    records. Loop-phase early exits (`break`/`continue` in branch arms) make
    each record end in a nested exit-path choice that a nested `hir.pick` in
    the replicate dispatches on; nested loops produce nested hit lists. The
    list crosses phases as an ordinary crossing. Later-phase loop results
    (break-value choices) stay rejected.

    This is the only pass that runs or consumes PhaseAnalysis; the phase map
    never crosses the pass boundary.

    The pass currently covers straight-line content, call decomposition,
    single-boundary branch extraction, and single-boundary loop-record
    extraction, and rejects inputs that need the remaining sorting machinery
    with explicit diagnostics: early function exits (function-level exit
    chains), results delivered at a phase different from their op's own phase
    by identity-provided or never-produced control flow (which need result
    forwarding and unreachable-consumer handling), break-value choices,
    hoisting and content more than one phase away from its control flow, and
    backward uses from floating content nested under earlier-phase control
    flow. The pass's own output vocabulary
    (`uir.phase`, `uir.phase_return`, `uir.phase_call`) is likewise rejected
    in its input, as are ops inside regions the phase analysis does not
    model.
  }];
}

def SplitPhases2Pass : Pass<"split-phases2", "mlir::ModuleOp"> {
  let summary = "Split UIR unified functions into per-phase HIR functions";
  let description = [{
    Runs phase analysis on each `uir.func`, then splits it into per-phase
    `hir.func` ops using the phase map. Produces `uir.split_func` witnesses
    recording how each unified function was split, and `hir.multiphase_func`
    ops for groups of consecutive internal phases.

    The pass processes functions in topological order (callees before callers)
    so that `uir.split_func` witnesses are available when decomposing
    `uir.call` ops in callers.

    After this pass, `uir.func` and `uir.call` ops are gone. Structured CF
    ops (`uir.if`, `uir.loop`) survive in the per-phase `hir.func` bodies
    and are lowered by FlattenCF.

    With the `sort-only` option, the pass stops after the sorting stage:
    each `uir.func` is rewritten in place such that no control flow region
    contains content of a phase different from the region's own phase
    (phase grouping dissolved, calls decomposed, earlier-phase content
    hoisted, later-phase content extracted into replicate/pick receiving
    constructs), without creating any per-phase functions. The
    `annotate-phases` option additionally annotates all ops with the
    maintained phase map, for testing.
  }];
  let options = [
    Option<"sortOnly", "sort-only", "bool", "false",
           "Stop after the sorting stage, leaving rewritten uir.func ops">,
    Option<"annotatePhases", "annotate-phases", "bool", "false",
           "Annotate ops with the maintained phase map (testing)">,
  ];
  let dependentDialects = [
    "silicon::hir::HIRDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

#endif // SILICON_UIR_PASSES_TD
