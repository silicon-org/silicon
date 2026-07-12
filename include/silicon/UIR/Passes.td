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

    This is the only pass that runs or consumes PhaseAnalysis; the phase map
    never crosses the pass boundary.

    The pass currently covers straight-line content whose values stay within
    their own phase, and rejects inputs that need the remaining sorting
    machinery with explicit diagnostics: cross-phase value crossings (which
    will become phase-op results), `uir.call` decomposition into
    `uir.phase_call`, early function exits (function-level exit chains), and
    hoisting/extraction of content nested under differently-phased control
    flow. The pass's own output vocabulary (`uir.phase`, `uir.phase_return`,
    `uir.phase_call`) is likewise rejected in its input, as are ops inside
    regions the phase analysis does not model.
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
