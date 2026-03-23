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

    See `docs/design/control-flow.md` for the FlattenCF design.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "silicon::hir::HIRDialect",
  ];
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

    See `docs/design/phase-splits.md` and `docs/design/phase-inference.md`.
  }];
  let dependentDialects = [
    "silicon::hir::HIRDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

#endif // SILICON_UIR_PASSES_TD
