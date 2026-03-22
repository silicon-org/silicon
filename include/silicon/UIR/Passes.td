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

#endif // SILICON_UIR_PASSES_TD
