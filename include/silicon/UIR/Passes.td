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
  let summary = "Lower structured UIR control flow to block-based CF";
  let description = [{
    Converts structured control flow ops from the UIR dialect (`uir.if`,
    `uir.loop`, `uir.yield`, `uir.break`, `uir.continue`, `uir.return`,
    `uir.unreachable`) into block-based control flow (`cf.br`,
    `cf.cond_br`, `hir.return`).

    This pass runs after SplitPhases (which has dissolved `uir.expr` and
    `uir.pin`) and before PhaseEvalLoop. After this pass, no UIR control
    flow ops remain in the IR.

    See `docs/design/control-flow.md` for the FlattenCF design.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "silicon::hir::HIRDialect",
  ];
}

#endif // SILICON_UIR_PASSES_TD
