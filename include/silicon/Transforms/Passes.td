//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_TRANSFORMS_PASSES_TD
#define SILICON_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def PhaseEvalLoopPass : Pass<"phase-eval-loop", "mlir::ModuleOp"> {
  let summary = "Iteratively lower, interpret, and specialize until all "
    "compile-time phases are evaluated";
  let description = [{
    Drives the iterative phased evaluation pipeline that peels off
    one compile-time phase per iteration. Each iteration runs a
    sub-pipeline of:

    1. **HIRToMIR** — lowers ready `hir.func` ops to `mir.func`.
    2. **Canonicalize + CSE** — simplifies and deduplicates.
    3. **Interpret** — evaluates zero-argument `mir.func` ops,
       producing `mir.evaluated_func` ops with constant results.
    4. **SpecializeFuncs** — chains evaluated results into the next
       phase of `hir.multiphase_func` ops via `hir.mir_constant`,
       and performs transitive call-site specialization.
    5. **Canonicalize + CSE + SymbolDCE** — cleans up after
       specialization.

    The loop terminates when no `hir.multiphase_func` or lowerable
    `hir.func` ops remain. It tracks progress by comparing the set
    of actionable ops before and after each iteration, reporting a
    compiler bug if no progress is made.

    An `hir.func` is considered actionable (ready for lowering) when
    it contains no `hir.opaque_unpack` ops, no `hir.opaque_type` in
    return type-of-args, and no calls to `hir.multiphase_func`
    symbols. An `hir.multiphase_func` is actionable when its first
    sub-function has been evaluated or is a zero-argument function
    ready for evaluation. Template multiphase functions (whose
    sub-functions take arguments) are processed through transitive
    specialization rather than direct evaluation.

    **Expects:** `hir.func`, `hir.split_func`, and
    `hir.multiphase_func` ops as produced by SplitPhases.

    **Produces:** all HIR functions lowered to `mir.func`, with all
    compile-time phases fully evaluated.
  }];
  let dependentDialects = [
    "hir::HIRDialect",
    "mir::MIRDialect",
  ];
  let options = [
    Option<"maxIterations", "max-iterations", "unsigned", "100",
           "Maximum number of iterations before reporting failure">
  ];
}

#endif // SILICON_TRANSFORMS_PASSES_TD
