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

def CheckCallsPass : Pass<"check-calls", "mlir::ModuleOp"> {
  let summary = "Move function signature constraints to call sites and bodies";
  let description = [{
    Connects the types a function declares with the types its callers and
    its own body use, by copying the declaration's *computation* into
    every place that consumes it.

    A `uir.func` does not declare its types as attributes: its signature
    region computes one type value per argument and result, possibly
    depending on the function's own arguments. This pass copies that
    computation to each consumer and ties the copy to the local types:

    - For every `uir.call` in a `uir.func`, the callee's signature
      computation is cloned in front of the call with the signature's
      block arguments substituted by the call's arguments, and each
      declared type is unified with the call's corresponding type
      operand through `hir.unify`. The unify results replace the call's
      type operands.
      The callee symbol may resolve to a `uir.func` or to a
      `uir.split_func` -- the late-binding interface of a function that
      has already been sorted or split, including a declaration-only one
      from a precompiled library. Both declare their types the same way
      and are handled identically; nothing but the signature region of a
      callee is ever read.
    - For every `uir.func`, its own signature computation is cloned to
      the top of its body, applied to the body's arguments. Each argument
      is then constrained by its declared type through a
      `hir.coerce_type` that takes over the argument's uses, and every
      `uir.return`'s type operands are unified with the declared result
      types.

    Copying rather than referring is what carries dependent types across
    the later phase split: a caller recomputes a dependent callee type
    from arguments it already holds and threads it through its own phase
    boundaries, instead of reading the callee's private context.
    Agreement between the two computations is structural, since both are
    copies of one region.

    Only `uir.func` ops are checked; a `uir.split_func` is read, never
    rewritten. A split function is born from a checked declaration --
    phase sorting snapshots one downstream of this pass -- and a
    precompiled library ships one checked when the library was compiled,
    so its signature already holds checked calls, and a call copied out
    of it into a caller carries that check along. Bodies belong to
    `uir.func` alone; a `uir.split_func` is the declaration by itself.

    Checking is therefore not idempotent and must run exactly once, on
    its own module, before sorting. `hir.unify` folds only when both its
    operands are the same value, so a second run stacks another unify on
    every checked call and copies the signatures it calls afresh, which
    compounds along a chain of declarations. Leaving the persisted
    interface alone is what keeps that from happening across compiles;
    nothing else records that a signature has been checked.

    Signatures are checked before bodies, in post-order over the call
    graph restricted to signature calls, so a callee's signature is fully
    resolved before it is copied into a caller. A `uir.func` signature
    that transitively calls itself has no such order and is reported as a
    user error. A `uir.split_func` is a leaf of that traversal: its types
    are resolved before this run begins, so it constrains no order, and a
    cycle running through one is not an ordering problem and is not
    reported. Bodies are then checked in parallel.

    **Expects:** structured, exit-normalized input -- a signature region
    that is a single block exiting through one `uir.signature`. Neither
    flat CFG (post FlattenCF) nor unnormalized multi-exit signatures can
    be inlined at a use, and both are rejected. The shape belongs to the
    declaration rather than to any one use of it, so it is checked once
    per `uir.func`, whether that function is called or not, before
    anything is rewritten; for a `uir.split_func` its own verifier
    enforces the same shape.

    Every `uir.call` must sit inside the signature or body region of a
    declared function, since those are the regions the pass walks or has
    walked before the declaration came to be, and the module must still
    be unsorted -- a `uir.phase` op means sorting
    has already consumed the types this pass produces. Both are the
    compiler running its passes out of order and are reported as
    compiler bugs.

    **Produces:** `hir.unify` and `hir.coerce_type` ops linking declared
    types to actual usage, and calls whose type operands name both sides.

    **Pipeline position:** after exit normalization, and before phase
    sorting with a canonicalization stage in between: `check-calls`,
    `canonicalize`, `cse`, `canonicalize`, `sort-phases`,
    `thread-phases`, `split-funcs`. The second `canonicalize` is not
    redundant: where a body computes its declared type itself, the two
    computations are distinct values until CSE merges them, and only
    then does the `hir.unify` between them become an identity the
    canonicalizer can fold.

    That stage is load-bearing. Coercing a body argument to its declared
    type turns an `hir.type_of` of the argument into an `hir.type_of` of
    the coercion's result, whose type sorting would have to place one
    phase before a value of the same phase -- a backward crossing it
    rejects. Folding `type_of(coerce_type(_, %ty))` to `%ty` answers the
    query with the declared type the signature copy computes, which is
    one phase earlier. The fold reaches only queries whose answer is
    recoverable in the same region, so an `hir.type_of` inside a branch
    arm or of a call result still reaches sorting, where the type of a
    phase-P value demanded at phase P-1 remains unresolved.
  }];
  let dependentDialects = [
    "hir::HIRDialect",
    "uir::UIRDialect",
  ];
}

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
