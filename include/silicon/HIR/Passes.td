//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_PASSES_TD
#define SILICON_HIR_PASSES_TD

include "silicon/HIR/Dialect.td"
include "mlir/Pass/PassBase.td"

def InferTypesPass : Pass<"infer-types"> {
  let summary = "Resolve type inference constraints via unification";
  let description = [{
    Processes `hir.unify` ops to resolve type inference constraints.
    The pass implements a worklist-based unification algorithm that
    handles three cases:

    1. **Both operands inferrable:** keeps the earlier one in dominance
       order and replaces all uses of the later one.
    2. **One inferrable, one concrete:** replaces the inferrable with
       the concrete value, provided the concrete dominates. If it does
       not dominate but can be cloned (pure, single-result, no
       regions, all operands dominate), the pass clones the concrete
       op to the inferrable's location.
    3. **Both concrete:** if both operands are defined by structurally
       equivalent ops (same op kind, same attributes, same number of
       operands), the later op is replaced with the earlier one and
       new `hir.unify` ops are created for each pair of differing
       operands, recursively resolving them.

    Unify ops that cannot be resolved in any of these ways are left
    in the IR for later passes (e.g., CheckTypes) to diagnose.

    This pass uses dominance analysis to ensure replacements maintain
    valid SSA form. It should be run after CheckCalls has introduced
    unification constraints from function signatures and call sites.
  }];
}

def CheckCallsPass : Pass<"check-calls", "mlir::ModuleOp"> {
  let summary = "Move function signature constraints to call sites";
  let description = [{
    Inlines function signature type constraints into call sites and
    function bodies, connecting declared types with actual usage
    through `hir.unify` and `hir.coerce_type` ops.

    For each `hir.unified_call`, the pass clones the callee's
    signature region and inlines it ahead of the call. The
    signature's entry block arguments are replaced with the actual
    call arguments, and `hir.unify` ops are created to constrain
    declared argument and result types against the call's type
    operands. For single-block signatures, ops are spliced directly
    before the call. For multi-block signatures, the call's block is
    split and the signature blocks form a preamble, with type values
    threaded as block arguments to the continuation block.

    For function bodies, the signature region is cloned into the body
    region. `hir.coerce_type` ops are inserted to constrain block
    arguments with their declared types, and `hir.unify` ops link
    return type operands with the declared return types. Multi-block
    signatures are handled by creating a preamble that branches into
    the original body block with type values as block arguments.

    The pass processes functions in post-order over the call graph in
    function signatures, detecting and reporting recursive signature
    calls as errors. Function bodies are then checked in parallel.

    **Expects:** `hir.unified_func` ops with signature and body
    regions.

    **Produces:** `hir.unify` and `hir.coerce_type` ops linking
    declared types to actual usage, ready for the InferTypes pass.
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
  ];
}

def CheckTypesPass : Pass<"check-types"> {
  let summary = "Check for unresolved type mismatches after inference";
  let description = [{
    Runs after type inference and checks for remaining `hir.unify` ops
    whose operands are both concrete but different types. These indicate
    type mismatches in the source program (e.g., a function body
    returning `()` when the declared return type is `int`). Emits
    user-facing error diagnostics for each such mismatch.
  }];
}

def SplitPhasesPass : Pass<"split-phases", "mlir::ModuleOp"> {
  let summary = "Split functions into multiple phases based on constness";
  let description = [{
    Splits `hir.unified_func` ops into per-phase `hir.func` ops
    grouped under `hir.split_func` and `hir.multiphase_func`
    wrappers. This is the central pass in the phased execution
    pipeline, translating the unified representation into separate
    compile-time and run-time functions.

    The pass uses `PhaseAnalysis` to assign each op and value to a
    phase number. It then:

    1. **Determines the phase range** from op/value phases and
       declared argument/result phases, including callee-relative
       phases from `hir.unified_call` ops.

    2. **Groups consecutive phases** into phase groups. A group ends
       at each externally visible phase (one with caller-provided
       arguments or caller-visible results). Trailing internal phases
       form their own group.

    3. **Creates per-phase `hir.func` ops** named using a
       group-based convention (`@name.<group>` for single-phase
       groups, `@name.<group><letter>` for multi-phase groups).

    4. **Moves body block arguments** to their declared phase
       functions. Arguments at phase 0 stay in the phase-0 function;
       shifted arguments (`const` or `dyn`) are moved to their
       target phase.

    5. **Decomposes `hir.unified_call` ops** into per-phase
       `hir.call` ops. Each call targets the corresponding phase
       function from the callee's `hir.split_func`. Cross-phase
       values (arguments and results that span multiple phases) are
       threaded through opaque pack/unpack context arguments.

    6. **Distributes ops across phase functions.** Ops are moved from
       the phase-0 function to their assigned phase. Values used
       across phase boundaries are forwarded as return
       values/arguments of adjacent phase functions. Pure ops (type
       constructors, constants) are cloned into phases that need them
       rather than threaded.

    7. **Wraps multi-phase groups** in `hir.multiphase_func` ops
       that track the sub-functions and their shared opaque context
       for later evaluation by the PhaseEvalLoop.

    8. **Creates `hir.split_func`** ops that list all phase functions
       and their phase numbers, preserving the original function's
       argument/result names and phases.

    **Expects:** `hir.unified_func` ops with phase annotations from
    `PhaseAnalysis`, after CheckCalls and InferTypes.

    **Produces:** `hir.split_func`, `hir.func`, and
    `hir.multiphase_func` ops ready for the PhaseEvalLoop pipeline.
  }];
}

def SpecializeFuncsPass : Pass<"specialize-funcs", "mlir::ModuleOp"> {
  let summary = "Specialize functions with evaluated compile-time results";
  let description = [{
    Performs two kinds of specialization:

    1. **Multiphase chaining**: when the first sub-function of a
       `multiphase_func` has been evaluated (`mir.evaluated_func` exists),
       chain its results into the next sub-function by replacing the opaque
       context arg with `hir.mir_constant` ops. Dissolve the
       `multiphase_func` when only one sub-function remains.

    2. **Transitive call-site specialization**: after chaining, walk the
       specialized function's body for `hir.call` ops that pass
       `hir.mir_constant` opaque values as arguments. Clone the callee,
       inline the constant, expand the `opaque_unpack`, and update the call.
  }];
  let dependentDialects = [
    "silicon::mir::MIRDialect",
  ];
}

#endif // SILICON_HIR_PASSES_TD
