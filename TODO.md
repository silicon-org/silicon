# TODO

- cross-ex4.si, cross-ex8.si: `mir.return` type mismatch (`!si.uint<N>` vs `!si.opaque`) when a function calls another dependent-type function.
  The return type in the MIR func doesn't match because the opaque signature types aren't properly resolved after specialization.
- cross-ex5.si: `hir.unify` survives to HIRToMIR with different operands.
  The `const { 4 + 4 }` expression creates a unify between the computed type and the expected type that InferTypes can't resolve.
  We can't symbolically prove equivalence, but we want the canonicalizer to be able to at least resolve constant expressions down to simple constants that will then unify.
- Type unification can't prove structural equivalence of `uint` types with different width operands.
  Need to teach InferTypes/CheckTypes to structurally compare `uint_type` ops by unifying their widths.
- The `in_range` example in `docs/examples/basics/operators.md` fails with `mir.return` type mismatch (`!hir.any` vs `!si.bool`).
  Tagged `silicon-todo` to skip doc testing.
  Root cause: the `&&` short-circuit produces a value typed `!hir.any` that doesn't get coerced to `!si.bool` before the MIR return.
  Unrelated to multi-block signatures (which are now handled by phi tracing in `resolveTypeValue`).
- Document how integer literal type inference works in docs/examples/.
- HIR needs a `hir.constant` op that can materialize any of the `#si.*` constant attributes.
  Use that op to materialize most constants that have a fully-known type and/or value; this would replace constant-like ops such as `hir.int_type`, `hir.unit_type`, etc.
  We'd still need `hir.constant_int` to construct an integer literal with an inferrable type.
- Change syntax from `hir.return %a, %b -> (%a.ty, %b.ty)` to `hir.return %a, %b : %a.ty, %b.ty`
- Rewrite FlattenCF to process regions from outer to inner, iterating over the ops once and inlining structured CF ops as it goes along.
  Once a structured CF op is inlined, its blocks and regions are inserted ahead of that op, such that the linear region/block scan will naturally visit the newly-inlined region.
  For inspiration, look at how the SCF-to-CF lowering in MLIR does things.
- Improve UIR op parsing/printing (e.g. uir.call, maybe others) to not print the implied `!hir.any` type
- Add an HIR identifier op that just pass through an operand to its result.
  We'll use this op to have a mechanism to capture the location of a named expression like `x`.
  When reporting phase errors, we can stop at `x` and indicate an error right there, instead of pointing the user at a definition like the `x: const int` arg of a function, which is far more confusing.

## Phase Inference Redesign

See `docs/design/phase-inference.md`, `docs/design/unified-dialect.md`, and `docs/design/control-flow.md` (FlattenCF section).

**Migration strategy:** Build new passes additively alongside the old ones.
Old passes continue working on `hir.unified_func` and flat CF; new passes consume `uir.*` ops.
Once all pieces work, switch codegen to UIR, swap passes, then remove old code.

- SplitPhases2 replicate model (`extractReplicates`):
  - Audit for overfitting; likely needs a rewrite.
    The special-casing of void vs top-level dyn exprs and the juggling of lists of dyn expr ops smells like overfitting to a few examples rather than the real design.
    The intended model is simpler: pull each dyn expr out one layer of control flow, generating that layer's hit lists, then iterate so the next layer's dyn exprs get pulled out one more layer, aggregating into further hit lists.
    Check whether the current code actually implements this or just passes a handful of cases.
  - Captured values: pack producing-phase values used in the dyn body into hits entries (currently packs nothing); a captured value defined inside the producing CF region currently makes `fixupCrossPhaseRefs` emit a dominance-violating opaque_pack instead of erroring -- detect and reject loudly until supported
  - Loop replicate threading: `extractLoopReplicates` uses dummy empty-pack threaded inits; no real accumulation (`x = x + 1`) or zero-iteration pass-through. Needs the unified `uir.loop` to carry the threaded value at the dyn phase (PhaseAnalysis currently pins carried inits to the loop phase)
- Phase 2: extend existing passes for region support (additive, old paths stay intact)
  - InferTypes: walk into `uir.if`/`uir.expr`/`uir.loop` regions in addition to flat blocks
  - InferTypes: optimistic hoisting of type op trees out of regions for cross-boundary RAUW
  - CheckCalls: walk into regions to find calls inside structured CF
  - CheckTypes: walk into regions, check `uir.yield`/`uir.break` type consistency
- Phase 3: switch codegen and pipeline (one atomic change, mark failing tests `XFAIL: *`)
  - Codegen: emit `uir.if` instead of `cf.cond_br` + merge blocks for if/else
  - Codegen: emit `uir.loop` instead of `cf.br` back-edges for while/loop
  - Codegen: emit `uir.return` for early returns inside structured CF (+ `uir.unreachable` after)
  - Codegen: emit `uir.expr`/`uir.pin` for `const { ... }` / `dyn { ... }` / `let` bindings
  - Codegen: create `hir.inferrable` for region op result types, `hir.unify` inside regions at yield
  - Codegen: inlinability check after emitting `uir.expr` (inline trivial, replace with `uir.pin`)
  - Pipeline: swap old `SplitPhases` → new `SplitPhases`, verify all tests pass
- Phase 4: cleanup (remove old code)
  - Remove old `SplitPhases` pass
  - Remove old `TestPhaseAnalysis` pass and `PhaseAnalysis`
  - Delete `hir.unified_func`, `hir.unified_call`, `hir.split_func`, `hir.expr`, `hir.yield`
  - Fix remaining XFAILed tests

## Dialect Review: Missing Error Handling

- **SplitPhases: silent skip when `calleeSplitFunc` is null.**
  Line 383: if a `UnifiedCallOp` references a callee with no `SplitFuncOp`, it is silently skipped.

## Dialect Review: Missing Tests

### Roundtrip tests (`basic.mlir`)

- **HIR**: `FuncTypeOp` and `NextPhaseOp` untested; visibility variants for `UnifiedFuncOp`/`SplitFuncOp`/`MultiphaseFuncOp` untested
- **Base**: limited `OpaqueAttr` roundtrip coverage (no heterogeneous or nested opaque tests)

### Verifier error tests (`errors.mlir`)

- **HIR**: missing tests for `SplitFuncOp` array-size mismatches, `MultiphaseFuncOp` `argIsFirst` size mismatch, `ReturnOp` `values`/`typeOfValues` size mismatch, `SignatureOp` outside valid parent, `NextPhaseOp` outside `hir.func`

### Pass error tests

- **HIRToMIR**: remaining untested `emitBug` paths (negative uint width, excessive uint width, non-constant func_type args/results, non-constant call result type, return op typeOfArgs/typeOfValues mismatch between multiple returns); most are guarded by `shouldLower` and hard to trigger from IR
- **MIRToCIRCT**: empty error file, 8 `emitBug` paths untested
- **PhaseEvalLoop**: no test for "still pending" multiphase_func note, sub-pipeline failure propagation
- **SpecializeFuncs**: no test for void-result evaluated func chaining
- **SplitPhases**: no test for "op uses value from later phase" `emitBug` (defensive check, hard to trigger from well-formed IR)
- **InferTypes**: no error test file

### Pass positive tests

- **HIRToMIR**: remaining untested patterns (`arith.select` requires arith dialect registration in silicon-opt, `hir.opaque_unpack` blocked by shouldLower, `hir.inferrable` requires unresolved type setup)
- **Interpret**: missing tests for `arith.select` (arith dialect not registered in silicon-opt), `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip
- **MIRToCIRCT**: missing tests for multi-instance counter, source materialization (iN→i1)

### Other focused test gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir` -- unless we've already removed the op
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

## Phase Inference Design Review

- Multiple `uir.continue` ops in a loop face the same challenge as multiple `uir.return` ops in a function.
  If they carry different SSA values at a phase earlier than the continue op itself, that is an error.
  Work out how this is solved for `uir.return`, draw conclusions, and tighten the design.
  Related to the if-yield result-phase requirement below.

- CF ops may need to lower-bound their result phases to their op phase.
  Consider: `uir.if` at phase 0 with a yield carrying `%a` at phase -1.
  Currently this produces an if result at -1 (transparent yield).
  But the if decides at phase 0 — it can't produce a value for phase -1,
  because the decision of which branch to take isn't available until phase 0.
  The earliest phase a CF result can be at is `p(CF)` (when the decision is made).
  A phase +1 result is fine (future). A phase -1 result is problematic.
  Think about whether CF ops should clamp `actualPhase[result] = max(p(CF), yieldedPhase)`.
  Example:

  ```mlir
  uir.func @test(%cond: 0, %a: -1) -> (result: -1) {
    ...
    %0 = uir.if %cond : %t {
      uir.yield %a : %t   // %a at -1, but if decides at 0
    } else {
      uir.yield %a : %t
    }
    uir.return %0 : %t  // result at -1, but if can't decide until 0
  }
  ```

- Review how `const { dyn { expr } }` and `dyn { const { expr } }` should behave for value results.
  Currently by the spec, `const { dyn { 42 } }` is an error: the dyn block pins its result at `p(const)+1 = 0`, but the const block demands `p(const) = -1`.
  This means `const { dyn { ... } }` is ALWAYS an error as a value expression (the dyn block is one phase too late), while `dyn { const { ... } }` can work (earlier values carry to later phases).
  For control flow (return/break/continue), balanced nesting DOES cancel because only `p(enclosing_block) = p(target)` is checked.
  Open questions:
  - Should `dyn { ... }` inside a const block be treated as "transparent" when the inner result is trivially available earlier (e.g., a literal)?
  - Should the result phase of a block expression be the block's phase, or the phase of the actual result expression inside?
  - Is the asymmetry between `const { dyn { ... } }` (always error) and `dyn { const { ... } }` (can succeed) intentional and desirable?
  - What are the implications for returning dyn values from const contexts?
    See `tmp/phase-inference/42-balanced-nesting-values.si` for test cases exploring this.

## Language Cleanup

- Remove `const fn` / `dyn fn` support from parser and codegen.
  These don't bring distinct semantics beyond arg/result modifiers and `const { ... }` / `dyn { ... }` blocks.
  See docs/design/phase-inference.md for the rationale.

## HIR Ops

- Comparison ops (`hir.gt`, `hir.lt`, `hir.geq`, `hir.leq`, `hir.eq`, `hir.neq`) should take a second type operand for the type of the values being compared, e.g. `hir.gt %a, %b : %int_ty -> %bool_ty` instead of just `hir.gt %a, %b : %bool_ty`.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
- Consider implementing `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix)
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
