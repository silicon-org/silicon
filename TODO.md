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

## Phase Inference Redesign

See `docs/design/phase-inference.md`, `docs/design/unified-dialect.md`, and `docs/design/control-flow.md` (FlattenCF section).

**Migration strategy:** Build new passes additively alongside the old ones.
Old passes continue working on `hir.unified_func` and flat CF; new passes consume `uir.*` ops.
Once all pieces work, switch codegen to UIR, swap passes, then remove old code.

- Phase 1: new passes (additive, no churn, testable via `silicon-opt`)
  - Implement `PhaseAnalysis2` DFS (top-down latest push, bottom-up earliest for pure ops)
  - Implement `TestPhaseAnalysis2` test pass (writes phase map as attributes for lit tests)
  - Implement new `SplitPhases2` pass consuming `uir.func` → `hir.func` + `uir.split_func`
  - User-facing phase error diagnostics using const/dyn vocabulary (no numeric phases)
  - Error DFS: propagate required phase downward, report at leaves, annotate calls on the way back
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

## Language Cleanup

- Remove `const fn` / `dyn fn` support from parser and codegen.
  These don't bring distinct semantics beyond arg/result modifiers and `const { ... }` / `dyn { ... }` blocks.
  See docs/design/phase-inference.md for the rationale.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
- Consider implementing `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix)
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
