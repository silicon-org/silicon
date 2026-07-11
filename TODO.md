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
- Consider migrating `hir::getTypeOf` (Ops.cpp) to op-declared information, e.g. a `getTypeOfResult` method on `TypeOperandsOpInterface`; it currently hardcodes the op-to-type-operand mapping in a TypeSwitch.
- Add an HIR identifier op that just pass through an operand to its result.
  We'll use this op to have a mechanism to capture the location of a named expression like `x`.
  When reporting phase errors, we can stop at `x` and indicate an error right there, instead of pointing the user at a definition like the `x: const int` arg of a function, which is far more confusing.

## Phase Inference Redesign (sampling model)

The semantics were settled in a design interview on 2026-07-08.
**Read `docs/design/sampling-model.md` (the model), `docs/design/function-exit-choices.md` (early exits), `docs/design/implementation-plan.md` (ops, milestones M0-M6), and `docs/design/phase-ops.md` (pass architecture: `uir.phase` contract between sorting and splitting; supersedes the merged-pass plan) before touching anything here.**
The old replicate design in `docs/design/control-flow.md` and the transparent-yield/futures model in `docs/design/phase-inference.md` are superseded where they conflict; docs get rewritten at M6.

- Milestones: M0 ops done → M1 PhaseAnalysis updates done (2026-07-10, incl. design-review fixes) → M2 sorting stage (mostly done 2026-07-10; see below) → M3 phase-op rebuild (immediate next steps below; docs/design/phase-ops.md) → M4 evaluation (interpreter lists/choices/vars, SpecializeFuncs expansion) → M5 language surface (assignment, loops, var, definite-return check) → M6 docs rewrite + old-stack removal.

### Immediate next steps: phase-op rebuild (docs/design/phase-ops.md; sorting first, splitter last)

1. Define `uir.phase`, `uir.phase_return`, `uir.phase_call` in tablegen with parsing tests in basic.mlir and verifier tests in errors.mlir; nail the `uir.phase_return` two-output shape (func results + phase yields) first.
2. Hand-write the target phase-op IR for the canonical examples (straight-line multi-phase, call, if choice, loop records) before any pass code; these are the sort/split contract and become lit tests.
3. Rebuild sorting case-by-case, one PR per case, porting machinery from lib/UIR/Transforms/SortPhases.cpp: straight-line single-phase wrap → multi-phase with crossings as phase results → call decomposition into `uir.phase_call` (from callee declaration, no witness) → signature-region phases → branch extraction (if/pick) → loop records/replicate → break-value choices → partial activations.
4. Then the new territory on the new representation: function-level exit chains on `uir.phase_return` (docs/design/function-exit-choices.md) and lifting the M2 v1 hard errors listed below.
5. Last: minimal phase-blind func splitting — per-phase funcs, opaque context from phase results, sig-crossing relocation into the previous phase's body, split_func/multiphase_func emission; diagnostic (not verifier) on non-phase-op content.
6. Old SortPhases/SplitPhases2 stay intact and green throughout as porting source and reference; switch silc at parity and delete them plus the sort-only flag in one removal PR (folds into M6).
- Early-exit design settled 2026-07-10 (docs/design/function-exit-choices.md): exit chains built at return sites, receiving phases keep multiple returns, picks born at split time, forward-only.
- M2 landed in lib/UIR/Transforms/SortPhases.cpp (sorting stage of split-phases2, sort-only/annotate-phases options): dissolve, call decomposition, hoisting, branch/loop extraction, nested lists, distance-2, break-value choices, return carriers, partial-activation records (per-exit record pushes with nested exit-path choices per sampling-model §6); tests in test/UIR/sort-phases{,-errors}.mlir.
- M2 remaining: all-empty-deletion negative test, and lifting the v1 hard errors on shared break-value slices, break/continue-terminated expr carriers (§4 D-rule break), rejoining-arm content in exit-carrying branches (needs the arm-local sub-choice from function-exit-choices.md §3), nested divergence (exits inside exit-carrying or rejoining arms), and exit-carrying branches with demanded later-phase results.
- Sorting invariant gaps to revisit in M3: uir.expr with unreachable terminator errors in dissolution; return-terminated exprs at the function body's top level stay undissolved until distribution.
- M3 pass structure superseded (2026-07-11, docs/design/phase-ops.md): sorting emits `uir.phase`/`uir.phase_return`/`uir.phase_call` as an explicit IR contract and is the only consumer of PhaseAnalysis; func splitting is a separate phase-blind pass; `uir.phase_call` is built from callee declarations, so no witnesses at sort time, no topological order, no cycle rejection, and the sort-only flag goes away.
- Floating-expr fallback settled 2026-07-10 (see implementation-plan.md): undemanded effectful floating exprs default to their block phase; a cap-at-block-phase alternative was rejected (breaks dyn-direction sliding of argument sub-exprs). Watch item: removing an expr's only consumer shifts its effects to the block phase — revisit if this bites.
- Never-produced CF results get the bottom phase `never` in PhaseAnalysis (settled 2026-07-10); a language-level never *type* (Rust `!`, `TyKind::Never`-style, coerces to any type) is a separate type-system feature for M5+; The sorting stage (M2) must treat `never` results as unreachable, not as choice-ification obligations.
- SplitPhases2.cpp determinePhaseRange/distribution test `phase == INT16_MIN` instead of `!isRealPhase`, so a `kNeverPhase` op entry would poison the range; the sorter's computePhaseRange does it correctly.
- PhaseAnalysis loose ends: only entry blocks of body/signature are processed in run() while collectTerminators walks all blocks — decide whether pre-FlattenCF multi-block regions are legal and verify or handle; step-3b return-value collection guards `i < values.size()` instead of asserting the verifier invariant.
- PhaseAnalysis resultProviders misattributes break values (consumed by constrainRegionResult, step-5 recompute, and the step-6 stamp): for an if result, an arm terminated by `uir.break %v` reads %v (a loop result value) as providing the if's result; if results on break-terminated paths never materialize and should not be providers.
- InferTypes/CheckCalls/CheckTypes region support (walk into `uir.if`/`uir.expr`/`uir.loop`, optimistic type-tree hoisting) still needed; slot in before/alongside M2.
- Old-stack removal (old `SplitPhases`, `TestPhaseAnalysis`, `hir.unified_func`/`unified_call`/`split_func`/`expr`/`yield`) folds into M6.

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

Settled by the sampling model (`docs/design/sampling-model.md`); kept here until the M6 docs rewrite lands them:

- Multiple `uir.continue`/`uir.return` with earlier-phase values: floor rule with SSA-identity-only same-value exemption (§7); later-phase values choice-ify.
- CF result phase lower bound: yes — a CF op at phase p selecting distinct values cannot deliver at a phase < p (causality); the transparent-yield "future" for later phases is replaced by choice-ification (§7).
- `const { dyn { expr } }` value results: governed by the causality rule — a phase p op can never consume a phase > p value; block result phase is the yielded value's phase, and the asymmetry vs `dyn { const { ... } }` is intentional.

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
