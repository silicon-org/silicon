# TODO

- **Unify void and non-void specialization paths in SpecializeFuncs.**
  Root cause: SplitPhases skips `opaque_pack`/`opaque_unpack` insertion when there are no cross-phase context values (`if (contextReturns == 0) continue`).
  This causes sub-functions to sometimes have an opaque result and sometimes not, forcing SpecializeFuncs to maintain separate code paths for `resultAttrs.empty()` vs non-empty.
  Fix: SplitPhases should always emit the opaque pack/unpack pair, even when empty.
  Then `expandOpaqueContext` handles the empty case naturally (erases the trivial unpack and context arg), and the `resultAttrs.empty()` branch in SpecializeFuncs can be removed.
  The `phaseFuncs.size() < 2` dissolution case remains but doesn't need call erasure — it just redirects `split_func` symbol references from the `multiphase_func` to its remaining sub-function.
- **Remove `eraseVoidCalls` from SpecializeFuncs; rely on SymbolDCE.**
  `eraseVoidCalls` walks the entire module to erase zero-result calls before erasing the symbol.
  It's unclear whether such calls actually exist — sub-functions are dispatched via `split_func`, not direct `hir.call`/`mir.call`.
  `EvaluatedFuncOp` has the `Symbol` trait, and SymbolDCE already runs inside PhaseEvalLoop and after it.
  With the SplitPhases fix above, `evaluated_func` ops would always have one result (possibly an empty opaque), making the zero-result filter in `eraseVoidCalls` a no-op anyway.
  Remove `eraseVoidCalls`, the trailing cleanup loop, and the eager `symbolTable.erase(evalFunc)` calls; let SymbolDCE handle cleanup.
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
- **Implicit type widening (`uint<8>` to `uint<16>`) not supported.**
  Input: `pub fn widen(a: uint<8>) -> uint<16> { a }` — error: `hir.unify survived to HIR-to-MIR lowering`.
  There's no implicit widening conversion.
  Low priority; explicit cast syntax would be the right fix, but that syntax doesn't exist yet either.

## End-to-End Test Cleanup

Tests to convert to focused lit tests (then remove the e2e test):

- **`test/EndToEnd/duplicate-calls.si`**: regression for `isResolvableType`; already covered by `hir-to-mir.mlir::UnifyInReturnType`.
  Add a CheckCalls test for duplicate call-site unification, then remove.
- **`test/EndToEnd/unused-args.si`**: tests HIRToMIR deriving arg types from `typeOfArgs` when args are unused.
  Add a test case to `test/Conversion/hir-to-mir.mlir` with unused block args, then remove.
- **`test/EndToEnd/dyn-args.si`**: tests dyn-arg and mixed const+dyn arg splitting.
  Add dyn-arg and mixed const+dyn test cases to `test/HIR/split-phases.mlir`, then remove.

Tests to remove (redundant with existing coverage):

- **`test/EndToEnd/deduplication.si`**: covered by `specialize-funcs.mlir::TwoCalls`; CHECK lines are weak.
- **`test/EndToEnd/operations.si`**: same pattern as `add-const.si` for `-`, `*`, `/`, `%`; each op already tested in `Codegen/basic.si` and `MIR/interpret.mlir`.
- **`test/EndToEnd/double-const.si`**: `const const fn` already tested in `fn-phase-modifiers.si::const_const_fn` with richer assertions.
- **`test/EndToEnd/dyn-arg-return.si`**: CHECK lines map 1:1 to single-pass outputs; dyn return covered in `split-phases.mlir::DynReturn` and `Codegen/signature.si`.

## Dialect Review: Crash Risks

- **Interpret: division by zero crash.**
  `mir.div` and `mir.mod` with a zero RHS will crash.
  Add a zero check and emit a diagnostic.
  Also add shift-amount validation for `mir.shl`/`mir.shr` (negative or >=64 is UB).

- **Interpret: infinite loop on non-terminating programs.**
  The interpreter loop has no iteration limit; a back-edge that never returns will hang.
  Add a configurable `maxSteps` limit with a diagnostic on exhaustion.

- **Interpret: unbounded call stack.**
  Recursive functions grow the call stack until OOM.
  Add a `maxCallDepth` check.

- **CheckCalls: `assert` instead of diagnostic for missing SignatureOp.**
  Lines 221 and 390 of `CheckCalls.cpp` use `assert(terminatorOp)`.
  Replace with `emitBug` diagnostic.

- **SplitPhases: `resolveTypeIntoRegion` null deref on block args.**
  Line 39-40: `val.getDefiningOp()` can return null for block arguments.

- **SplitPhases: `clonePureOp` assumes op has results.**
  Line 63: `op->getResult(0)` called without checking `getNumResults() > 0`.

- **SpecializeFuncs: unbounded recursion in `transitiveSpecialize`.**
  A cycle in the call graph could recurse unboundedly.
  Add a visited set or depth limit.

- **HIRToMIR: `isResolvableType`/`resolveHIRType` can infinite-loop on cyclic IR.**
  Both are recursive without cycle detection.
  Add a `SmallPtrSet<Operation*>` visited set.

- **HIRToMIR: `FuncOpConversion` only examines entry block.**
  Multi-block functions (from if/else) have the return op in successor blocks.
  Walk all blocks to find `hir.return`.

- **PhaseEvalLoop: silent success when `maxIterations` exhausted.**
  If the loop runs all 100 iterations without converging, it silently succeeds.
  Add an error diagnostic.
  Make `maxIterations` a pass option so tests can exercise this path.

## Dialect Review: Missing Error Handling

- **Interpret: silent fallthrough on unknown condition type.**
  `cf.cond_br` and `arith.select` check `BoolAttr`, `IntAttr`, `IntegerAttr` but fall through silently for others.

- **SplitPhases: silent skip when `calleeSplitFunc` is null.**
  Line 383: if a `UnifiedCallOp` references a callee with no `SplitFuncOp`, it is silently skipped.

- **SpecializeFuncs: `nextFunc` null silently skipped.**
  Line 327-330: missing sub-function produces a debug message but should be `emitBug` + `signalPassFailure()`.

- **HIRToMIR: `CoerceTypeOp` silently passes non-constant type operands.**
  Falls through and drops the type annotation without diagnostic.

- **HIRToMIR: no conversion patterns for `RefTypeOp`, `ExprOp`, `YieldOp`, `LetOp`, `StoreOp`, `NextPhaseOp`.**
  Produce opaque "failed to legalize" errors if they survive to HIRToMIR.

- **PhaseEvalLoop: dangling symbol references silently ignored.**
  If a `multiphase_func`'s first phase symbol doesn't resolve, it's silently not counted.

- **CheckTypes: `OpaqueTypeOp` in `isConcreteTypeConstructor` but not in `describeTypeValue`.**
  Produces confusing `(unknown)` in error messages.

- **MIRToCIRCT: `isModule` behavior mismatch.**
  Docs say "only `isModule` funcs and transitive callees" but implementation lowers ALL convertible `mir.func` ops.

- **MIRToCIRCT: signed comparison predicates for unsigned types.**
  All comparison ops use signed predicates even for `!si.uint<N>`.

- **MIRToCIRCT: constant value truncation.**
  `static_cast<int64_t>` silently truncates values larger than 64 bits.

- **MIRToCIRCT: missing `mir.bool_to_i1` conversion pattern.**

## Dialect Review: Missing Documentation

- **InferTypes**: no `summary` or `description` in `Passes.td`
- **CheckCalls**: no `description` in `Passes.td`
- **SplitPhases**: no `description` in `Passes.td` (most complex pass, 1180 lines)
- **PhaseEvalLoop**: no `description` in `Passes.td`
- **HIRToMIR**: no `summary` or `description` in `Passes.td`
- **Base types**: no `description` on any type in `Types.td`
- **Base attrs**: no `summary`/`description` on `BaseTypeAttr`, `IntAttr`, `BaseUnitAttr`, `SiBoolAttr`
- **HIR ops**: ~20 ops missing `summary`, most ops missing `description`
- **MIR ops**: 19 ops missing `summary`/`description` (all binary/cmp ops, `ConstantOp`, `ReturnOp`, `CallOp`)

## Dialect Review: Missing Tests

### Roundtrip tests (`basic.mlir`)

- **HIR**: 15 of 16 binary/cmp ops have no roundtrip test (only `hir.add` tested); `FuncTypeOp` and `NextPhaseOp` untested; visibility variants for `UnifiedFuncOp`/`SplitFuncOp`/`MultiphaseFuncOp` untested
- **MIR**: 16 ops (all binary + all cmp) have zero roundtrip tests
- **Base**: limited `OpaqueAttr` roundtrip coverage (no heterogeneous or nested opaque tests)

### Verifier error tests (`errors.mlir`)

- **MIR `errors.mlir` is empty**: `FuncOp::verify()` has 4+ error paths, none tested
- **HIR**: missing tests for `SplitFuncOp` array-size mismatches, `MultiphaseFuncOp` `argIsFirst` size mismatch, `ReturnOp` `values`/`typeOfValues` size mismatch, `SignatureOp` outside valid parent, `NextPhaseOp` outside `hir.func`
- **HIR**: `ExprOp`/`YieldOp` lack a verifier for operand count/type matching
- **Base**: no `errors.mlir` file

### Pass error tests

- **HIRToMIR**: 13 error paths untested (non-constant/negative/excessive uint width, non-constant func_type args/results, coerce_type type mismatch, call non-constant result type, etc.)
- **MIRToCIRCT**: empty error file, 8 `emitBug` paths untested
- **PhaseEvalLoop**: no test for `maxIterations` exhaustion, "still pending" multiphase_func note, sub-pipeline failure propagation
- **CheckTypes**: only 1 type mismatch combination tested (unit vs int); add int vs uint, ref vs int, etc.
- **SpecializeFuncs**: no test for void-result evaluated func chaining
- **SplitPhases**: no test for "op uses value from later phase" error
- **InferTypes**: no error test file

### Pass positive tests

- **HIRToMIR**: ~7 conversion patterns untested (`hir.mir_constant`, `hir.coerce_to_i1`, `arith.select`, `hir.opaque_pack`, `hir.opaque_unpack`, `hir.inferrable`, `hir.type_of`)
- **Interpret**: missing tests for `arith.select`, `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip, zero-result function
- **MIRToCIRCT**: missing tests for `!si.bool` type conversion, multi-instance counter, source materialization (iN→i1), zero-result functions

### Other focused test gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir`
- Add unused block args test case to `test/Conversion/hir-to-mir.mlir`
- Add per-argument `dyn` and mixed const+dyn arg cases to `test/HIR/split-phases.mlir`
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

## Dialect Review: Missing Constraints and Traits

- **MIR binary/cmp ops use `AnyType`**: should constrain to numeric types; cmp results should be `BoolType`
- **MIR `BoolToI1Op`**: `$input` is `AnyType`, should be `BoolType`
- **MIR `OpaquePackOp`/`OpaqueUnpackOp`**: result/input unconstrained, should be `OpaqueType`
- **MIR `FuncOp`**: missing `RecursiveMemoryEffects`
- **MIR `ReturnOp`**: no verifier checking return types match enclosing `mir.func` results
- **HIR `ExprOp`/`YieldOp`**: no verifier for operand count/type matching
- **HIR constant ops**: missing `ConstantLike` trait
- **Base `UIntType`**: no width verifier (e.g. reject width=0)

## Dialect Review: Missing Canonicalizers/Folders

- **MIR binary ops**: no constant folding (`mir.add #si.int<1>, #si.int<2>` → `#si.int<3>`)
- **MIR `BoolToI1Op`**: no fold for constant bool input
- **MIR `OpaquePackOp`/`OpaqueUnpackOp`**: no round-trip canonicalization
- **HIR `CoerceTypeOp`**: no fold for identity coercions
- **HIR `OpaquePackOp`**: no fold for pack-of-unpack
- **HIR `TypeOfOp` canonicalization for `constant_bool`**: exists but untested

## Remove single-return/single-signature assumption

The codebase assumes each function body has exactly one `ReturnOp` (in the last block) and each signature region has exactly one `SignatureOp`.
This is baked into `getReturnOp()` and `getSignatureOp()` helper functions which grab `getBody().back().getTerminator()` / `getSignature().back().getTerminator()`.
We want to allow 0, 1, or many return/signature terminators across arbitrarily structured regions, and delete these two helpers.

Prerequisite changes needed:

- **HIR `FuncOp::verify`** (`Ops.cpp`): uses `getReturnOp()` to check `resultNames` count — needs to walk all returns or defer the check
- **HIR `UnifiedFuncOp::verifyRegions`** (`Ops.cpp`): asserts last body block has `ReturnOp` and last signature block has `SignatureOp` — relax to allow any structure
- **HIR `SplitFuncOp::verifyRegions`** (`Ops.cpp`): same pattern for signature
- **`UnifiedCallOp::verify`** (`Ops.cpp`): looks up callee's `getSignatureOp()` to validate arg/result counts — walk or collect from region instead
- **HIRToMIR `FuncOpConversion`** (`HIRToMIR.cpp`): derives MLIR function signature types (`typeOfArgs`, `typeOfValues`) from the single return op — this is the most structurally coupled site; needs a new strategy for collecting type info (perhaps from the signature region or a dedicated attribute)
- **SplitPhases** (`SplitPhases.cpp`): uses both `getReturnOp()` (for effective result phases, result names) and `getSignatureOp()` (for per-phase argument splitting) — needs to walk or iterate
- **MIR `FuncOp::getReturnOp`** (`MIR/Ops.cpp`): same pattern as HIR; delete alongside
- **CheckCalls** (`CheckCalls.cpp`): already uses `walk` for returns, so no change needed

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
- Consider implementing `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix)
