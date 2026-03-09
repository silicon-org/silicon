# TODO

- Add a test to phase splitting to verify that the pass works if a `hir.unified_call` targets a `hir.split_func`; this is relevant for the library compilation model, where a library already contains split funcs and user code must be able to properly phase-split unified calls to a func that is already split in the IR.

## End-to-End Test Cleanup

Tests to convert to focused lit tests (then remove the e2e test):

- **`test/EndToEnd/duplicate-calls.si`**: regression for `isResolvableType`; already covered by `hir-to-mir.mlir::UnifyInReturnType`.
  Add a CheckCalls test for duplicate call-site unification, then remove.
- **`test/EndToEnd/unused-args.si`**: tests HIRToMIR deriving arg types from `typeOfArgs` when args are unused.
  Add a test case to `test/Conversion/hir-to-mir.mlir` with unused block args, then remove.
- **`test/EndToEnd/dyn-args.si`**: tests dyn-arg and mixed const+dyn arg splitting.
  Add dyn-arg and mixed const+dyn test cases to `test/HIR/split-phases.mlir`, then remove.

## Dialect Review: Missing Error Handling

- **SplitPhases: silent skip when `calleeSplitFunc` is null.**
  Line 383: if a `UnifiedCallOp` references a callee with no `SplitFuncOp`, it is silently skipped.

- **MIRToCIRCT: `isModule` behavior mismatch.**
  Docs say "only `isModule` funcs and transitive callees" but implementation lowers ALL convertible `mir.func` ops.

- **MIRToCIRCT: signed comparison predicates for unsigned types.**
  All comparison ops use signed predicates even for `!si.uint<N>`.

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

- **HIR**: `FuncTypeOp` and `NextPhaseOp` untested; visibility variants for `UnifiedFuncOp`/`SplitFuncOp`/`MultiphaseFuncOp` untested
- **Base**: limited `OpaqueAttr` roundtrip coverage (no heterogeneous or nested opaque tests)

### Verifier error tests (`errors.mlir`)

- **MIR `errors.mlir` is empty**: `FuncOp::verify()` has 4+ error paths, none tested
- **HIR**: missing tests for `SplitFuncOp` array-size mismatches, `MultiphaseFuncOp` `argIsFirst` size mismatch, `ReturnOp` `values`/`typeOfValues` size mismatch, `SignatureOp` outside valid parent, `NextPhaseOp` outside `hir.func`
### Pass error tests

- **HIRToMIR**: 13 error paths untested (non-constant/negative/excessive uint width, non-constant func_type args/results, coerce_type type mismatch, call non-constant result type, etc.)
- **MIRToCIRCT**: empty error file, 8 `emitBug` paths untested
- **PhaseEvalLoop**: no test for "still pending" multiphase_func note, sub-pipeline failure propagation
- **CheckTypes**: only 1 type mismatch combination tested (unit vs int); add int vs uint, ref vs int, etc.
- **SpecializeFuncs**: no test for void-result evaluated func chaining
- **SplitPhases**: no test for "op uses value from later phase" error
- **InferTypes**: no error test file

### Pass positive tests

- **HIRToMIR**: ~7 conversion patterns untested (`hir.mir_constant`, `hir.coerce_to_i1`, `arith.select`, `hir.opaque_pack`, `hir.opaque_unpack`, `hir.inferrable`, `hir.type_of`)
- **Interpret**: missing tests for `arith.select`, `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip, zero-result function
- **MIRToCIRCT**: missing tests for `!si.bool` type conversion, multi-instance counter, source materialization (iN→i1), zero-result functions

### Other focused test gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir` -- unless we've already removed the op
- Add unused block args test case to `test/Conversion/hir-to-mir.mlir`
- Add per-argument `dyn` and mixed const+dyn arg cases to `test/HIR/split-phases.mlir`
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

## Dialect Review: Missing Constraints and Traits

- **MIR binary/cmp ops use `AnyType`**: should constrain to numeric types; cmp results should be `BoolType`

## Dialect Review: Missing Canonicalizers/Folders

- **MIR binary ops**: no constant folding (`mir.add #si.int<1>, #si.int<2>` → `#si.int<3>`)
- **MIR `OpaquePackOp`/`OpaqueUnpackOp`**: no round-trip canonicalization
- **HIR `OpaquePackOp`**: no fold for pack-of-unpack

## Remove single-return/single-signature assumption

The codebase assumes each function body has exactly one `ReturnOp` (in the last block) and each signature region has exactly one `SignatureOp`.
This is baked into `getReturnOp()` and `getSignatureOp()` helper functions which grab `getBody().back().getTerminator()` / `getSignature().back().getTerminator()`.
We want to allow 0, 1, or many return/signature terminators across arbitrarily structured regions, and delete these two helpers.

Prerequisite changes needed:

- **HIR `FuncOp::verify`** (`Ops.cpp`): uses `getReturnOp()` to check `resultNames` count — needs to walk all returns or defer the check
- **HIR `UnifiedFuncOp::verifyRegions`** (`Ops.cpp`): asserts last body block has `ReturnOp` and last signature block has `SignatureOp` — relax to allow any structure
- **HIR `SplitFuncOp::verifyRegions`** (`Ops.cpp`): same pattern for signature
- **`UnifiedCallOp::verify`** (`Ops.cpp`): looks up callee's `getSignatureOp()` to validate arg/result counts — walk or collect from region instead
- **HIRToMIR `FuncOpConversion`** (`HIRToMIR.cpp`): now walks all return ops and asserts they use identical type operands; consider adding a dedicated signature op in the function body to annotate arg/result types in one place, instead of deriving them from return ops
- **SplitPhases** (`SplitPhases.cpp`): uses both `getReturnOp()` (for effective result phases, result names) and `getSignatureOp()` (for per-phase argument splitting) — needs to walk or iterate
- **MIR `FuncOp::getReturnOp`** (`MIR/Ops.cpp`): same pattern as HIR; delete alongside
- **CheckCalls** (`CheckCalls.cpp`): already uses `walk` for returns, so no change needed

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
- Consider implementing `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix)
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
