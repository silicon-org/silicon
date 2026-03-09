# TODO

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

- **HIRToMIR**: remaining untested `emitBug` paths (non-constant uint width, excessive uint width, non-constant func_type args/results, non-constant call result type, return op typeOfArgs/typeOfValues mismatch between multiple returns); most are guarded by `shouldLower` and hard to trigger from IR
- **MIRToCIRCT**: empty error file, 8 `emitBug` paths untested
- **PhaseEvalLoop**: no test for "still pending" multiphase_func note, sub-pipeline failure propagation
- **SpecializeFuncs**: no test for void-result evaluated func chaining
- **SplitPhases**: no test for "op uses value from later phase" `emitBug` (defensive check, hard to trigger from well-formed IR)
- **InferTypes**: no error test file

### Pass positive tests

- **HIRToMIR**: ~7 conversion patterns untested (`hir.mir_constant`, `hir.coerce_to_i1`, `arith.select`, `hir.opaque_pack`, `hir.opaque_unpack`, `hir.inferrable`, `hir.type_of`)
- **Interpret**: missing tests for `arith.select`, `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip, zero-result function
- **MIRToCIRCT**: missing tests for `!si.bool` type conversion, multi-instance counter, source materialization (iN→i1), zero-result functions

### Other focused test gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir` -- unless we've already removed the op
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

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
