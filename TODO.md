# TODO

- Rename `AnyHIRType` to `HIRAnyType`
- Add a type operand to `hir.constant_int`, such that we can infer the type to be `int` or `uint<N>`.
  When we infer the concrete type of such a constant, also check that the integer fits into the chosen type.
  We may want to do this as part of type inference, since this is a user-facing error.
- Investigate why the `eraseVoidCalls` is needed in function specialization; is this a hack?
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
- **`if` conditions accept any type without a type error.**
  `if cond { ... }` where `cond: int` silently works because `coerce_to_i1` doesn't enforce a bool type.
  The old `hir.if` also didn't check, so this isn't a regression, but we should add a proper check.
  Codegen should unify the condition with `bool` and emit a type error for non-bool conditions; no implicit truthiness.
- **`getTypeOf` returns null for block arguments.**
  After the IfOp removal, block arguments at CFG merge points have no type metadata.
  `getOrCreateTypeOf` falls back to inserting a `TypeOfOp`, which works but is less efficient than directly resolving the type.
  Consider teaching `getTypeOf` to look at predecessor branch operands, with a dominance check to avoid returning values from non-dominating blocks.

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

## Focused Test Coverage Gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir`
- Add unused block args test case to `test/Conversion/hir-to-mir.mlir`
- Add per-argument `dyn` and mixed const+dyn arg cases to `test/HIR/split-phases.mlir`
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
