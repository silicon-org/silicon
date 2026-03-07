# TODO

- SplitPhases does not handle `const { ... }` blocks that contain calls; the ExprOp region isolation check fails because calls and their type ops span different phases (see `test/EndToEnd/const-block.si`)
- Update docs silicon code blocks for `dyn` args, `const`/`dyn` return types, and function-level modifiers once the relevant bugs are fixed
- Fix unused-args.si: `CoerceTypeOp` is now `Pure`, so DCE removes coerce_type on unused block args; make the return op carry argument types of the function, like `: (%arg0.ty, %arg1.ty, ...) : (%res0.ty, %res1.ty, ...)` and use these operands to determine an HIR function's arg types instead of type coercion
- Fix dyn-args.si: SplitPhases `coerce_type` on `dyn` args triggers "op uses value from later phase" and region isolation errors
- It looks like symbol DCE does not run: even with a simple example, the output still contains private split funcs, even if there are no more calls to them
- Make split func discardable by removing that `canDiscardOnUseEmpty` override
- Fix formatting of custom-printed func ops when `attributes {isModule}` is present: the printer emits a double space before `attributes` (one trailing space after `)` plus one from the attr-dict printer) and no space between the closing `}` of the attr-dict and the opening `{` of the body region (e.g., `attributes {isModule}{` instead of `attributes {isModule} {`); affects `hir.func`, `hir.unified_func`, and `mir.func`
- The input `mod hello(a: uint<42>) {}` to silc triggers the error `error: 'hir.uint_type' op using value defined outside the region`
- Replace `const` attribute on `hir.expr` with a dedicated mandatory `phaseShift` attribute that is included in the assembly format like `hir.expr $phaseShift ...`

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
