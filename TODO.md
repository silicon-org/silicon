# TODO

- SplitPhases does not handle `const { ... }` blocks that contain calls; the ExprOp region isolation check fails because calls and their type ops span different phases (see `test/EndToEnd/const-block.si`)
- Update docs silicon code blocks for `dyn` args, `const`/`dyn` return types, and function-level modifiers once the above bugs are fixed
- Fix unused-args.si: `CoerceTypeOp` is now `Pure`, so DCE removes coerce_type on unused block args; HIR-to-MIR needs an alternative way to determine arg types
- Fix dyn-args.si: SplitPhases `coerce_type` on `dyn` args triggers "op uses value from later phase" and region isolation errors
