# TODO

- Codegen does not support `IfExpr` — `if`/`else` expressions fail with "unsupported expression kind" during AST-to-HIR conversion
- SplitPhases does not handle `const { ... }` blocks that contain calls; the ExprOp region isolation check fails because calls and their type ops span different phases (see `test/EndToEnd/const-block.si`)
- Dependent types (`uint<n>`) fail during HIR-to-MIR lowering with "block argument type could not be determined"
- `dyn` arguments crash with vector assertion failure during full compilation (parsing works fine)
- `const`/`dyn` return types fail with "resultNames has 0 entries but return has 1 values" during full compilation
- `const fn`/`dyn fn` function-level phase modifiers fail with "resultNames has N entries but return has 1 values" during full compilation
- Update docs silicon code blocks for `dyn` args, `const`/`dyn` return types, and function-level modifiers once the above bugs are fixed
- Use Iosevka font for code blocks in docs, if Hugo supports customizing the font
