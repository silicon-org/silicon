# TODO

- Codegen does not support `IfExpr` — `if`/`else` expressions fail with "unsupported expression kind" during AST-to-HIR conversion
- SplitPhases does not handle `const { ... }` blocks that contain calls; the ExprOp region isolation check fails because calls and their type ops span different phases (see `test/EndToEnd/const-block.si`)
- Dependent types (`uint<n>`) fail during HIR-to-MIR lowering with "block argument type could not be determined"
- Unused function args lose their `coerce_type` during canonicalization/DCE, causing HIR-to-MIR to fail with "block argument type could not be determined" (affects all functions, e.g. `fn foo(a: int, b: int) -> int { a }`)
- Update docs silicon code blocks for `dyn` args, `const`/`dyn` return types, and function-level modifiers once the above bugs are fixed
- Use Iosevka font for code blocks in docs, if Hugo supports customizing the font
- In the rendered version of the operators.md doc, the headers `## Bitwise` and `## Precedence` don't get rendered as headings in the HTML, but just as verbatim `## ...`; this is probably a bug in the custom syntax highlighting
- The HIR-to-MIR lowering uses a fallback map from split_func signatures when coerce_type ops on block args have been DCE'd; consider making coerce_type not Pure on block args instead, then get rid of the fallback in HIRToMIR
