# TODO

- Codegen does not support `IfExpr` — `if`/`else` expressions fail with "unsupported expression kind" during AST-to-HIR conversion
- SplitPhases does not handle `const { ... }` blocks that contain calls; the ExprOp region isolation check fails because calls and their type ops span different phases (see `test/EndToEnd/const-block.si`)
- Dependent types (`uint<n>`) fail during HIR-to-MIR lowering with "block argument type could not be determined"
- Since we create copies or symlinks anyway, move hugo.toml, themes, layouts, and chromaLexers in docs/ into a subdirectory docs/hugo/ and adjust the CMake
- Update docs with the recently added features
- Add clang waiver comment on recursion in the parser; see examples in CIRCT
