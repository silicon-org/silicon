# TODO

- Codegen does not support `IfExpr` — `if`/`else` expressions fail with "unsupported expression kind" during AST-to-HIR conversion
- Codegen does not support `ReturnExpr` — `return` statements fail with "unsupported expression kind" during AST-to-HIR conversion
- `const { ... }` blocks fail with "phase eval loop made no progress" when used in a function body
- Dependent types (`uint<n>`) fail during HIR-to-MIR lowering with "block argument type could not be determined"
- Nested block comments (`/* outer /* inner */ outer */`) are not handled correctly by the lexer — only the first `*/` is consumed
- Add `dyn` keyword to func args to do the opposite of const
- Add `dyn { ... }` expression to do the opposite of `const { ... }`
- Allow entire functions to be marked as `const` or `dyn`, or any multiple repetition thereof; which shifts all phases in the signature and the body accordingly
- Since we create copies or symlinks anyway, move hugo.toml, themes, layouts, and chromaLexers in docs/ into a subdirectory docs/hugo/ and adjust the CMake
