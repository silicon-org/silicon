# Silicon

An experimental hardware language and compiler. Requires xdsl 0.21.1 or higher.

Rough roadmap:

- [x] Modules
- [ ] Module instances
- [x] Port declarations
- [x] Let bindings
- [x] Type inference
- [x] Basic Intermediate Representation
- [x] Canonicalize IR
- [x] Functions and calls
- [ ] Reference types and operations (`&T`, `*x = 4`, `4 + *x`)
- [x] Parametric functions
- [ ] Unit tests and execution
- [ ] Assertions
- [ ] If expressions and loop statements
- [ ] Clock domain associations for values
- [x] Tuple types
- [ ] Array types
- [ ] Enum types
- [ ] Struct types
- [ ] Constraints on ports/types in module signature, checked inside/outside
- [ ] Make `return` an expression
- [ ] Make blocks and expression
