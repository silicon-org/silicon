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
- [x] Reference types and operations (`&T`, `*x = 4`, `4 + *x`)
- [ ] Dynamic types (`dyn T`)
- [x] Parametric functions
- [ ] Unit tests and execution
- [ ] Assertions
- [ ] If expressions and loop statements
- [ ] Clock domain associations for values
- [x] Tuple types
- [ ] Array types
- [ ] Enum types
- [ ] Struct types
- [x] Constraints on ports/types in module signature, checked inside/outside
- [ ] Make `return` an expression
- [x] Make blocks an expression

## Setup

Clone the repository and the submodules:
```
git clone --recurse-submodules --shallow-submodules git@github.com:silicon-org/silicon.git
cd silicon
```

Configure the build:
```
cmake -G Ninja -B build ../circt/llvm/llvm \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS="circt;silicon" \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD/circt \
  -DLLVM_EXTERNAL_SILICON_SOURCE_DIR=$PWD \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
