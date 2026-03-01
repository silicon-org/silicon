---
weight: 5
---

# Getting Started

This guide walks you through cloning, building, and exploring the Silicon compiler.

## Prerequisites

You will need:

- **CMake** (3.13.4 or later)
- **Ninja** build system
- **C++17** compiler (GCC or Clang)
- **Python 3** (for running lit tests)

## Clone

Clone the repository with its submodules (CIRCT, MLIR, LLVM):

```sh
git clone --recurse-submodules --shallow-submodules \
  git@github.com:silicon-org/silicon.git
cd silicon
```

The `--shallow-submodules` flag keeps the download manageable by not pulling full history for CIRCT and LLVM.

## Configure

Run CMake to configure the build.
Silicon is built as an external project inside the LLVM build system, so the source directory points at LLVM:

```sh
cmake -S circt/llvm/llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_EXTERNAL_PROJECTS="circt;silicon" \
  -DLLVM_EXTERNAL_CIRCT_SOURCE_DIR=$PWD/circt \
  -DLLVM_EXTERNAL_SILICON_SOURCE_DIR=$PWD \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

This produces a `build/` directory with all the build artifacts.

## Build

Build and run all tests in one go.
Tests live under `test/` and use [LLVM lit](https://llvm.org/docs/CommandGuide/lit.html) with `FileCheck`.
This is the main driver for development:

```sh
ninja -C build check-silicon
```

Or build individual tools:

```sh
ninja -C build silc           # the silc compiler CLI
ninja -C build silicon-opt    # the silicon-opt IR utility
ninja -C build silicon-tools  # both of the above
```

Or build various parts of the Silicon project:

```sh
ninja -C build silicon            # build everything
ninja -C build silicon-docs       # build documentation
ninja -C build silicon-website    # build website (requires hugo)
ninja -C build silicon-tools      # build tools such as silc and silicon-opt
ninja -C build silicon-libraries  # build everything else
```

The first build takes a while since it compiles the relevant parts of LLVM, MLIR, and CIRCT.
Subsequent incremental builds are fast.

## Useful Commands

### Compile a Silicon source file

```sh
build/bin/silc input.si
```

### Run a pass on an MLIR file

```sh
build/bin/silicon-opt input.mlir --my-pass-name
```

This is the main workflow when developing or debugging transformation passes.
You can add `--debug-only=<pass>` to print all `LLVM_DEBUG(...)` output from a specific pass.

## Project Layout

```
circt/             CIRCT dependency (submodule)
  llvm/            LLVM dependency (submodule)
    mlir/          MLIR dependency
docs/              documentation and website source
include/silicon/   public headers
  Codegen/         conversion from AST to HIR
  Conversion/      conversion passes between dialects
  HIR/             high-level dialect (types are SSA values)
  MIR/             mid-level dialect (types are materialized)
  Support/         various utilities
  Syntax/          lexer, parser, AST
  Transforms/      multi-dialect passes
lib/               implementation files (mirrors include/silicon/)
test/              llvm-lit tests (mirrors include/silicon/)
tools/
  silc/            main compiler CLI
  silicon-opt/     utility to run individual passes
```

## Where to Start

- **Working on syntax** (lexer, parser, AST): edit under `lib/Syntax/` and `include/silicon/Syntax/`, test with `silc`.
- **Working on passes** (transformations, lowerings): edit under `lib/Transforms/` or `lib/Conversion/`, test with `silicon-opt` and MLIR input files.
- **Adding a new IR op**: define it in the relevant `.td` file under `include/silicon/`, then add parsing tests to the corresponding `basic.mlir` and verifier tests to `errors.mlir`.
- **Reading the design docs**: see [Design]({{< relref "design" >}}) for the phased execution model and HIR/MIR split.

Before committing, run `clang-format` on any changed C++ files.
Formatting is enforced by CI before merging.
