# Introduction

Silicon is an modern hardware description language and compiler, with ergonomic syntax inspired by Rust and Zig.
It allows for rich metaprogramming in the language itself through phased compile-time execution.
Think of it as Zig comptime or C++ templates on steroids.

The compiler is built on top of [CIRCT](https://circt.llvm.org/), [MLIR](https://mlir.llvm.org/), and [LLVM](https://llvm.org/).

This documentation is organized into a few distinct parts:

- [Getting Started]({{< relref "getting-started" >}}) -- clone, build, and explore the project
- [Design]({{< relref "design" >}}) -- goals and architecture of the language and compiler
- [Internals]({{< relref "internals" >}}) -- compiler IR dialects and transformation passes
