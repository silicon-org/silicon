---
title: Const Arguments
weight: 1
---

# Const Arguments

Marking a function argument as `const` tells the compiler to evaluate it one phase earlier — at compile time.
The compiler splits the function into separate phases: one that runs at compile time with the `const` arguments, and one that runs later with the remaining arguments.

```silicon
fn add(const a: int, b: int) -> int { a + b }
fn main() -> int { add(42, 7) }
```

Here `a` is known at compile time.
The compiler evaluates `42` during compilation, and the remaining runtime code only needs to work with `b`.

When `main` calls `add(42, 7)`, the compiler can evaluate the entire expression at compile time, producing the constant result `49`.
