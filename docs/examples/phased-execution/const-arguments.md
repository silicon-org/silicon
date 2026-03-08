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

## Dyn Arguments

The `dyn` keyword is the opposite of `const`: it shifts an argument to a *later* phase.
While `const` pulls computation earlier (toward compile time), `dyn` pushes it later (toward runtime).

```silicon
fn send(dyn x: int, y: int) -> int { x + y }
```

Here `x` is deferred to a later phase than the rest of the function.

## Function-Level Modifiers

Instead of annotating individual arguments, you can place `const` or `dyn` before the `fn` keyword to shift *all* phases of a function uniformly.

```silicon
const fn early(a: int, b: int) -> int { a + b }
dyn fn late(a: int, b: int) -> int { a + b }
```

Multiple modifiers stack: `const const fn` shifts by -2, while `const dyn fn` cancels out to a net shift of 0.
