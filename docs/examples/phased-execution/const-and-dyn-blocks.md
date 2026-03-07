---
title: Const and Dyn Blocks
weight: 4
---

# Const and Dyn Blocks

In addition to annotating function arguments, you can use `const { ... }` and `dyn { ... }` blocks to control the phase of individual expressions within a function body.

## Const Blocks

A `const { ... }` block forces the enclosed expression to evaluate one phase earlier than the surrounding code.
This is useful when you want to compute a value at compile time without making it a separate function argument.

```silicon
fn example(x: int) -> int {
    const { 2 + 3 } + x
}
```

Here `2 + 3` is evaluated at compile time, producing `5`, and the remaining code adds `x` at runtime.

## Dyn Blocks

A `dyn { ... }` block is the opposite: it defers the enclosed expression to one phase later than the surrounding code.

```silicon
fn example(const x: int) -> int {
    x + dyn { 1 }
}
```

Here `x` is a compile-time value, but `dyn { 1 }` pushes the literal `1` to a later phase, preventing the entire expression from being folded at compile time.
