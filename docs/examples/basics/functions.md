---
title: Functions
weight: 2
---

# Functions

Functions are declared with `fn`, followed by a name, parameters with type annotations, and an optional return type.

```silicon
fn add(a: int, b: int) -> int { a + b }
```

Functions can call other functions.
Arguments are passed by position.

```silicon
fn square(x: int) -> int { x * x }
fn sum_of_squares(a: int, b: int) -> int { square(a) + square(b) }
```

## Visibility

By default, functions are private.
Use `pub` to make a function visible outside its module.

```silicon
pub fn visible() -> int { 42 }
fn helper() -> int { 1 }
fn main() -> int { visible() + helper() }
```
