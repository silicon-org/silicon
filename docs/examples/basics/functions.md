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

## Explicit Return

Use `return` to return a value from a function early.
If no `return` is used, the function returns the value of its last expression.

```silicon
fn always_seven() -> int { return 7 }
```

## Visibility

By default, functions are private.
Use `pub` to make a function visible outside its module.

```silicon
pub fn visible() -> int { 42 }
fn helper() -> int { 1 }
fn main() -> int { visible() + helper() }
```

## Phase Modifiers on Arguments

Function arguments can be annotated with `const` or `dyn` to shift their phase.
A `const` argument is evaluated one phase earlier (at compile time), while a `dyn` argument is deferred one phase later (to runtime).

```silicon
fn add_const(const a: int, b: int) -> int { a + b }
```

```
fn send(dyn x: int, y: int) -> int { x + y }
```

Return types can also carry phase modifiers.

```
fn make_const(x: int) -> const int { x }
fn make_dyn(x: int) -> dyn int { x }
```

See the [Phased Execution]({{< relref "/examples/phased-execution" >}}) section for a deeper explanation of how phases work.

## Function-Level Phase Modifiers

Placing `const` or `dyn` before the `fn` keyword shifts *all* phases of the function by the modifier amount.
This is a shorthand that affects every argument and the return type uniformly.

```silicon
const fn early(a: int) -> int { a + 1 }
dyn fn late(a: int) -> int { a + 1 }
```

Multiple modifiers stack: `const const fn` shifts by -2, and `const dyn fn` cancels out to a shift of 0.

```silicon
const const fn very_early(a: int) -> int { a }
```
