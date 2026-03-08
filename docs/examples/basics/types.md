---
title: Types
weight: 4
---

# Types

Silicon has the following types:

- `bool` — a boolean type with values `true` and `false`
- `int` — a generic signed integer
- `uint<N>` — an unsigned integer with a specific bit width

```silicon
fn add(a: int, b: int) -> int { a + b }
fn is_positive(x: int) -> bool { x > 0 }
```

The `uint` type is parameterized by its bit width, which can itself be an expression.
This means the type of one argument can depend on the *value* of another — a dependent type.
See {{< page-link "/examples/phased-execution" >}} for how `const` arguments make this possible.

Comparison operators (`==`, `!=`, `<`, `>`, `<=`, `>=`) always return `bool`.
The condition of an `if` expression must be a `bool`.

```silicon
fn max(a: int, b: int) -> int {
  if a > b { a } else { b }
}
```

Functions that don't return a value have an implicit unit type.

```silicon
fn returns_nothing() {}
```
