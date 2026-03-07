---
title: Types
weight: 4
---

# Types

Silicon currently has two numeric types:

- `int` — a generic signed integer
- `uint<N>` — an unsigned integer with a specific bit width

```silicon
fn add(a: int, b: int) -> int { a + b }
```

The `uint` type is parameterized by its bit width, which can itself be an expression.
This means the type of one argument can depend on the *value* of another — a dependent type.
See {{< page-link "/examples/phased-execution" >}} for how `const` arguments make this possible.

Functions that don't return a value have an implicit unit type.

```silicon
fn returns_nothing() {}
```
