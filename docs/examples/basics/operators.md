---
title: Operators
weight: 5
---

# Operators

Silicon supports arithmetic, bitwise, comparison, and logical operators.

## Arithmetic

```silicon
fn arithmetic(a: int, b: int) -> int {
  let sum = a + b;
  let diff = a - b;
  let prod = a * b;
  let quot = a / b;
  let rem = a % b;
  sum
}
```

## Bitwise

```silicon
fn bitwise(a: int, b: int) -> int {
  let and = a & b;
  let or = a | b;
  let xor = a ^ b;
  let shl = a << b;
  let shr = a >> b;
  and
}
```

## Comparison

Comparison operators return `bool`.

```silicon
fn is_equal(a: int, b: int) -> bool { a == b }
fn less_than(a: int, b: int) -> bool { a < b }
```

## Logical

The `&&` and `||` operators are short-circuiting logical operators that work on `bool` values.
`a && b` only evaluates `b` if `a` is `true`.
`a || b` only evaluates `b` if `a` is `false`.

```silicon
fn in_range(x: int, lo: int, hi: int) -> bool {
  x >= lo && x <= hi
}
```

## Precedence

Operators follow the usual precedence rules.
Multiplication binds tighter than addition, shifts bind tighter than comparisons, and so on.

```silicon
fn precedence(a: int, b: int, c: int) -> int {
  a + b * c
}
```

This is parsed as `a + (b * c)`, not `(a + b) * c`.
Parentheses can be used to override the default precedence:

```silicon
fn override_precedence(a: int, b: int, c: int) -> int {
  (a + b) * c
}
```

See {{< page-link "/language/appendix/operators" >}} for the full precedence table.
