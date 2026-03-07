---
title: Operators
weight: 5
---

# Operators

Silicon supports the usual arithmetic, bitwise, and comparison operators.

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

## Precedence

Operators follow the usual precedence rules.
Multiplication binds tighter than addition, shifts bind tighter than comparisons, and so on.

```silicon
fn precedence(a: int, b: int, c: int) -> int {
  a + b * c
}
```

This is parsed as `a + (b * c)`, not `(a + b) * c`.
See [Appendix: Operators and Symbols]({{< relref "language/appendix/operators" >}}) for the full precedence table.
