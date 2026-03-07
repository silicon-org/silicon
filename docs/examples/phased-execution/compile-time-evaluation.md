---
title: Compile-Time Evaluation
weight: 3
---

# Compile-Time Evaluation

When all arguments to a function are known at compile time, the compiler evaluates the entire function and replaces the call with the result.

```silicon
fn sub_const(const a: int, b: int) -> int { a - b }
fn main_sub() -> int { sub_const(10, 3) }

fn mul_const(const a: int, b: int) -> int { a * b }
fn main_mul() -> int { mul_const(4, 5) }

fn div_const(const a: int, b: int) -> int { a / b }
fn main_div() -> int { div_const(10, 3) }

fn mod_const(const a: int, b: int) -> int { a % b }
fn main_mod() -> int { mod_const(10, 3) }
```

The compiler evaluates each `main_*` function entirely at compile time, producing the constant results `7`, `20`, `3`, and `1`.

This is a key building block of Silicon's metaprogramming: by marking the right arguments as `const`, you can move arbitrary computation into the compiler, generating specialized runtime code from compile-time results.
