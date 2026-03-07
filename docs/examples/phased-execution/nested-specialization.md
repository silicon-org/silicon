---
title: Nested Specialization
weight: 2
---

# Nested Specialization

When a function with `const` arguments calls another function with `const` arguments, the compiler specializes them transitively.
Each function is split into phases, and the compile-time results flow from callee to caller.

```silicon
fn double(const x: int, y: int) -> int { x + x + y }
fn quad(const x: int, y: int) -> int { double(x, y) + double(x, y) }
fn main() -> int { quad(5, 1) }
```

The compiler processes this in phases:

1. **Phase -1 of `double`**: evaluates `x + x` at compile time. For `x = 5`, this produces `10`.
2. **Phase -1 of `quad`**: calls `double`'s compile-time phase twice, both with `x = 5`.
3. **Phase 0 of `main`**: calls `quad`'s compile-time phase with `x = 5`, then its runtime phase with `y = 1`. The final result is `10 + 1 + 10 + 1 = 22`.
