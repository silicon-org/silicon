---
title: Variables
weight: 3
---

# Variables

Use `let` to bind a variable.
You can provide an explicit type annotation, or let the compiler infer it.

```silicon
fn main() -> int {
  let x: int = 10;
  let y = 20;
  x + y
}
```

The last expression in a block is its value.
Here `x + y` evaluates to `30` and becomes the return value of `main`.
