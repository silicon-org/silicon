---
title: Hello World
weight: 1
---

# Hello World

The simplest Silicon program is a function that returns a value.
Silicon programs are made up of functions, and the last expression in a function body is its return value — no `return` keyword needed.

```silicon
fn main() -> int { 42 }
```

Functions with no return value have an implicit unit return type.

```silicon
fn do_nothing() {}
```

You can compile any Silicon file with `silc`:

```sh
silc hello.si
```
