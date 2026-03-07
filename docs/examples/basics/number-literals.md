---
title: Number Literals
weight: 7
---

# Number Literals

Number literals can be written in decimal, hexadecimal, binary, or octal.
Underscores can be used as visual separators anywhere in a number.

```silicon
fn main() -> int {
  let decimal = 1_000;
  let hex = 0xff;
  let binary = 0b1010;
  let octal = 0o755;
  decimal
}
```
