---
title: Keywords
weight: 1
---

# Appendix: Keywords

The Silicon language uses a small set of keywords that are reserved and cannot be used as identifiers.
This appendix provides a quick reference for all keywords, organized by whether they are currently in use or reserved for future functionality.

## Keywords Currently in Use

The following keywords currently have functionality in the Silicon language.

- **`const`**: Mark a function argument or result as computed in an earlier phase and being constant now, wrap an expression in a `const { ... }` block to evaluate before the surrounding code runs, or place before `fn` to shift all phases of a function earlier. Multiple modifiers stack: `const const fn` shifts by -2.
- **`dyn`**: Mark a function argument or result as computed in a later phase and being unknown now, wrap an expression in a `dyn { ... }` block to evaluate after the surrounding code runs, or place before `fn` to shift all phases of a function later. Modifiers can be mixed: `const dyn fn` cancels out to a net shift of 0.
- **`else`**: Define a fallback branch in an `if` expression.
- **`fn`**: Declare a function.
- **`if`**: Branch conditionally based on an expression.
- **`int`**: The generic signed integer type.
- **`let`**: Bind a variable in the current scope, with an optional type annotation and initializer.
- **`pub`**: Make a function visible outside its module.
- **`return`**: Return a value from the current function early.
- **`uint`**: An unsigned integer type parameterized by bit width, as in `uint<32>`.

## Keywords Reserved for Future Use

The following keywords have no functionality yet, but are reserved by the language so that they can be used in the future.
Silicon will produce an error if you try to use any of these as an identifier.

- `for`
- `loop`
- `match`
- `while`
