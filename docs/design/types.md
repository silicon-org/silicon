# Types

This document describes the type system of Silicon, covering the available types, type conversions, and casting conventions.
It builds on the inference mechanism described in {{< page-link "/design/inference" >}} and the phased execution model in {{< page-link "/design/phase-splits" >}}.

> [!WARNING]
> This document is a work in progress.

## Integer Types

Silicon has two integer types:

- **`int`**: An unbounded signed integer with no width specification.
  This type exists at compile time only and is used for metaprogramming, loop bounds, array sizes, and other compile-time computations.
  It cannot survive into the final hardware phase.

- **`uint<N>`**: An unsigned integer with explicit bit width N.
  This is the primary hardware integer type.
  The width N is itself a value, enabling dependent types where widths are computed from other values (e.g., `uint<a + 1>`).

A **`sint<N>`** (signed integer with explicit bit width) is planned but not yet designed.
Similarly, a **`bool`** type is planned; see below.

## Design Principles

The type system follows a few core principles around conversions:

1. **No implicit casts.**
   Conversions between types are always explicit in the source code.
   This prevents subtle bugs, especially around bit widths which are critical in hardware design.

2. **Lossless operations are low-noise.**
   Conversions that are provably lossless should be easy to write and read, with minimal syntactic overhead.

3. **Lossy operations are obviously lossy.**
   Any conversion that may lose information requires a naming convention that makes this visible at the call site.

4. **Consistent naming convention.**
   Cast functions follow a Rust-inspired tiered naming scheme that signals safety:

   | Suffix     | Behavior                      | When to use                                    |
   | ---------- | ----------------------------- | ---------------------------------------------- |
   | _(none)_   | Infallible, provably lossless | Always safe by construction                    |
   | `_checked` | Returns an option             | When the caller wants to handle failure        |
   | `_assert`  | Panics if lossy               | When failure is a bug, not a data condition    |
   | `_unsafe`  | Silently lossy                | When the caller knowingly discards information |

   The base name with no suffix is reserved for operations that the compiler can prove are lossless.
   If it can't prove this, the code does not compile — the programmer must pick a suffixed variant.

## Cast Operations

### `zext` — Zero-Extend

Widens a `uint<W>` to a `uint<M>` where `M > W`, filling the new high bits with zeros.
This is always lossless, so only the base form exists.

```
let x: uint<8> = ...;
let y: uint<16> = zext(x, 16);
```

The compiler verifies that the target width is strictly greater than the source width.
Equal widths are a no-op and should be a compile error (or warning) — use the value directly.

### `trunc` — Truncate

Narrows a `uint<W>` to a `uint<M>` where `M < W`, discarding the high bits.
This is inherently lossy, so no base form exists.

```
let y = trunc_assert(x, 4);  // panics if discarded bits are nonzero
let y = trunc_unsafe(x, 4);  // silently discards high bits
```

The `_checked` variant (returning an option) is planned but deferred; it can be implemented as a std lib function wrapping `trunc_unsafe` with a comparison.

### `reinterpret` — Reinterpret Cast

Reinterprets the bits of a value as a different type of the same width.
The primary use case is converting between `uint<N>` and `sint<N>`.

```
let y = reinterpret(x, sint<8>);  // uint<8> → sint<8>, same bits
```

This is deferred until `sint<N>` is designed.
Unlike the width-based casts, `reinterpret` takes a full target type rather than just a width, since the type changes fundamentally.

### Phase-Dependent Assertion Semantics

The `_assert` suffix produces different behavior depending on the phase of the value being converted:

- **Compile-time value**: The assertion fires at compile time, producing a compiler error.
- **Dynamic value in an earlier phase**: The assertion fires when that phase executes, producing a runtime error during compilation.
- **Final-phase (hardware) value**: The assertion becomes a simulation assertion.
  Synthesis behavior is an open question (strip, lower to SVA, etc.).

## Integer Literals and Type Inference

Integer literals are type-agnostic until inference resolves their type.
Codegen emits every literal as a `hir.constant_int` with an inferrable type operand:

```mlir
%ty = hir.inferrable
%c = hir.constant_int 42 : %ty
```

The surrounding context drives inference:

- `let x: int = 42` — `%ty` unifies with `int`, always valid.
- `let y: uint<8> = 42` — `%ty` unifies with `uint<8>`, CheckTypes verifies 42 fits in 8 bits.
- `let z: uint<1> = 42` — `%ty` unifies with `uint<1>`, CheckTypes reports an error.

CheckTypes runs after InferTypes, so the type is resolved by the time range checking happens.

## Materializing `int` to `uint<N>`

The conversion from `int` to `uint<N>` occurs at phase boundaries, when a compile-time integer value is materialized into a hardware-width integer.
This is not a user-facing cast operation — it happens as part of the lowering pipeline.

For compile-time constants (including literals), CheckTypes can statically verify that the value fits in N bits.

For computed `int` values that aren't known until interpretation, a `mir.int_to_uint` op handles the conversion at the MIR level:

```mlir
%result = mir.int_to_uint %computed_int : !si.uint<8>
```

The interpreter evaluates this and produces an error if the value doesn't fit.
Since `int` is a compile-time-only type, this conversion must be fully resolved before the final hardware phase.

## Boolean Conversions

Boolean conversions are handled by expressions, not cast operations:

- **`uint<N>` → `bool`**: Use a comparison: `x != 0`.
- **`bool` → `uint<N>`**: Use a conditional expression: `if b { 1 } else { 0 }`.

This keeps the language explicit and avoids overloading `zext` with cross-type semantics.
Convenience functions like `bool.as_uint(N)` are candidates for a future std lib.

## IR Representation

Cast ops in HIR carry only the value and a result type operand.
The type operand appears after `:`, mirroring the position where MLIR usually places types, but using SSA values instead (since all HIR values have MLIR type `!hir.any`):

```mlir
%result = hir.zext %value : %result_type
%result = hir.trunc_assert %value : %result_type
%result = hir.trunc_unsafe %value : %result_type
```

The width argument from the surface syntax (e.g., the `16` in `zext(a, 16)`) is unified with the result type's width through normal inference:

```mlir
// zext(a, 16)
%sixteen = hir.constant_int 16
%result_type = hir.uint_type %sixteen
%result = hir.zext %a : %result_type
```

This allows inference to flow in both directions.
For example, `let b: uint<42> = zext(a, _)` infers the width argument from context:

```mlir
// zext(a, _)
%w = hir.inferrable
%result_type = hir.uint_type %w
%result = hir.zext %a : %result_type
// %w resolves to 42 through unification with the let binding's type
```

### Type Checking

CheckTypes validates cast ops after inference has resolved all types:

- **`zext`**: Operand must be `uint<W>`, result must be `uint<M>`, and `M > W`.
- **`trunc_assert` / `trunc_unsafe`**: Operand must be `uint<W>`, result must be `uint<M>`, and `M < W`.
- **`constant_int`**: If the type resolved to `uint<N>`, the constant value must fit in N unsigned bits.

These are all straightforward post-inference checks on resolved types.

## Future Work

- **`sint<N>`**: Signed integer type with explicit bit width.
  Introduces `sext` (sign-extend, lossless widening) alongside `zext`.
- **`reinterpret`**: Same-width type reinterpretation, primarily for `uint<N>` ↔ `sint<N>`.
- **`_checked` variants**: Return an option type; useful for conditional truncation with validity tracking.
  Can be implemented as std lib functions wrapping the `_unsafe` variant.
- **Method syntax**: `a.zext(16)` instead of `zext(a, 16)`.
  Requires type-dependent function resolution.
- **Subtyping and coercion**: Not currently planned, but the design should not preclude it.
