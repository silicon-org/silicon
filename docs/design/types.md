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

## Tuple Types

Tuples are ordered, fixed-size collections of values.
They are the primary aggregate type in Silicon, used for grouping related values and for multi-return functions.

### Surface Syntax

Tuple literals use parenthesized, comma-separated expressions:

```
let pair: (uint<8>, uint<16>) = (a, b);
let x = pair.0;
let y = pair.1;
```

Fields are accessed by index using `.0`, `.1`, etc.
Tuples are always positional — fields do not have names.

Tuples can contain any combination of types, including `int`:

```
let mixed: (int, uint<8>) = (42, x);
```

A tuple containing `int` fields cannot survive into the final hardware phase.
The compiler reports an error if such a tuple reaches hardware lowering without all `int` fields having been resolved.

### No Bit-Width or Packed Layout

Unlike integer types, tuples do not define a bit width or a packed bit layout.
A `(uint<8>, uint<8>)` is not interchangeable with a `uint<16>`.
If the user needs to pack or unpack values into a flat bit vector, they should do so explicitly using shifts, concatenation, and truncation — or via helper functions, which are a great use case for const-phase metaprogramming.

### Multi-Return Functions

Functions can declare multiple named results using a tuple-like syntax after `->`:

```
fn adder(a: uint<8>, b: uint<8>) -> (sum: uint<9>, overflow: uint<1>) {
    let full = zext(a, 9) + zext(b, 9);
    return (trunc_unsafe(full, 9), trunc_unsafe(full >> 9, 1));
}
```

The result names and phase annotations are part of the function declaration, not the tuple type.
They serve as documentation and are stored as metadata on the function (as `FnRes` entries tracking name, phase, and type), but the caller receives an ordinary unnamed tuple:

```
let result = adder(x, y);
result.0    // the sum
result.1    // the overflow
```

The number of results determines the return and call conventions:

- **0 results**: `return` or `return ()`. Calls produce `()` (unit).
- **1 result**: `return x`. Calls produce the result value directly, not a 1-tuple.
- **>1 results**: `return (a, b, c)`. Calls produce a tuple.

The parentheses in `return (a, b, c)` are required for multi-return, making it visually clear that a tuple is being constructed.

### Destructuring

Tuple destructuring in `let` bindings and patterns is planned but deferred to keep the initial implementation simple:

```
// Future:
let (sum, overflow) = adder(x, y);
```

### Per-Field Phases

> [!NOTE]
> This section is a forward-looking note. The design is not yet fleshed out.

In the future, individual tuple fields (and later struct fields) may carry their own phase annotations, such as `(const uint<8>, uint<8>)`.
This would allow a single aggregate to span multiple phases, with some fields resolved at compile time and others surviving into hardware.
The semantics of such mixed-phase aggregates — when the tuple itself "exists", how it interacts with phase boundaries, and whether it can be partially materialized — remain open questions.

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

### Tuple Ops

Tuple types, construction, and field access in HIR:

```mlir
// Type construction
%t0 = hir.uint_type %eight
%t1 = hir.uint_type %sixteen
%tuple_ty = hir.tuple_type %t0, %t1

// Tuple creation
%tup = hir.tuple_create %a, %b : %tuple_ty

// Field access
%x = hir.tuple_get %tup[0] : %elem_ty
%y = hir.tuple_get %tup[1] : %elem_ty
```

In MIR, the tuple type is a concrete MLIR type `!si.tuple<...>`:

```mlir
%tup = mir.tuple_create %a, %b : !si.tuple<!si.uint<8>, !si.uint<16>>
%x = mir.tuple_get %tup[0] : !si.uint<8>
```

#### Multi-Return Functions in IR

At the IR level, functions with multiple results use MLIR's native multi-result mechanism.
The surface-level tuple is destructured on return and reconstructed at call sites:

```mlir
// fn adder(a: uint<8>, b: uint<8>) -> (sum: uint<9>, overflow: uint<1>)
hir.func @adder(%a, %b) -> (%sum_ty, %ovf_ty) {
    ...
    hir.return %sum, %ovf : %sum_ty, %ovf_ty
}

// let result = adder(x, y);
%sum, %ovf = hir.call @adder(%x, %y) : ...
%result = hir.tuple_create %sum, %ovf : %result_tuple_ty
```

This means that when all uses of `result` are `.0` and `.1` accesses, canonicalization folds through the `tuple_create`/`tuple_get` pair and the tuple disappears entirely, leaving clean multi-result SSA:

```mlir
// After canonicalization of: result.0 + result.1
%sum, %ovf = hir.call @adder(%x, %y) : ...
%r = hir.add %sum, %ovf : ...
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
- **Tuple destructuring**: `let (a, b) = expr` in bindings and match patterns.
- **Named tuple fields**: Allowing field names on tuples (or introducing lightweight struct syntax) for ergonomic access to multi-return results by name.
- **Per-field phase annotations**: Mixed-phase aggregates like `(const uint<8>, uint<8>)`, allowing tuples and structs to span phase boundaries.
- **Structs**: Named aggregate types, building on the tuple IR infrastructure.
