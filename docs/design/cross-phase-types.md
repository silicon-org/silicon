# Cross-Phase Type Influence

This document explores examples where computation in earlier phases influences the type structure of subsequent phases.
Examples start simple and build to complex cases.
Each example shows the Silicon source (or MLIR when the frontend can't express it), the expected phase expansion, and the current compiler behavior.

## Terminology

- **Phase -N**: A compile-time phase, N steps earlier than the function body.
- **Dependent type**: A type whose structure depends on a runtime or const value (e.g., `uint<N>` where `N` is a const argument).
- **Type threading**: A const value computed in an earlier phase flows through opaque context to determine the type of a value in a later phase.

## Example 1: Const Arg as Type Width

The simplest case: a `const` argument `N` directly determines the `uint<N>` type of a later-phase argument.

```silicon
fn id(const N: int, x: uint<N>) -> uint<N> { x }
pub fn main() -> uint<8> { id(8, 42) }
```

**Expected phase expansion:**

1. Phase -1 of `id`: receives `N`, packs `N` and derived type `uint<N>` into context.
2. Phase 0 of `id`: unpacks context, has `x: uint<N>` with N now concrete (8), returns `x`.
3. `main` specializes `id` with N=8. The specialized `id` becomes `uint<8> -> uint<8>`.

**Expected final output:** `42 : !si.uint<8>`.

**Parsed HIR (works):**

```mlir
hir.unified_func private @id(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N       // uses block arg %N
  %2 = hir.uint_type %N
  hir.signature (%0, %1) -> (%2)
} {
  %0 = hir.type_of %x
  hir.return %x -> (%0)
}
```

**Compiler result: FAILS** in the `check-calls` pass with a dominance error.

**Root cause:** `CheckCalls::cloneSignatureIntoBody` splices signature ops into the body, then inserts `coerce_type` ops for block args.
The `replaceUsesWithIf` on the block args replaces uses in the just-spliced `uint_type` ops with the `coerce_type` results, but the `coerce_type` is defined *after* the `uint_type`, creating `uint_type %coerce_result` where `%coerce_result` does not yet dominate.

This is a bug in CheckCalls, not a fundamental limitation.
The existing `dependent-type.mlir` test avoids it because its signature uses `%T` directly (the block arg) in the `hir.signature` terminator without intermediate `uint_type` ops.

## Example 2: Computed Type Width

A const argument is transformed by arithmetic before use as a type width.

```silicon
fn double_width(const N: int, x: uint<N>) -> uint<N + N> {
  x + x
}
```

**Status:** Not testable — hits the same CheckCalls bug as Example 1, and additionally `uint<N + N>` requires the parser to emit `hir.uint_type(hir.add(%N, %N))` in the signature, which is a computed dependent type.
This is aspirational and likely requires future work on type-level arithmetic.

## Example 3: Const Computation Flows into Type

A fully-const function computes a value that becomes a type width.

```silicon
fn compute_width(const a: int, const b: int) -> int { a + b }
fn use_width(const W: int, x: uint<W>) -> uint<W> { x }
pub fn main() -> uint<8> { use_width(compute_width(3, 5), 42) }
```

**Expected:** `compute_width(3, 5)` evaluates to 8 at compile time.
`use_width(8, 42)` specializes with W=8.

**Compiler result: FAILS** — same CheckCalls dominance bug on `use_width`.
Parsing succeeds and correctly wraps `compute_width(3, 5)` in an `hir.expr` block.

## Example 4: Nested Specialization with Dependent Types

An inner function's type depends on an outer function's const computation.

```silicon
fn inner(const N: int, x: uint<N>) -> uint<N> { x }
fn outer(const M: int, x: uint<M>) -> uint<M> { inner(M, x) }
pub fn main() -> uint<16> { outer(16, 100) }
```

**Expected:** Both `inner` and `outer` are specialized.
The type `uint<16>` propagates through both levels.

**Compiler result: FAILS** — same CheckCalls dominance bug on both `inner` and `outer`.

## Example 5: Const Block Producing a Type Width

A `const { ... }` block computes a value used as a type width argument.

```silicon
fn identity(const N: int, x: uint<N>) -> uint<N> { x }
pub fn main(x: uint<8>) -> uint<8> {
  identity(const { 4 + 4 }, x)
}
```

**Expected:** `const { 4 + 4 }` evaluates to 8 at compile time.
`identity(8, x)` is specialized.

**Compiler result: FAILS** — same CheckCalls dominance bug on `identity`.

## Example 6: Conditional Type Selection

A const condition selects between different type widths.

```silicon
fn select_width(const wide: bool, const N: int) -> int {
  if wide { N + N } else { N }
}
fn typed_op(const W: int, x: uint<W>) -> uint<W> { x }
pub fn main(x: uint<16>) -> uint<16> {
  typed_op(select_width(true, 8), x)
}
```

**Expected:** `select_width(true, 8)` evaluates to 16.
`typed_op(16, x)` is `uint<16> -> uint<16>`.

**Compiler result: FAILS** — same CheckCalls dominance bug on `typed_op`.

## Example 7: Three-Phase Type Threading (MLIR)

Phase -2 computes a value, phase -1 uses it to construct a type, phase 0 uses the result.
Since this can't be expressed in the frontend (the frontend doesn't support `const const`), we test directly at the MLIR level.

### Unified form

```mlir
hir.unified_func @triple(%N: -2, %x: -1) -> (result: -1) {
  %type_type = hir.type_type
  %int_type = hir.int_type
  %T = hir.uint_type %N
  hir.signature (%int_type, %T) -> (%T)
} {
  %T = hir.uint_type %N
  hir.return %x -> (%T)
}

hir.unified_func @main() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} {
  %int_type = hir.int_type
  %type_type = hir.type_type
  %c8 = hir.constant_int 8 : %int_type
  %c42 = hir.constant_int 42 : %int_type
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %r = hir.unified_call @triple(%c8, %c42) : (%t0, %t1) -> (%t2) (!hir.any, !hir.any) -> !hir.any [-2, -1] -> [-1]
  %tr = hir.type_of %r
  hir.return %r -> (%tr)
}
```

**SplitPhases: WORKS.** Correctly splits into three phases:

- `@triple.0`: phase -2, receives N, packs empty context.
- `@triple.1`: phase -1, receives x and context, unpacks, returns x.
- `@triple.2`: phase 0, receives context, does nothing (result was at phase -1).

### Pre-split form (bypassing CheckCalls)

To test the PhaseEvalLoop, we can provide pre-split IR that has already been through check-calls, infer-types, and split-phases:

```mlir
// triple phase -2: receives N (int), returns packed context
hir.func private @triple.0(%N) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.int_type
  %1 = hir.coerce_type %N, %0
  %2 = hir.type_of %1
  %3 = hir.opaque_pack(%1)
  %4 = hir.opaque_type
  hir.return %3 -> (%4)
}

// triple phase -1: receives x, context from phase -2; returns x
hir.func private @triple.1(%x, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %2 = hir.opaque_type
  hir.signature (%0, %1) -> (%2)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.uint_type %0
  %2 = hir.coerce_type %x, %1
  hir.return %2 -> (%1)
}

hir.split_func @triple(%N: -2, %x: -1) -> (result: -1) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1) -> (%1)
} [
  -2: @triple.0,
  -1: @triple.1
]

// main phase 0a: calls triple phase -2 with N=8, packs ctx
hir.func private @main.0a() -> (ctx) {
  %0 = hir.opaque_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.opaque_type
  %3 = hir.call @triple.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  %5 = hir.opaque_type
  hir.return %4 -> (%5)
}

// main phase 0b: calls triple phase -1 with x=42 and packed ctx
hir.func private @main.0b(%ctx) -> (ctx) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.constant_int 42 : %1
  %3 = hir.opaque_type
  %4 = hir.opaque_type
  %5:1 = hir.call @triple.1(%2, %0) : (%1, %3) -> (%4)
  %6 = hir.opaque_pack(%5#0)
  %7 = hir.opaque_type
  hir.return %6 -> (%7)
}

// main phase 0c: unpacks the result
hir.func private @main.0c(%ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.type_of %0
  hir.return %0 -> (%1)
}

hir.split_func @main() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} [
  0: @main.0
]

hir.multiphase_func @main.0() -> (result) [
  @main.0a,
  @main.0b,
  @main.0c
]
```

**PhaseEvalLoop: PARTIALLY WORKS.**

- Iteration 0: `@triple.0` and `@main.0a` are successfully lowered to MIR.
  `@main.0a` is interpreted, producing `#si.opaque<[#si.opaque<[#si.int<8>]>]>`.
- Iteration 1: `SpecializeFuncs` creates `@triple.1_0`, the specialized version with `mir_constant #si.int<8>` baked in.
  The body correctly has `hir.uint_type(mir_constant 8)` and `coerce_type`.
- **FAILS** because the specialized function's **signature region** references a value from the **body region** (the `mir_constant` that replaced the `opaque_unpack`), violating region isolation.

**Root cause:** When SpecializeFuncs replaces `opaque_unpack` results with `mir_constant` values, the signature region's `uint_type` ops (which had `uint_type %N` copied from the split_func's signature) end up referencing the body-region value.
This is a SpecializeFuncs bug — it doesn't handle the case where the split_func signature has dependent type ops that reference block args.

## Example 8: Two Const Args, Same Dependent Type

Two runtime arguments share the same dependent type from a const arg.

```silicon
fn typed_add(const N: int, a: uint<N>, b: uint<N>) -> uint<N> { a + b }
pub fn main() -> uint<8> { typed_add(8, 10, 20) }
```

**Expected:** Both `a` and `b` get type `uint<8>`.
Result is `30 : !si.uint<8>`.

**Compiler result: FAILS** — same CheckCalls dominance bug.

## Example 9: Const Fn Computing a Type Width

A `const fn` computes a width, which is then used in a type position.

```silicon
const fn pick_width(const sel: bool) -> int {
  if sel { 16 } else { 8 }
}
fn process(const W: int, x: uint<W>) -> uint<W> { x }
pub fn main(x: uint<16>) -> uint<16> { process(pick_width(true), x) }
```

**Expected:** `pick_width(true)` runs at compile time (phase -2), returns 16.
`process(16, x)` specializes.

**Compiler result: FAILS** — same CheckCalls dominance bug on `process`.

## Example 10: Chained Const Functions

```silicon
const fn double(x: int) -> int { x + x }
fn make(const N: int, x: uint<N>) -> uint<N> { x }
pub fn main() -> uint<16> { make(double(8), 100) }
```

**Expected:** `double(8)` evaluates to 16.
`make(16, 100)` specializes.

**Compiler result: FAILS** — same CheckCalls dominance bug on `make`.

## Example 11: Dyn + Dependent Type

```silicon
dyn fn deferred_id(const N: int, x: uint<N>) -> uint<N> { x }
pub dyn fn main(x: uint<8>) -> uint<8> { deferred_id(8, x) }
```

**Expected:** `deferred_id` is dyn-shifted.
Phase 0 receives N=8, phase 1 receives x: uint<8>.

**Compiler result: FAILS** — same CheckCalls dominance bug.

## Summary of Findings

### What Works Today

1. **Dependent types using `type_type` directly** (the `@identity` pattern from `dependent-type.mlir`):
   ```mlir
   hir.unified_func @identity(%T: -1, %x: 0) -> (result: 0) {
     %type_type = hir.type_type
     hir.signature (%type_type, %T) -> (%T)
   }
   ```
   This works because `%T` is used directly in the `hir.signature` terminator without intermediate type-constructor ops.

2. **All-int functions with const args** (`add-const.si`, `nested-calls.si`): These work because `int_type` takes no operands, so no block-arg references survive into cloned signature ops.

3. **Phase splitting of dependent-type functions** (when bypassing CheckCalls): `SplitPhases` correctly handles `uint_type %N` in signatures.

4. **PhaseEvalLoop partial evaluation** (when bypassing CheckCalls): The loop successfully evaluates earlier phases and specializes later phases.
   The `mir_constant` bake-in works for body ops.

### What's Broken

**Bug 1: CheckCalls dominance violation** (blocks ALL `uint<N>` dependent types from the frontend)

Any function with `fn foo(const N: int, x: uint<N>)` fails because:

1. The parser emits `hir.uint_type %N` in the signature region.
2. CheckCalls clones signature ops into the body, then `replaceUsesWithIf` on block args makes the cloned `uint_type` reference `coerce_type` results defined later.

**Fix options:**

- Move `coerce_type` insertion to before the spliced signature ops (so they dominate).
- Or skip the `replaceUsesWithIf` for cloned signature ops (they don't need coercion — they compute types, not data).

**Bug 2: SpecializeFuncs region isolation violation** (blocks dependent types through the full pipeline even at MLIR level)

When SpecializeFuncs specializes a function, the specialized clone's signature region may reference values from the body region (because `opaque_unpack` in the body was replaced with `mir_constant`, and the signature's `uint_type` still references the old block arg which was mapped to the same value).

**Fix options:**

- SpecializeFuncs should handle signature regions specially, cloning necessary `mir_constant` ops into the signature.
- Or redesign how split_func signatures interact with specialization.

### Complexity Spectrum

| Level | Description | Works? | Blocker |
|-------|-------------|--------|---------|
| 1 | `int`-only const args | Yes | — |
| 2 | Type-value passthrough (`@identity` pattern) | Yes | — |
| 3 | `uint<N>` dependent types (frontend) | No | CheckCalls bug |
| 4 | `uint<N>` dependent types (MLIR, split) | Partial | SpecializeFuncs bug |
| 5 | Computed type widths (`uint<N+N>`) | No | CheckCalls + type arithmetic |
| 6 | Multi-phase type threading | Partial | SpecializeFuncs bug |
| 7 | Type-level function calls | No | Requires `const fn` type computation |

### Recommendations

1. **Fix CheckCalls first** — this is the critical blocker for ALL dependent type examples from the frontend.
   The fix is likely a few lines: either reorder the coerce_type insertion or skip RAUW for cloned signature ops.

2. **Fix SpecializeFuncs** — needed for the full pipeline to work with dependent types even at the MLIR level.
   The signature region needs special handling during specialization.

3. **Add test cases** — once the bugs are fixed, add `test/EndToEnd/dependent-type-uint.si` with examples 1, 3, 4, 8 from this document.
