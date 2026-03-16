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

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS** (integer literal `42` now infers type `uint<N>` from context).
Fails at SplitPhases/HIRToMIR — `@id.0` contains `uint_type %N` (block arg width) which HIRToMIR can't lower (Bug 1).

At the MLIR level with pre-split IR (bypassing SplitPhases), the full pipeline works — see `test/EndToEnd/dependent-type-uint.mlir`.

## Example 2: Computed Type Width

A const argument is transformed by arithmetic before use as a type width.

```silicon
fn double_width(const N: int, x: uint<N>) -> uint<N + N> {
  x + x
}
```

**Status:** Not testable — the CheckCalls dominance bug is fixed, but `uint<N + N>` requires the parser to emit `hir.uint_type(hir.add(%N, %N))` in the signature, which is a computed dependent type.
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

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases** with `call argument requires phase -2 but value is only available at phase -1` — the literal args `3` and `5` inside `compute_width(3, 5)` are at main's body phase (-1 relative to the `use_width` const arg), but `compute_width`'s parameters are at phase -1 of `compute_width`, which is phase -2 from main's perspective.
This is a phase-depth issue: nested const calls in const-arg position require their own arguments to be available two phases early.

## Example 4: Nested Specialization with Dependent Types

An inner function's type depends on an outer function's const computation.

```silicon
fn inner(const N: int, x: uint<N>) -> uint<N> { x }
fn outer(const M: int, x: uint<M>) -> uint<M> { inner(M, x) }
pub fn main() -> uint<16> { outer(16, 100) }
```

**Expected:** Both `inner` and `outer` are specialized.
The type `uint<16>` propagates through both levels.

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases/HIRToMIR** — same as Example 1 (Bug 1: `@outer.0` contains `uint_type %M` with block arg width).

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

**Compiler result: CheckCalls and check-types PASS** (InferTypes decomposes `unify(uint<a>, uint<b>)` into `uint<unify(a, b)>` when widths don't dominate).
Fails later at SplitPhases/HIRToMIR — same as Example 1 (Bug 1: SplitPhases packs computed types instead of raw values).

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

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases** with `call argument requires phase -2 but value is only available at phase -1` — same phase-depth issue as Example 3: `select_width(true, 8)` is a nested const call whose arguments need to be two phases early.

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

**SplitPhases: WORKS.** Correctly splits into three phases.
However, `@triple.0` contains `uint_type %N` (block arg width) which prevents HIRToMIR from lowering it.
The unified form fails at `check-types` because `uint_type(coerce_type(%N))` can't be proven equal to `uint_type(%N)`.

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

**PhaseEvalLoop: WORKS** with the pre-split form.
See `test/EndToEnd/dependent-type-uint-3phase.mlir`.

- Iteration 0: `@triple.0` and `@main.0a` are successfully lowered to MIR.
  `@main.0a` is interpreted, producing `#si.opaque<[#si.opaque<[#si.int<8>]>]>`.
- Iteration 1: `SpecializeFuncs` creates `@triple.1_0`, the specialized version with `mir_constant #si.int<8>` baked in.
  The signature region correctly contains cloned `mir_constant` and `uint_type` ops (region isolation fixed).
  `@triple.1_0` is lowered to MIR (HIRToMIR now accepts `mir_constant` as uint width).
- Iteration 2: `@main.0b` is specialized and lowered. `@main.0c` is evaluated, producing `#si.int<42>`.

## Example 8: Two Const Args, Same Dependent Type

Two runtime arguments share the same dependent type from a const arg.

```silicon
fn typed_add(const N: int, a: uint<N>, b: uint<N>) -> uint<N> { a + b }
pub fn main() -> uint<8> { typed_add(8, 10, 20) }
```

**Expected:** Both `a` and `b` get type `uint<8>`.
Result is `30 : !si.uint<8>`.

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases/HIRToMIR** — same as Example 1 (Bug 1: `@typed_add.0` contains `uint_type %N` with block arg width).

At the MLIR level with pre-split IR, the full pipeline works and produces `#si.int<30>` — see `test/EndToEnd/dependent-type-uint.mlir` (Example 8 section).

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

**Compiler result: Fails at SplitPhases** with `compiler bug: op uses value from later phase` and region isolation errors.
The `const fn` body contains an `if` expression whose branches produce values across phase boundaries, which SplitPhases can't handle yet.

## Example 10: Chained Const Functions

```silicon
const fn double(x: int) -> int { x + x }
fn make(const N: int, x: uint<N>) -> uint<N> { x }
pub fn main() -> uint<16> { make(double(8), 100) }
```

**Expected:** `double(8)` evaluates to 16.
`make(16, 100)` specializes.

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases** with `call argument requires phase -2 but value is only available at phase -1` — same phase-depth issue as Examples 3 and 6: `double(8)` is a nested const call whose argument needs to be two phases early.

## Example 11: Dyn + Dependent Type

```silicon
dyn fn deferred_id(const N: int, x: uint<N>) -> uint<N> { x }
pub dyn fn main(x: uint<8>) -> uint<8> { deferred_id(8, x) }
```

**Expected:** `deferred_id` is dyn-shifted.
Phase 0 receives N=8, phase 1 receives x: uint<8>.

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases/HIRToMIR** — same as Example 1 (Bug 1: `@deferred_id.0` contains `uint_type %N` with block arg width).

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

3. **Phase splitting of dependent-type functions**: `SplitPhases` correctly handles `uint_type %N` in signatures.

4. **PhaseEvalLoop with pre-split `uint<N>` dependent types**: The full pipeline works when the IR is pre-split with the const-phase function packing the raw `%N` value (not the computed `uint_type %N`), and the runtime-phase function computing `uint_type` from the specialized constant.
   See `test/EndToEnd/dependent-type-uint.mlir` and `test/EndToEnd/dependent-type-uint-3phase.mlir`.

5. **CheckCalls with `uint<N>` signatures**: The dominance bug is fixed; `hir.uint_type %N` in signatures is correctly handled.

6. **SpecializeFuncs region isolation**: Fixed; `cloneTypeIntoSig` correctly clones dependent type ops (including `mir_constant` feeding `uint_type`) into signature regions.

7. **HIRToMIR with specialized `uint<N>`**: `shouldLower`, `isResolvableType`, and `resolveHIRType` all accept both `hir.constant_int` and `hir.mir_constant` as uint width sources.

8. **Integer literal type inference**: `constant_int` ops keep their `inferrable` type, allowing unification with context (e.g., `uint<N>` from a callee signature). All frontend `.si` examples now pass through CheckCalls, InferTypes, and CheckTypes. The remaining blocker for these examples is Bug 1 (SplitPhases packing).

### What's Broken

**Bug 1: SplitPhases packs computed types instead of raw values** (blocks `uint<N>` from unified form through the full pipeline)

When splitting `fn id(const N: int, x: uint<N>)`, `SplitPhases` creates `@id.0` with `uint_type %N` in the body (to pack type information for the next phase).
Since `%N` is a block arg (not a constant), `shouldLower` correctly rejects `@id.0` — the `uint_type` can't be materialized without knowing `N`.

The hand-crafted pre-split IR that works packs `%N` directly and defers `uint_type` computation to the next phase (where it gets specialized with a concrete value).
SplitPhases should do the same: pack the _operands_ of type constructors rather than the computed types.

**Bug 2: Nested const calls require deeper phase availability** (blocks Examples 3, 6, 10)

When a const call like `compute_width(3, 5)` is used as a const argument to another function (e.g., `use_width(compute_width(3, 5), 42)`), the arguments to the inner call need to be available at phase -2 from the caller's perspective (one level for `use_width`'s const arg, another for `compute_width`'s const args).
Currently, literal values in the function body are only available at phase -1 (one step earlier than body phase 0), so the compiler rejects them.
This needs either automatic `const { ... }` wrapping of literal arguments in nested const-call positions, or the ability for `hir.expr` to auto-lower literals that are trivially const.

**Bug 3: SplitPhases can't split `const fn` bodies with control flow** (blocks Example 9)

`const fn` bodies with `if` expressions fail during phase splitting because the branches produce values across phase boundaries.
SplitPhases doesn't yet handle block successors and region isolation for control flow in earlier-phase function bodies.

### Complexity Spectrum

| Level | Description                                  | Works? | Blocker                                    |
| ----- | -------------------------------------------- | ------ | ------------------------------------------ |
| 1     | `int`-only const args                        | Yes    | —                                          |
| 2     | Type-value passthrough (`@identity` pattern) | Yes    | —                                          |
| 3     | `uint<N>` dependent types (MLIR, pre-split)  | Yes    | —                                          |
| 4     | Multi-phase type threading (MLIR, pre-split) | Yes    | —                                          |
| 5     | `uint<N>` dependent types (MLIR, unified)    | No     | Bug 1: SplitPhases packing                 |
| 6     | `uint<N>` dependent types (frontend `.si`)   | No     | Bug 1                                      |
| 7     | Nested const calls (`f(g(x), y)`)            | No     | Bug 2: phase-depth availability            |
| 8     | `const fn` with control flow                 | No     | Bug 3: SplitPhases control flow            |
| 9     | Computed type widths (`uint<N+N>`)           | No     | Type arithmetic not implemented            |
| 10    | Type-level function calls                    | No     | Requires `const fn` type computation       |

### Recommendations

1. **Fix SplitPhases type packing (Bug 1)** — when the const-phase function needs to pack type information, pack the raw operands (e.g., `%N`) instead of computed types (e.g., `uint_type %N`), and have the next phase reconstruct the types from the specialized constants.
   This is the critical blocker for Examples 1, 4, 5, 8, 11 — all of which now pass type checking.

2. **Fix nested const-call phase depth (Bug 2)** — when a const call appears as an argument to another const parameter, its own arguments need to be available at a deeper phase.
   This blocks Examples 3, 6, 10. Possible approaches: auto-wrap trivially-const literals in `hir.expr` blocks, or allow SplitPhases to recognize that literals are phase-agnostic.

3. **Fix SplitPhases control flow in `const fn` (Bug 3)** — `const fn` bodies with `if` expressions need proper phase splitting with block successor handling.
   This blocks Example 9.

4. **Add test cases** — once Bug 1 is fixed, add `test/EndToEnd/dependent-type-uint.si` with examples 1, 4, 5, 8, 11 from this document.
