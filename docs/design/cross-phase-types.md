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

**Compiler result: WORKS end-to-end** from both hand-crafted unified MLIR and `.si` source.
See `test/EndToEnd/dependent-type-uint-unified.mlir` and `test/EndToEnd/dependent-type-uint-frontend.si`.

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
**Fails at SplitPhases** with `call argument requires phase -2 but value is only available at phase -1` (Bug 1).

## Example 4: Nested Specialization with Dependent Types

An inner function's type depends on an outer function's const computation.

```silicon
fn inner(const N: int, x: uint<N>) -> uint<N> { x }
fn outer(const M: int, x: uint<M>) -> uint<M> { inner(M, x) }
pub fn main() -> uint<16> { outer(16, 100) }
```

**Expected:** Both `inner` and `outer` are specialized.
The type `uint<16>` propagates through both levels.

**WORKS** end-to-end.
See `test/EndToEnd/dependent-type-uint-nested.si`.

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

**Compiler result: CheckCalls and CheckTypes PASS.**
**Fails** at HIRToMIR: `hir.unify` survives with different operands — the `const { 4 + 4 }` expression creates a unification between computed and expected types that InferTypes can't resolve (Bug 3).

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
**Fails at SplitPhases** — same phase-depth issue as Example 3 (Bug 1).

## Example 7: Three-Phase Type Threading (MLIR)

Phase -2 computes a value, phase -1 uses it to construct a type, phase 0 uses the result.
Since this can't be expressed in the frontend (the frontend doesn't support `const const`), we test directly at the MLIR level.

The Silicon source this corresponds to:

```silicon
fn foo(const const N: int, const x: uint<N>, y: uint<N>) -> uint<N> {
  x + y
}
pub fn main() -> uint<8> { foo(8, 42, 100) }
```

Three phases, each doing distinct work:
- Phase -2 (`foo.0`): receives `N=8`, packs it into context.
- Phase -1 (`foo.1`): unpacks `N`, constructs `uint<N>`, coerces `x` to that type, packs both `N` and `x` into context.
- Phase 0 (`foo.2`): unpacks `N` and `x`, constructs `uint<N>` for `y`, computes `x + y`.

Key insight: in the unified form, CheckCalls inlines `foo`'s signature at the call site in `main`, creating unify ops between the literals' inferrable types and `uint_type(%N)` from the signature.
This means by the time SplitPhases runs, literals like `42` and `100` already carry the correct dependent type `uint<N>` — no coercion is needed at call boundaries.

### Unified form

```mlir
hir.unified_func @foo(%N: -2, %x: -1, %y: 0) -> (result: 0) {
  %int_type = hir.int_type
  %T = hir.uint_type %N
  hir.signature (%int_type, %T, %T) -> (%T)
} {
  %T = hir.uint_type %N
  %sum = hir.add %x, %y : %T
  hir.return %sum -> (%T)
}

hir.unified_func @main() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} {
  %int_type = hir.int_type
  %c8 = hir.constant_int 8 : %int_type
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %t3 = hir.inferrable
  %c42 = hir.constant_int 42 : %t1
  %c100 = hir.constant_int 100 : %t2
  %r = hir.unified_call @foo(%c8, %c42, %c100) : (%t0, %t1, %t2) -> (%t3) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  %tr = hir.type_of %r
  hir.return %r -> (%tr)
}
```

The unified form passes check-types (the earlier `uint_type(coerce_type(%N))` issue has been fixed).
However, the `type_of %r` / `hir.return %r -> (%tr)` pattern in the hand-written main is not what the codegen produces — `getOrCreateTypeOf` resolves `type_of(unified_call_result)` to the call's type-of-results operand directly, so no `type_of` op is created.
### Pre-split form (bypassing CheckCalls)

To test the PhaseEvalLoop, we provide pre-split IR that represents what check-calls, infer-types, and split-phases would produce.
Note that `N` must be threaded through context across all phases so that later phases can reconstruct `uint<N>`.
Callers must also derive the type from context and use it to type their literals (matching what CheckCalls + InferTypes would infer in the unified form).

See `test/EndToEnd/dependent-type-uint-3phase.mlir` for the full IR.

**PhaseEvalLoop: WORKS** with the pre-split form, producing `#si.uint<8, 142>` (= 42 + 100).

## Example 8: Two Const Args, Same Dependent Type

Two runtime arguments share the same dependent type from a const arg.

```silicon
fn typed_add(const N: int, a: uint<N>, b: uint<N>) -> uint<N> { a + b }
pub fn main() -> uint<8> { typed_add(8, 10, 20) }
```

**Expected:** Both `a` and `b` get type `uint<8>`.
Result is `30 : !si.uint<8>`.

**WORKS** end-to-end.
See `test/EndToEnd/dependent-type-uint-add.si` and `test/EndToEnd/dependent-type-uint.mlir` (Example 8 section).

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

**Compiler result: Fails at SplitPhases** with `compiler bug: op uses value from later phase` and region isolation errors (Bug 4).

## Example 10: Chained Const Functions

```silicon
const fn double(x: int) -> int { x + x }
fn make(const N: int, x: uint<N>) -> uint<N> { x }
pub fn main() -> uint<16> { make(double(8), 100) }
```

**Expected:** `double(8)` evaluates to 16.
`make(16, 100)` specializes.

**Compiler result: CheckCalls, InferTypes, and CheckTypes all PASS.**
**Fails at SplitPhases** — same phase-depth issue as Examples 3 and 6 (Bug 1).

## Example 11: Dyn + Dependent Type

```silicon
dyn fn deferred_id(const N: int, x: uint<N>) -> uint<N> { x }
pub dyn fn main(x: uint<8>) -> uint<8> { deferred_id(8, x) }
```

**Expected:** `deferred_id` is dyn-shifted.
Phase 0 receives N=8, phase 1 receives x: uint<8>.

**Compiler result: WORKS end-to-end.**
CheckCalls, InferTypes, and CheckTypes all pass; SplitPhases, HIRToMIR, and PhaseEvalLoop all succeed.
See `tmp/cross-ex11.si`.

## Summary of Findings

### What Works Today

- **Simple dependent types** (`fn id(const N: int, x: uint<N>) -> uint<N>`) work end-to-end from both hand-crafted MLIR and `.si` source.
  See `test/EndToEnd/dependent-type-uint-unified.mlir` and `test/EndToEnd/dependent-type-uint-frontend.si`.
- **PhaseEvalLoop with pre-split `uint<N>` dependent types**: See `test/EndToEnd/dependent-type-uint.mlir` and `test/EndToEnd/dependent-type-uint-3phase.mlir`.
- **Dependent types using `type_type` directly** (the `@identity` pattern): `%T` is used directly in `hir.signature` without intermediate type-constructor ops.
- **All-int functions with const args** (`add-const.si`, `nested-calls.si`): `int_type` takes no operands, so no block-arg references survive into cloned signature ops.
- **`mir.uint_type` for runtime type construction**: HIRToMIR lowers `hir.uint_type %N` with non-constant width to `mir.uint_type %N`.
  The interpreter evaluates this to `#si.type<!si.uint<N>>` once the width is specialized.
- **Dyn + dependent types** (Example 11) work end-to-end from `.si` source.

### What's Broken

**Bug 1: Nested const calls require deeper phase availability** (blocks Examples 3, 6, 10)

When a const call like `compute_width(3, 5)` is used as a const argument to another function (e.g., `use_width(compute_width(3, 5), 42)`), the arguments to the inner call need to be available at phase -2 from the caller's perspective (one level for `use_width`'s const arg, another for `compute_width`'s const args).
Currently, literal values in the function body are only available at phase -1 (one step earlier than body phase 0), so the compiler rejects them.
This needs either automatic `const { ... }` wrapping of literal arguments in nested const-call positions, or the ability for `hir.expr` to auto-lower literals that are trivially const.

**Bug 2: FIXED — SpecializeFuncs return type mismatch with nested specializations** (was blocking Examples 4, 8)

Root cause was in SplitPhases: `reconstructSignatures` fell back to standalone `opaque_type` for signature type values that depended on earlier-phase values (e.g., `uint_type(%N)` where `%N` is a const arg).
Fix: SplitPhases now creates a parallel `opaque_unpack` in the signature for phases with cross-phase context, and derives arg/result types from the body's `coerce_type` ops and return value type operands.
Examples 4 and 8 now work end-to-end.

**Bug 3: Unresolved unify with const block type widths** (blocks Example 5)

`const { 4 + 4 }` produces a type unification between the computed and expected types that InferTypes can't resolve, causing `hir.unify` to survive to HIRToMIR.

**Bug 4: SplitPhases can't split `const fn` bodies with control flow** (blocks Example 9)

`const fn` bodies with `if` expressions fail during phase splitting because the branches produce values across phase boundaries.
SplitPhases doesn't yet handle block successors and region isolation for control flow in earlier-phase function bodies.

### Complexity Spectrum

| Level | Description                                  | Works? | Blocker                                    |
| ----- | -------------------------------------------- | ------ | ------------------------------------------ |
| 1     | `int`-only const args                        | Yes    | —                                          |
| 2     | Type-value passthrough (`@identity` pattern) | Yes    | —                                          |
| 3     | `uint<N>` dependent types (MLIR, pre-split)  | Yes    | —                                          |
| 4     | Multi-phase type threading (MLIR, pre-split) | Yes    | —                                          |
| 5     | `uint<N>` dependent types (MLIR, unified)    | Yes    | —                                          |
| 6     | `uint<N>` dependent types (frontend `.si`)   | Yes    | —                                          |
| 7     | Nested specialization with dependent types   | Yes    | —                                          |
| 8     | Two args sharing dependent type + binary ops | Yes    | —                                          |
| 9     | Nested const calls (`f(g(x), y)`)            | No     | Bug 1: phase-depth availability            |
| 10    | `const fn` with control flow                 | No     | Bug 4: SplitPhases control flow            |
| 11    | Computed type widths (`uint<N+N>`)           | No     | Type arithmetic not implemented            |
| 12    | Type-level function calls                    | No     | Requires `const fn` type computation       |

### Recommendations

1. ~~**Fix SpecializeFuncs return type mismatch (Bug 2)**~~ — DONE. Fixed in SplitPhases by deriving dependent types from the body's opaque context.

2. **Fix nested const-call phase depth (Bug 1)** — when a const call appears as an argument to another const parameter, its own arguments need to be available at a deeper phase.
   This blocks Examples 3, 6, 10. Possible approaches: auto-wrap trivially-const literals in `hir.expr` blocks, or allow SplitPhases to recognize that literals are phase-agnostic.

3. **Fix SplitPhases control flow in `const fn` (Bug 4)** — `const fn` bodies with `if` expressions need proper phase splitting with block successor handling.
   This blocks Example 9.
