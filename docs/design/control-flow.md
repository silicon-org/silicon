# Control Flow

This document describes the design of control flow in Silicon: how `if`, `for`, `while`, `loop`, `break`, `continue`, and early `return` work across compile-time phases and in the final hardware phase.
It builds on the phased execution model described in {{< page-link "/design/phase-splits" >}}.

> [!WARNING]
> This document is a work in progress.
> It captures the current state of design discussions and has not been fully validated against the implementation.
> Details may change as we prototype the individual pieces.

## Overview

Control flow in Silicon spans two fundamentally different execution models:

- **Early phases** (compile-time): Standard imperative execution with a program counter.
  `if`, `for`, `while`, `loop`, `break`, `continue`, and `return` have their usual software semantics.
  Code is interpreted or JIT-compiled.
- **Final phase** (hardware): No program counter.
  Control flow must be lowered to structural hardware: muxes for conditionals, unrolled instances for loops.

Silicon's phased execution model bridges these two worlds.
The `const`/`dyn` annotations shift code to earlier or later phases relative to the current block.
Compile-time control flow drives the _generation_ of hardware; it does not become hardware itself.

### Phases are relative

The `const` and `dyn` annotations are relative phase shifts, not absolute markers:

- **`const`**: Evaluate one phase earlier than the current block. The value is a known constant when the current block executes.
- **(nothing)**: Evaluate at the current block's phase.
- **`dyn`**: Defer to one phase later than the current block. The value is symbolic/unknown when the current block executes.

Hardware conversion picks a function, runs it through all its phases, and converts the last remaining phase into hardware.
A function `fn foo(const n: int, x: int) -> int` has two phases: phase -1 for `n`, phase 0 for `x` and the result.
A function `fn foo(n: int, dyn x: int) -> dyn int` also has two phases: phase 0 for `n`, phase 1 for `x` and the result.
Both produce the same effective result — the `const`/`dyn` annotations determine relative ordering, not absolute "is hardware" vs "is not hardware."

Avoid skipping phases.
A function like `fn foo(const n: int, dyn x: int) -> dyn int` creates three phases (-1, 0, 1), but phase 0 is empty.
Prefer `fn foo(const n: int, x: int) -> int` or `fn foo(n: int, dyn x: int) -> dyn int` instead.

## Design Decisions

### Control flow inherits phase from surrounding block

The phase of an `if`, `for`, `while`, `loop`, `return`, `break`, or `continue` is determined by the surrounding code block, **not** by the phase of the condition or loop bound.
Phase shifts are only introduced by explicit `const { ... }` and `dyn { ... }` blocks.

This means:

- An `if` at the function body's phase is always control flow at that phase, regardless of whether the condition is const or dyn.
- The condition's phase determines the _mechanism_ (compile-time elimination vs hardware mux), not the phase of the body.
- There are no implicit phase shifts from control flow constructs.

This avoids a class of subtle bugs where changing a parameter annotation (e.g., `x: int` to `dyn x: int`) would silently change the execution semantics of an entire function body.

### `if const` and `for const` shorthands

The most common cross-phase pattern is "evaluate control flow one phase earlier, but keep the body at the current phase."
For example, a compile-time loop that stamps out hardware, or a compile-time condition that includes or excludes a block.

`if const` and `for const` are shorthands for this pattern:

```silicon
// if const: condition evaluated one phase earlier, body at current phase.
if const n > 3 {
    x = x + n;
}

// Equivalent to:
const {
    if n > 3 {
        dyn {
            x = x + n;
        }
    }
}
```

```silicon
// for const: loop runs one phase earlier, body stamped out at current phase.
for const i in 0..n {
    x = x + i;
}

// Equivalent to:
const {
    for i in 0..n {
        dyn {
            x = x + i;
        }
    }
}
```

Only `if` and `for` get the `const` shorthand, since these are the overwhelmingly common cross-phase patterns.
`while` and `loop` do not have a `const` shorthand — the user writes `const { while ... { dyn { ... } } }` or `const { loop { dyn { ... } } }` explicitly if needed.

A bare `if cond` requires the condition to be at the current block's phase or earlier.
A bare `for i in 0..n` requires the bound to be at the current block's phase or earlier.
Using a condition or bound from a later phase is a compile error — the user must explicitly shift phases with `dyn { ... }`.

### `if` vs `if const` with a const condition

When the condition of a bare `if` happens to be a const value (available one phase earlier), the `if` still executes at the current phase.
The condition is a known constant flowing into a regular conditional — both branches structurally exist, and canonicalization/folding may eliminate the dead branch.

In contrast, `if const` is a **structural guarantee**: the decision is made during the earlier phase's execution, and the untaken branch never appears in the IR at all.
This matters when the branches contain structurally significant operations like instantiating different hardware modules or having different numbers of ports.

```silicon
fn foo(const n: u32, x: u32) -> u32 {
    // ✓ Works: structure determined at compile time.
    // The pipeline stage is either present or absent.
    if const n > 0 {
        x = pipeline_stage(x);
    }

    // This is a current-phase if with a const condition.
    // Both branches exist structurally and the const condition
    // selects between them. Optimization may fold this away,
    // but the compiler does not guarantee it.
    if n > 0 {
        x = pipeline_stage(x);
    }
}
```

### No dynamic loops

We do not support loops whose bound is a later-phase value.
A dynamic loop is essentially unbounded in space: the hardware would need to implement a finite state machine (FSM) that iterates over clock cycles.
If a loop has a statically known bound, it can be a `for const` loop that unrolls at compile time.

All loops in Silicon execute at the current phase or earlier.
They produce a fixed, unrolled structure in later phases.
This covers the overwhelmingly common hardware use case: parameterized repetition with known bounds.

### `while` and `loop`

In addition to `for`, Silicon supports `while` and `loop` (unconditional loop, like Rust's `loop`):

```silicon
// while: conditional loop.
while !done {
    // ...
    done = check();
}

// loop: unconditional loop, exited with break.
loop {
    // ...
    if found { break; }
}
```

Both fit the replicate model naturally.
The earlier phase runs the loop (with real control flow) and collects hits.
The replicate in the next phase expands based on however many iterations actually happened.

There is no iteration limit.
Compile-time phases are real programs — if a user writes an infinite loop, the compiler hangs, just as any program with an infinite loop would.
The user can abort, add a print for debugging, and retry.
This is consistent with Silicon's philosophy that compile-time code is just code.

### Recursion constraints

Silicon supports recursive functions, but recursive calls must be at the same phase as the function body.
A recursive call inside a `const { ... }` or `dyn { ... }` block is an error — that would create unbounded phase nesting, where each recursive call produces another level of replicate expansion.

In practice, recursion is only useful in earlier (compile-time) phases, since the final phase can't represent it in hardware (no call stack).
This follows the same principle as `return`, `break`, and `continue`: cross-phase control flow is not allowed.

### `return`, `break`, and `continue` phase constraints

`return`, `break`, and `continue` are control flow at the phase of their target:

- **`return`** targets the function body — it must be at the function body's phase.
- **`break`** and **`continue`** target a loop — they must be at the loop's phase.

These constructs are **not allowed inside phase-shifted blocks** (`dyn { ... }` or `const { ... }`), because that would require cross-phase control flow:

```silicon
fn foo(n: int, dyn x: int) -> dyn int {
    // function body at phase 0

    // ✓ return at phase 0 = function body's phase
    if n > 3 { return 42; }

    // ✗ ERROR: return inside dyn block is at phase 1, not phase 0
    dyn { return x; }

    for const i in 0..n {
        // The loop runs at phase -1. The body is at phase 0.
        // ✗ ERROR: break at phase 0, but loop is at phase -1.
        if i == 3 { break; }

        // ✓ break inside const block, at phase -1 = loop's phase.
        const { if i == 3 { break; } }

        x = x + i;
    }
    return x;
}
```

This model treats `return` as "assign to implicit result slots and jump to end."
Each result slot lives at its declared phase, and the jump is control flow at the function body's phase.
Similarly, `break` is "jump to after the loop" at the loop's phase.

### Result phase floor from value selection

Control flow ops that merge multiple paths — `if`, `loop`, `expr`, and function bodies — can produce result values.
When multiple terminators (`return`, `break`, `yield`) provide **distinct values** for the same result, the CF op performs a selection: which path was taken determines which value the result gets.
That selection happens at `p(block)`, so the result cannot be at an earlier phase:

> If multiple terminators provide distinct values for a result, then `p(result) >= p(block)`.

When all terminators provide the **same SSA value**, no selection occurs — the result is the same regardless of which path was taken.
In that case, the floor does not apply and `p(result)` can be at any phase.

This is the key distinction between CF ops (which may select) and terminators like `return`/`break`/`continue` (which only transport values).
A `return` does not select — it forwards a value to the function boundary.
The selection already happened at whatever `if` or `loop` chose which `return` to reach.

```silicon
fn pick_something(x: const bool) -> (y: const int, z: int) {
    // The function body is at phase 0. Result `y` is at phase -1,
    // result `z` is at phase 0.

    // ERROR: the if executes at phase 0, but it selects between
    // distinct values (42 vs 1337) for `y` which is at phase -1.
    // The selection can't happen before the if has executed.
    if x {
        return 42, 1337;
    } else {
        return 1337, 9001;
    }
}
```

The fix is to move the selection to the right phase:

```silicon
fn pick_something(x: const bool) -> (y: const int, z: int) {
    // Move the selection of `y` to phase -1 with a const block.
    let y_val: const int = const {
        if x { 42 } else { 1337 }
    };

    // The if at phase 0 still selects `z`, which is fine
    // since p(z) == p(block).
    if x {
        return y_val, 1337;
    } else {
        return y_val, 9001;
    }
}
```

In this version, the `const { if ... }` moves the selection of `y` to phase -1 where `x` is available.
The outer `if` still selects between distinct values for `z`, but `z` is at phase 0 which equals the block phase — no constraint violation.
Both returns now provide the same SSA value (`y_val`) for `y`, so the floor does not apply to it.

For the formal rule and implementation details, see {{< page-link "/design/phase-inference" >}}, "Result phase floor for merging CF ops."

### If-else with const conditions

An `if const` with bodies in both branches uses two separate replicates (one per branch), each with a 0-or-1 element hits vector.
Threaded values flow through both replicates; the one with 0 entries passes them through unchanged.

### Dyn `if` is value selection

A bare `if` with a dyn condition (at the final phase) becomes hardware.
Both branches exist structurally and results are selected with a mux.
No `return`, `break`, or `continue` is allowed inside — the `if` is purely for value selection:

```silicon
fn abs(x: int) -> int {
    var result: int;
    if x < 0 {
        result = -x;
    } else {
        result = x;
    }
    return result;
}
```

Early returns based on dynamic conditions require restructuring as if-else:

```silicon
// Instead of early return:
//   if x > 0 { return x; }
//   return 0;

// Use if-else value selection:
var result: int;
if x > 0 {
    result = x;
} else {
    result = 0;
}
return result;
```

Alternatively, the user can pull the dyn values to the call site:

```silicon
fn select(a: bool, x: int, y: int) -> int {
    if a { return x; }
    return y;
}

fn main() {
    // The call site wraps this in dyn, making select's
    // body evaluate at the dyn phase.
    w = dyn { select(cond, x, y) };
}
```

### Side effects in dyn branches

Side effects inside dyn `if` branches are gated by enable signals.
Both branches exist in hardware, but side-effecting operations (assertions, prints, debug statements) only fire when their branch's condition is true:

```silicon
if x > 0 {
    assert(x < 100);   // gated by x > 0
    result = x;
} else {
    print(x);           // gated by !(x > 0)
    result = 0;
}
```

The compiler derives enable signals from the CFG structure.
The user never writes them explicitly.

## Const Control Flow and Phase Splitting

The interesting design challenge is how `for const` and `if const` interact with the body at the current phase.
Consider:

```silicon
fn foo(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = x + i;
        print("iteration");
    }
    return x;
}
```

Phase splitting separates this into a phase -1 function (the loop, the print) and a phase 0 template (the body).
But at split time, we don't know `n`.
Phase 0 can't contain a fixed number of copies of the body — it must be a _template_ that phase -1's execution instantiates.

### The replicate model

The core idea: phase -1 is a plain program that collects data, and phase 0 contains a `replicate` op that consumes that data to stamp out instances.

**During phase splitting**, each `for const` / `if const` body is extracted into the body region of an `hir.replicate` op in the next phase.
The const values captured by the body (loop variables, const locals) become entries in a **hits vector** that the earlier phase builds up and returns as an opaque list.

**After splitting:**

```
// Phase -1: a plain program that collects hits
fn foo_phase_neg1(const n: u32) -> opaque_list {
    hits = opaque_list_create()
    for i in 0..n {
        entry = opaque_pack(i)
        opaque_list_push(hits, entry)
        print("iteration")
    }
    return hits
}

// Phase 0: IR with a replicate op
fn foo_phase0(%hits, %x) {
    %x_final = hir.replicate %hit in %hits, (%x_iter = %x) {
        %i = hir.opaque_unpack %hit
        %x_next = hir.add %x_iter, %i
        hir.yield %x_next
    }
    return %x_final
}
```

Note that the `print("iteration")` stays in phase -1 — it runs at compile time.
Only the body of the `for const` (the `x = x + i` part) is extracted into the replicate.

**Key properties of this model:**

1. **Phase -1 is trivially JIT-able.**
   It's a plain function that does control flow and appends to vectors.
   No coupling to the MLIR context.
   No IR manipulation during execution.
   The hits vector is built using `opaque_list_create` and `opaque_list_push` ops, which allocate and mutate a heap-backed dynamic array.

2. **Phase 0 is pure IR.**
   The `replicate` op is a standard IR construct with a body region.
   Expansion happens as an IR transformation, not during interpretation.

3. **No tight coupling between JIT and IR construction.**
   Phase -1 produces data; phase 0 consumes it.
   The boundary is a serialized blob of constant values.

### The replicate op

The `hir.replicate` op consumes a hits vector and stamps out its body once per entry, threading values between iterations.
Its assembly syntax is:

```mlir
%result = hir.replicate %hit in %hits, (%x = %x_init) {
  %i = hir.opaque_unpack %hit
  %x_next = hir.add %x, %i
  hir.yield %x_next
}
```

The components:

- **`%hit in %hits`**: Iterates over the entries in the hits vector.
  `%hits` is the opaque list from the previous phase.
  `%hit` is a block argument of the body's entry block, receiving one opaque entry per iteration.
  The body uses `hir.opaque_unpack` to extract the individual const values from the entry.
- **`(%x = %x_init)`**: Declares threaded values with their initial values.
  `%x_init` flows into the first iteration.
  The names (`%x`) become block arguments of the body, available directly without a `^bb0` line.
- **`hir.yield`**: Produces the threaded values for the next iteration.
- **Result**: The replicate op's result is the yield from the last iteration.
  Code after the replicate uses this result as a normal SSA value.
  For zero iterations, the result is the initial value from the threaded args.

Since the replicate op lives in HIR, all values (hits, threaded args, results) have the implicit `!hir.any` type.
No type annotations are needed in the syntax.
The replicate is always fully expanded by `SpecializeFuncs` and never lowered to MIR.

A replicate with no threaded args (side-effect-only) omits the threaded arg list:

```mlir
hir.replicate %hit in %hits {
  hir.call @side_effect()
  hir.yield
}
```

### Replicate expansion

Expansion happens inside the existing `SpecializeFuncs` pass, which already has a worklist for transitive specialization of function calls.
Expanding a replicate is conceptually the same operation as specializing a function: take a template, substitute known constants, produce concrete instances.

The worklist processes:

- **Specialize function F with these constants** → may reveal replicates with now-known hits vectors.
- **Expand replicate R with this constant hits vector** → may reveal calls that need specialization, or nested replicates.

This avoids a separate pass and handles transitive/nested cases naturally.

**Expansion mechanics** for a hits vector of length N:

1. For each entry `k` in `0..N`:
   - Clone the body.
   - Replace the `%hit` block argument with the entry's opaque value, which allows `opaque_unpack` to fold into concrete constants.
   - Replace the threaded block arguments with the previous iteration's yield (or the initial values for `k=0`).
   - The `hir.yield` operands become the threaded values for iteration `k+1`.
2. Replace the replicate op's results with the final iteration's yield operands.
3. Delete the replicate op.

After expansion with a 4-element hits vector `[(0,), (1,), (2,), (3,)]`:

```mlir
%x1 = add %x, 0
%x2 = add %x1, 1
%x3 = add %x2, 2
%x_final = add %x3, 3
return %x_final
```

### `if const` as 0-or-1 element replicates

An `if const` is just a replicate with a 0-or-1 element hits vector:

```silicon
if const cond {
    x = x + 1;
}
```

The earlier phase returns `[()]` if `cond` is true, `[]` if false.
A replicate with 0 entries expands to nothing — threaded values pass through unchanged.
A replicate with 1 entry inlines the body once.
No separate mechanism needed.

For `if const`-`else` with bodies in both branches, each branch gets its own replicate with its own hits vector.
Exactly one will have a single entry; the other will be empty.
Threaded values flow through both, and the empty replicate passes them unchanged.

### General principle: `dyn` blocks inside control flow become replicates

The `if const` and `for const` patterns are instances of a more general principle: **every `dyn { ... }` block inside control flow at the producing phase becomes a replicate at the receiving phase.**
The producing phase's control flow determines the hits vectors; the receiving phase has replicates that expand based on those hits.

This applies uniformly across all phase boundaries, not just to the `const` sugar.
For example, a bare `if` at the function body's phase with `dyn { ... }` blocks in both branches:

```silicon
fn foo(cond: Bool, dyn a: Int, dyn b: Int) -> dyn Int {
    if cond { dyn { a } } else { dyn { b } }
}
```

The `if` is at phase 0.
The `dyn { ... }` blocks shift their contents to phase 1.
During phase splitting, each `dyn { ... }` block becomes a replicate at phase 1 with a 0-or-1 hits vector:

- Phase 0 runs the `if`. The taken branch appends `()` to its hits vector; the other branch's hits vector stays empty.
- Phase 1 has two chained replicates. The one with `[()]` inlines its body (computing `a` or `b`); the one with `[]` passes the threaded value through unchanged.

This means the body-level CFG stays entirely at phase 0.
Phase 1 has no branches — just flat replicates that the producing phase's control flow selectively activates via hits vectors.
No CFG replication into later phases is needed.

The same pattern applies at every phase boundary:

| Producing phase | Receiving phase | Example |
|---|---|---|
| Phase -2 | Phase -1 | `const { for i in 0..n { dyn { ... } } }` inside a `const` block |
| Phase -1 | Phase 0 | `if const cond { body }` (sugar for `const { if cond { dyn { body } } }`) |
| Phase 0 | Phase 1 | `if cond { dyn { a } } else { dyn { b } }` |

### Break and continue in `for const`

Break and continue affect the hits vector by controlling what the earlier phase collects.
Since `break` and `continue` must be at the loop's phase, they need to be inside `const { ... }` blocks within a `for const` body:

```silicon
for const i in 0..10 {
    const {
        if i == 5 { break; }
        if i % 2 == 0 { continue; }
    }
    x = x + i;
}
// hits = [(1,), (3,)]  — only odd iterations before the break
```

- **`break`**: The earlier phase stops the loop, so it stops appending to the hits vector.
  The replicate sees fewer entries.
- **`continue`**: The earlier phase skips the rest of the loop body for that iteration.
  If the `continue` is before the body's extraction point, that iteration isn't recorded.
  If after, it is.

No special handling in the replicate — break and continue are pure earlier-phase control flow that determines what gets collected.

### Early return in `for const`

Early `return` inside a `for const` body follows the same rule as other control flow: it must be at the function body's phase.
Since the `for const` body is at the function body's phase, `return` is directly allowed:

```silicon
fn foo(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        if i == 3 { return x; }   // ✓ phase 0, function body's phase
        x = x + i;
    }
    return x;
}
```

This works with the "assign + jump" model: `return x` assigns `x` to the result slot and jumps to the function end.
The jump is at phase 0.
The earlier phase (phase -1) sees the `return` as part of the body template — when the replicate expands and the return is reached, it exits the function.

In the replicate model, the hits vector is truncated at the return point.
Phase -1 stops collecting after the return's condition is met.
The replicate sees only the entries before the return, and the threaded value at the return point becomes the result.

For `break` and `continue`, the user wraps them in `const { ... }` to match the loop's phase:

```silicon
fn bar(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        const { if i == 5 { break; } }   // ✓ phase -1, loop's phase
        x = x + i;
    }
    return x;
}
```

### Nested control flow

Nested `for const` loops and `if const` blocks produce nested hits vectors.
The outer hits vector entries contain inner hits vectors as fields:

```silicon
fn foo(const n: u32, a: u32, b: u32) -> (u32, u32) {
    for const i in 0..n {
        a = f(a, i);                         // always runs
        if const expensive_check(i) {
            b = g(b, i);                     // conditionally runs
        }
    }
    return (a, b);
}
```

The earlier phase returns entries like `(i, inner_hits)` where `inner_hits` is itself a list:

```
hits = [
    (0, []),         // i=0, check failed
    (1, [(1,)]),     // i=1, check passed
    (2, []),         // i=2, check failed
    (3, [(3,)]),     // i=3, check passed
]
```

The current phase has nested replicates:

```mlir
%a_final, %b_final = hir.replicate %hit in %hits, (%a_iter = %a, %b_iter = %b) {
    %i, %inner_hits = hir.opaque_unpack %hit
    %a_out = hir.call @f(%a_iter, %i)
    %b_out = hir.replicate %inner_hit in %inner_hits, (%b_inner = %b_iter) {
        %j = hir.opaque_unpack %inner_hit
        %b_next = hir.call @g(%b_inner, %j)
        hir.yield %b_next
    }
    hir.yield %a_out, %b_out
}
```

The outer replicate expands first, producing 4 copies.
Each copy's inner replicate has a now-known hits vector (empty or single-element) and expands in turn.
The worklist in `SpecializeFuncs` handles this transitively.

### Multiple phases

For deeply nested phasing (phase -2 → phase -1 → phase 0), each level of `const` nesting produces one level of replicate.
The `PhaseEvalLoop` iterates: phase -2 runs and produces hits vectors for phase -1, which runs and produces hits vectors for phase 0.
Each iteration peels off one phase.

### Hits vector ops and the `opaque_list` type

The earlier phase builds the hits vector using dedicated ops that allocate and mutate a heap-backed dynamic array:

- **`opaque_list_create`**: Allocates an empty list.
  Has `MemAlloc` semantics — it creates a new heap allocation.
  Returns an `!si.opaque_list` value.
- **`opaque_list_push %list, %entry`**: Appends an `!si.opaque` entry to the list.
  Has `MemWrite` semantics — it mutates the list in-place, no result.
  This maps naturally to LLVM lowering: the list is a heap-allocated dynamic array and push appends to it.

These ops exist alongside the existing `opaque_pack`/`opaque_unpack` ops for individual entries.
The `!si.opaque_list` type is defined in the Base dialect, distinct from `!si.opaque`, so that verifiers and type converters can distinguish lists from scalar opaques.

In HIR, the replicate op's hits input is just `!hir.any` — the type distinction only matters after lowering to MIR, where the interpreter or JIT needs to handle the list correctly.

Example of the earlier phase after HIR-to-MIR lowering:

```mlir
%hits = mir.opaque_list_create
scf.for %i = 0 to %n {
  %entry = mir.opaque_pack(%i)
  mir.opaque_list_push %hits, %entry
}
return %hits
```

## FlattenCF: Structured CF to Block-Based CF

After phase splitting, each split function is single-phase.
The structured control flow ops from the `uir` dialect (`uir.if`, `uir.loop`, `uir.match`) can now be safely lowered to block-based CF (`cf.br`, `cf.cond_br`) and basic blocks.
This is done by the **FlattenCF** pass, which runs after SplitPhases and before PhaseEvalLoop.

See `docs/design/unified-dialect.md` for the full `uir` dialect design and op definitions.

### `uir.if` → conditional branches + merge block

```mlir
// Before (structured):
%r, %r_ty = uir.if %cond : %r_ty {
  %a = hir.add %x, %c42 : %x_ty
  uir.yield %a : %x_ty
} else {
  %b = hir.add %y, %c1 : %y_ty
  uir.yield %b : %y_ty
}

// After (block-based):
%cond_i1 = hir.coerce_to_i1 %cond
cf.cond_br %cond_i1, ^then, ^else
^then:
  %a = hir.add %x, %c42 : %x_ty
  cf.br ^merge(%a)
^else:
  %b = hir.add %y, %c1 : %y_ty
  cf.br ^merge(%b)
^merge(%r: !hir.any):
  ...
```

Only value results flow through the merge block — the if's `resultTypes` operands are already SSA values visible outside.

When a branch has `uir.return`, it becomes `hir.return` directly (each early return is its own return terminator).
When all branches exit early, no merge block is emitted.

### `uir.loop` → back-edge + exit block

```mlir
// Before (structured):
%r = uir.loop (%x = %init : %x_ty) : %r_ty {
  %next = ...
  %done = ...
  uir.if %done {
    uir.break %value : %v_ty
  }
  uir.continue %next : %x_ty
}

// After (block-based):
cf.br ^loop_header(%init : !hir.any)
^loop_header(%x: !hir.any):
  // (body ops, including lowered uir.if → cf.cond_br)
  %next = ...
  ...
  cf.br ^loop_header(%next : !hir.any)   // from uir.continue
^loop_exit(%r: !hir.any):                // from uir.break
  ...
```

The loop-carried iteration arguments become block arguments of the loop header.
The entry branch passes the loop's initial values into the header, and each `uir.continue` branches back to the header with the next-iteration values, exactly like the `iter_args` threading in MLIR's scf-to-cf lowering.
Only value results flow through the exit block — the loop's `resultTypes` operands are already visible outside.
`uir.break` becomes `cf.br ^loop_exit(values)`.
`uir.continue` becomes `cf.br ^loop_header(carried values)`.

### `uir.return` → `hir.return`

`uir.return` inside a structured CF region becomes `hir.return` directly.
Each early return is its own return terminator in its block — no shared return block.

### `uir.unreachable` → dead block elimination

The unreachable block is simply not emitted — it has no predecessors after the structured ops are lowered.

### `uir.expr` / `uir.pin` → dissolved

`uir.expr` ops are inlined (contents moved to parent block) since phase grouping is no longer needed after splitting.
`uir.pin` ops are removed by replacing outputs with inputs (identity removal).
FlattenCF handles both, ensuring no UIR ops survive.

## Dyn Control Flow: CFG to Dataflow

After all const control flow has been resolved (loops unrolled via replicate expansion, const ifs eliminated), the remaining control flow operates on values at the final phase.
This must be lowered from a CFG (basic blocks with conditional branches) to a flat dataflow graph of muxes suitable for CIRCT.

### Why CFG rather than structured ops

We considered keeping structured ops like `hir.if`, but a CFG representation with basic blocks and conditional branches is more general:

- Nested and irregular control flow doesn't require pattern-matching on structured op nesting.
- Standard compiler analyses (dominance, SSA) work directly.

The tradeoff is that we need a pass to recover the dataflow structure, but for acyclic CFGs (which is all we have, since there are no dynamic loops) this is well-studied.

### Converting phi nodes to muxes

MLIR represents merge points with block arguments (its version of phi nodes).
Each block argument at a merge point selects between values from different predecessors — this is exactly a mux.

**Example:**

```silicon
fn abs(x: i32) -> i32 {
    var result: i32;
    if x < 0 {
        result = -x;
    } else {
        result = x;
    }
    return result;
}
```

As CFG:

```mlir
^entry(%x: i32):
    %cond = cmp lt %x, 0
    br_cond %cond, ^then, ^else

^then:
    %neg = sub 0, %x
    br ^merge(%neg)

^else:
    br ^merge(%x)

^merge(%result: i32):
    return %result
```

After CFG-to-dataflow lowering:

```mlir
%cond = cmp lt %x, 0
%neg = sub 0, %x
%result = mux %cond, %neg, %x
return %result
```

### Path predicates

Each basic block in the acyclic CFG has a **path predicate** — the condition under which execution would have reached that block.
These are computed by walking blocks in topological order:

> For each block, its predicate is the OR of (predecessor's predicate AND the condition under which that predecessor branches to this block).

For simple if/else diamonds, this reduces to the branch condition or its negation.
For chained branches (multiple conditions, nested ifs), predicates compose:

```silicon
var result: int;
if a {
    result = x;
} else if b {
    result = y;
} else {
    result = z;
}
```

Becomes:

```mlir
%r1 = mux %b, %y, %z
%r2 = mux %a, %x, %r1
```

### Side effects and enable signals

In software, code in an untaken branch doesn't execute.
In hardware, all operations execute unconditionally and results are muxed.
This is fine for pure combinational operations, but side effects (assertions, prints, debug statements) must be gated.

Every side-effecting operation gets its block's path predicate wired to an **enable** input:

```mlir
// Before: assert in ^then block (predicate: %a & %b)
assert(%some_condition)

// After flattening:
%pred = and %a, %b
assert(%some_condition, enable: %pred)
```

### Enable threading across module instances

When a function call sits inside a conditional branch and that call becomes a hardware module instance, the instance always exists in hardware.
But side effects inside the callee should only fire when the caller's path predicate is true.

The solution is an **implicit enable signal** threaded through the hierarchy:

1. Every function/module that transitively contains side effects gets an implicit enable port, injected by the compiler.
2. When a call is made under a path predicate, the callee's enable is `AND(caller_enable, path_predicate)`.
3. Inside the callee, side-effecting ops are gated by the callee's enable, further ANDed with any local path predicates.
4. The top-level module's enable is hardwired to `true`.

**Optimization**: Functions that are purely combinational (no side effects, and don't transitively call anything with side effects) don't need an enable port.
This is computed bottom-up on the call graph.

The user never writes enable signals — the compiler derives them entirely from the CFG structure and the call graph.

## Pipeline

The full pipeline from source to hardware:

```
Parse → AST → UIR + HIR (uir.func, uir.if, uir.loop, ...)
→ CheckCalls, InferTypes, CheckTypes
→ SplitPhases (phase inference, splitting, dissolves uir.expr/uir.pin)
→ FlattenCF (uir.if → cf.cond_br, uir.loop → cf.br, ...)
→ PhaseEvalLoop:
    → HIRToMIR
    → Canonicalize, CSE
    → Interpret (runs earlier phase, produces hits vectors)
    → SpecializeFuncs (specializes functions AND expands replicates)
    → Canonicalize, CSE
    → (repeat until fixpoint)
→ All const control flow resolved, only final-phase CFG remains
→ CFG-to-dataflow (flatten ifs to muxes, compute path predicates)
→ Enable injection (gate side effects, thread enables through calls)
→ Lower to CIRCT (comb.mux, hw.module, etc.)
```

## Test Cases

The following examples exercise various aspects of the control flow design.
They are organized by category and escalate in complexity.

### Basic replicate

```silicon
// Simplest case: for const loop with no captured values, no threading.
fn emit_four() {
    for const i in 0..4 {
        side_effect();
    }
}

// Threading a single value.
fn sum_ones(x: u32) -> u32 {
    for const i in 0..4 {
        x = x + 1;
    }
    return x;
}

// Captured const loop variable.
fn sum_indices(x: u32) -> u32 {
    for const i in 0..4 {
        x = x + i;
    }
    return x;
}

// Captured const from outer scope (not the loop variable).
fn add_n_times(const n: u32, const val: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = x + val;
    }
    return x;
}

// Multiple threaded values.
fn dual_accum(a: u32, b: u32) -> (u32, u32) {
    for const i in 0..4 {
        a = a + i;
        b = b * i;
    }
    return (a, b);
}
```

### Edge cases in iteration count

```silicon
// Zero iterations — replicate is a no-op, threaded values pass through.
fn zero_iters(x: u32) -> u32 {
    for const i in 0..0 {
        x = x + 1;
    }
    return x; // must be original x
}

// Single iteration — degenerate case, body inlined once.
fn one_iter(x: u32) -> u32 {
    for const i in 0..1 {
        x = x + i;
    }
    return x;
}

// Iteration count determined by const argument.
fn n_iters(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = x + 1;
    }
    return x;
}

// Iteration count determined by const computation.
fn computed_bound(const a: u32, const b: u32, x: u32) -> u32 {
    const bound = a * b + 1;
    for const i in 0..bound {
        x = x + i;
    }
    return x;
}
```

### `if const` as 0-or-1 replicate

```silicon
// Simple conditional block.
fn maybe_inc(const cond: bool, x: u32) -> u32 {
    if const cond {
        x = x + 1;
    }
    return x;
}

// If-else with bodies in both branches.
fn branch_both(const cond: bool, x: u32) -> u32 {
    if const cond {
        x = x + 1;
    } else {
        x = x + 2;
    }
    return x;
}

// Const value captured from the if condition's scope.
fn conditional_add(const n: u32, x: u32) -> u32 {
    if const n > 3 {
        x = x + n;
    }
    return x;
}

// Nested if const.
// a=T,b=T → x+3; a=T,b=F → x+1; a=F → x.
fn nested_cond(const a: bool, const b: bool, x: u32) -> u32 {
    if const a {
        x = x + 1;
        if const b {
            x = x + 2;
        }
    }
    return x;
}
```

### Loop with conditional filtering

```silicon
// If const inside for const — some iterations produce hits, others don't.
// n=6: x + 0 + 2 + 4.
fn filter_evens(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        if const i % 2 == 0 {
            x = x + i;
        }
    }
    return x;
}

// Body both inside and outside the if const.
fn mixed_filter(const n: u32, a: u32, b: u32) -> (u32, u32) {
    for const i in 0..n {
        a = a + 1;                           // always runs
        if const i % 2 == 0 {
            b = b + i;                       // only even iterations
        }
    }
    return (a, b);
}

// Multiple conditions filtering different parts of the body.
fn multi_filter(const n: u32, x: u32, y: u32) -> (u32, u32) {
    for const i in 0..n {
        if const i % 2 == 0 {
            x = x + i;
        }
        if const i % 3 == 0 {
            y = y + i;
        }
    }
    return (x, y);
}
```

### Nested loops

```silicon
// Inner loop with const bound.
fn nested_simple(x: u32) -> u32 {
    for const i in 0..3 {
        for const j in 0..4 {
            x = x + 1;
        }
    }
    return x; // x + 12
}

// Inner bound depends on outer variable.
// i=0: nothing; i=1: +0; i=2: +0+1; i=3: +0+1+2.
// hits: [(0,[]), (1,[(0,)]), (2,[(0,),(1,)]), (3,[(0,),(1,),(2,)])].
fn triangular(x: u32) -> u32 {
    for const i in 0..4 {
        for const j in 0..i {
            x = x + j;
        }
    }
    return x;
}

// Both loop variables captured.
fn grid(x: u32) -> u32 {
    for const i in 0..3 {
        for const j in 0..3 {
            x = x + i * 3 + j;
        }
    }
    return x;
}

// Deeply nested (3 levels).
fn cube(x: u32) -> u32 {
    for const i in 0..2 {
        for const j in 0..2 {
            for const k in 0..2 {
                x = x + i * 4 + j * 2 + k;
            }
        }
    }
    return x;
}
```

### Break and continue

```silicon
// Break shortens the hits vector.
// n=10: hits = [(0,),(1,),(2,)], x + 0 + 1 + 2.
fn break_early(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        const { if i == 3 { break; } }
        x = x + i;
    }
    return x;
}

// Continue skips iterations.
// n=6: hits = [(1,),(3,),(5,)].
fn skip_evens(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        const { if i % 2 == 0 { continue; } }
        x = x + i;
    }
    return x;
}

// Break with body before the break condition.
// n=10: iterations 0,1,2 all run the body, then break at i=2.
fn partial_then_break(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = x + i;
        const { if i == 2 { break; } }
    }
    return x;
}

// Continue with body parts before and after.
// n=4: a gets all iterations, b gets all except i=2.
fn continue_split(const n: u32, a: u32, b: u32) -> (u32, u32) {
    for const i in 0..n {
        a = a + i;                              // always
        const { if i == 2 { continue; } }
        b = b + i;                              // skipped for i=2
    }
    return (a, b);
}
```

### Early return

```silicon
// Early return from inside a for const loop.
// n=10: iterations 0,1 do x+1, iteration 2 returns x+2.
fn find_first(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        if i == 2 { return x; }
        x = x + 1;
    }
    return x;
}

// Early return from inside an if const (no loop).
fn clamp(const max: u32, x: u32) -> u32 {
    if const max == 0 {
        return 0;
    }
    return x;
}

// Multiple early return points.
fn cascade(const a: bool, const b: bool, x: u32) -> u32 {
    if const a { return x + 1; }
    if const b { return x + 2; }
    return x + 3;
}
```

### `while` and `loop`

```silicon
// while loop at compile time — collects hits until condition is false.
fn collatz_steps(const n: u32, x: u32) -> u32 {
    const k = n;
    const {
        while k != 1 {
            dyn { x = x + k; }
            if k % 2 == 0 {
                k = k / 2;
            } else {
                k = k * 3 + 1;
            }
        }
    }
    return x;
}

// loop with break at compile time.
fn find_power_of_two(const start: u32, x: u32) -> u32 {
    const k = start;
    const {
        loop {
            if k % 2 == 0 { break; }
            dyn { x = x + k; }
            k = k + 1;
        }
    }
    return x;
}
```

### Dyn control flow (final-phase if)

```silicon
// Dyn if inside a for const loop — becomes muxes after expansion.
fn conditional_accum(const n: u32, x: u32, cond: bool) -> u32 {
    for const i in 0..n {
        if cond {
            x = x + i;
        } else {
            x = x;
        }
    }
    return x;
}

// Dyn if-else inside a for const loop.
fn select_per_iter(const n: u32, x: u32, sel: bool) -> u32 {
    for const i in 0..n {
        if sel {
            x = x + i;
        } else {
            x = x - i;
        }
    }
    return x;
}
```

### Function calls across phases

```silicon
// Body calls a function.
fn helper(a: u32, b: u32) -> u32 {
    return a + b;
}

fn caller(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = helper(x, i);
    }
    return x;
}

// Callee itself contains a for const loop with replication.
// Transitive specialization: inner_loop gets specialized for each value of i,
// and each specialization has its own replicate that expands.
fn inner_loop(const m: u32, x: u32) -> u32 {
    for const j in 0..m {
        x = x + j;
    }
    return x;
}

fn outer_loop(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = inner_loop(i, x);
    }
    return x;
}

// Two levels of call nesting.
fn level2(const k: u32, x: u32) -> u32 {
    for const j in 0..k {
        x = x + j;
    }
    return x;
}

fn level1(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        x = level2(i * 2, x);
    }
    return x;
}

fn top(x: u32) -> u32 {
    return level1(3, x);
    // level1 specialized with n=3
    // level2 specialized with k=0, k=2, k=4
}
```

### Multiple phases

```silicon
// Two levels of phased nesting.
// Phase -2 runs, produces hits for the outer loop.
// Phase -1 expands outer, runs inner loops, produces hits for inner.
// Phase 0 expands inner, produces flat hardware.
fn multi_phase(const const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        const m = i + 1;
        for const j in 0..m {
            x = x + i * 10 + j;
        }
    }
    return x;
}
```

### Interaction with enable threading

```silicon
// Side effects inside replicated bodies need enable signals.
fn guarded_assert(const n: u32, x: u32) -> u32 {
    for const i in 0..n {
        assert(x > 0);
        x = x + i;
    }
    return x;
}

// Side effects gated by both const conditions (filtering the hits vector)
// and dyn conditions (enable signals from CFG-to-dataflow lowering).
fn double_guarded(const n: u32, x: u32, dbg: bool) -> u32 {
    for const i in 0..n {
        if const i % 2 == 0 {
            if dbg {
                print(x);
            }
            x = x + i;
        }
    }
    return x;
}
```

### Corner cases

```silicon
// Empty for const body — replicate body is a no-op.
fn empty_body(x: u32) -> u32 {
    for const i in 0..4 {
    }
    return x;
}

// Independent instances (no threading).
fn independent(const n: u32, out: [u32; n]) -> [u32; n] {
    for const i in 0..n {
        out[i] = i * i;
    }
    return out;
}

// Loop variable not used in body — hits are [(),(),(),()].
fn ignore_index(x: u32) -> u32 {
    for const i in 0..4 {
        x = x + 1;
    }
    return x;
}

// Many captured const values.
fn many_captures(const a: u32, const b: u32, x: u32) -> u32 {
    for const i in 0..4 {
        const c = a + i;
        const d = b * i;
        const e = c ^ d;
        x = x + e;
    }
    return x;
}

// Replicate body with dyn control flow (multi-block CFG).
fn complex_body(const n: u32, x: u32, y: u32) -> (u32, u32) {
    for const i in 0..n {
        if x > y {
            x = x - 1;
        } else {
            y = y - 1;
        }
        x = x + i;
    }
    return (x, y);
}
```

## Control Flow Summary

The full set of control flow constructs in Silicon:

**Core constructs:**
- `if cond { ... } else { ... }` — conditional at current phase
- `for i in range { ... }` — bounded loop at current phase
- `while cond { ... }` — conditional loop at current phase
- `loop { ... }` — unconditional loop at current phase (exited with `break`)
- `break` — exit loop, must be at the loop's phase
- `continue` — skip to next iteration, must be at the loop's phase
- `return` — exit function, must be at the function body's phase

**Syntactic sugar:**
- `if const cond { ... }` — shorthand for `const { if cond { dyn { ... } } }`
- `for const i in range { ... }` — shorthand for `const { for i in range { dyn { ... } } }`

**Phase shift blocks:**
- `const { ... }` — evaluate one phase earlier
- `dyn { ... }` — defer to one phase later

