---
---

# Control Flow

This document describes the design of control flow in Silicon: how `if`, `for`, `while`, `break`, `continue`, and early `return` work across compile-time phases and in the final hardware phase.
It builds on the phased execution model described in {{< page-link "/design/phase-splits" >}}.

> [!WARNING]
> This document is a work in progress.
> It captures the current state of design discussions and has not been fully validated against the implementation.
> Details may change as we prototype the individual pieces.

## Overview

Control flow in Silicon spans two fundamentally different execution models:

- **Early phases** (compile-time): Standard imperative execution with a program counter.
  `if`, `for`, `while`, `break`, `continue`, and `return` have their usual software semantics.
  Code is interpreted or JIT-compiled.
- **Final phase** (hardware): No program counter.
  Control flow must be lowered to structural hardware: muxes for conditionals, unrolled instances for loops.

Silicon's phased execution model bridges these two worlds.
The `const`/`dyn` annotations determine which control flow runs at compile time and which survives into hardware.
Compile-time control flow drives the *generation* of hardware; it does not become hardware itself.

## Design Decisions

### No dynamic loops in hardware

We do not support `dyn` loops — loops whose bound is a runtime hardware signal.
A dynamic loop is essentially unbounded in space: the hardware would need to implement a finite state machine (FSM) that iterates over clock cycles.
If a loop has a statically known bound, it might as well be a `const` loop that unrolls at compile time.

All loops in Silicon are `const` loops.
They execute during an earlier phase and produce a fixed, unrolled structure in the next phase.
This covers the overwhelmingly common hardware use case: parameterized repetition with known bounds.

### If semantics follow from the phase of the condition

There is no separate `const if` vs `dyn if` syntax.
The phase of the condition expression determines the behavior:

- **Const condition**: The `if` evaluates at compile time.
  One branch is eliminated entirely, similar to `if constexpr` in C++ or `@compileIf` in Zig.
- **Dyn condition**: The `if` survives into hardware.
  Both branches execute unconditionally and the results are selected with a mux.

This falls out naturally from the phased execution model — no special-casing required.

## Const Control Flow and Phase Splitting

The interesting design challenge is how `const` control flow (loops, ifs) interacts with `dyn` blocks inside it.
Consider:

```silicon
fn foo(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn { x = x + i; }
        print("iteration");
    }
    return x;
}
```

Phase splitting separates this into a phase -1 function (the loop, the print) and a phase 0 template (the dyn block).
But at split time, we don't know `n`.
Phase 0 can't contain a fixed number of copies of the dyn block — it must be a *template* that phase -1's execution instantiates.

### The replicate model

The core idea: phase -1 is a plain program that collects data, and phase 0 contains a `replicate` op that consumes that data to stamp out hardware.

**During phase splitting**, each `dyn { ... }` block inside const control flow is extracted into the body region of an `hir.replicate` op in the next phase.
The const values captured by the dyn block (loop variables, const locals) become entries in a **hits vector** that phase -1 builds up and returns as an opaque list.

**After splitting:**

```
// Phase -1: a plain program that returns data
fn foo_phase_neg1(const n: u32) -> opaque_list {
    let mut hits = [];
    for i in 0..n {
        hits.push((i,));
        print("iteration");
    }
    return hits;
}

// Phase 0: IR with a replicate op
fn foo_phase0(%hits: !hir.opaque_list, dyn %x: u32) -> dyn u32 {
    %x_final = hir.replicate %hits
        iter_args(%x_iter = %x) -> (u32) {
        ^body(%i: u32, %x_in: u32):
            %x_next = add %x_in, %i
            hir.yield %x_next
    }
    return %x_final
}
```

**Key properties of this model:**

1. **Phase -1 is trivially JIT-able.**
   It's a plain function that does control flow and appends to vectors.
   No coupling to the MLIR context.
   No IR manipulation during execution.
   The hits vector is returned as opaque data via the existing `opaque_pack`/`opaque_unpack` machinery.

2. **Phase 0 is pure IR.**
   The `replicate` op is a standard IR construct with a body region.
   Expansion happens as an IR transformation, not during interpretation.

3. **No tight coupling between JIT and IR construction.**
   Phase -1 produces data; phase 0 consumes it.
   The boundary is a serialized blob of constant values.

### The replicate op

The `hir.replicate` op borrows its structure from `scf.for`'s `iter_args` model:

- **`%hits`**: An opaque list of entries from the previous phase.
  Each entry contains the const values captured by the dyn block for one "hit" (one loop iteration, one taken if-branch, etc.).
- **`iter_args(%x_iter = %x_init)`**: Declares threaded values with their initial values.
  `%x_init` flows into the first iteration.
- **Body parameters**: Receive the unpacked hits entry *and* the current threaded values.
  The hits entry's fields are typed according to the body's block argument types, which serve as the "schema" for deserialization.
- **`hir.yield`**: Produces the threaded values for the next iteration.
- **Result**: The replicate op's result is the yield from the last iteration.
  Code after the replicate uses this result as a normal SSA value.
  For zero iterations, the result is the initial value from `iter_args`.

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
   - Substitute the entry's const values into the const block arguments.
   - Replace the threaded block arguments with the previous iteration's yield (or the `iter_args` initial values for `k=0`).
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

### Ifs as 0-or-1 element replicates

An `if` with a dyn block in its body is just a replicate with a 0-or-1 element hits vector:

```silicon
if cond {
    dyn { x = x + 1; }
}
```

Phase -1 returns `[()]` if `cond` is true, `[]` if false.
A replicate with 0 entries expands to nothing — threaded values pass through unchanged.
A replicate with 1 entry inlines the body once.
No separate mechanism needed.

For `if-else` with dyn blocks in both branches, each branch gets its own replicate with its own hits vector.
Exactly one will have a single entry; the other will be empty.

### Break and continue

Break and continue in const loops naturally affect the hits vector:

- **`break`**: Phase -1 stops the loop, so it stops appending to the hits vector.
  The replicate sees fewer entries.
- **`continue`**: Phase -1 skips the rest of the loop body for that iteration.
  If the `continue` is before the dyn block, that iteration isn't recorded.
  If after, it is.

```silicon
for i in 0..10 {
    if i == 5 { break; }
    if i % 2 == 0 { continue; }
    dyn { x = x + i; }
}
// hits = [(1,), (3,)]  — only odd iterations before the break
```

No special handling in the replicate — break and continue are pure phase -1 control flow that determines what gets collected.

### Nested control flow

Nested loops and ifs produce nested hits vectors.
The outer hits vector entries contain inner hits vectors as fields:

```silicon
for i in 0..n {
    dyn { a = f(a, i); }           // always runs
    if expensive_check(i) {
        dyn { b = g(b, i); }       // conditionally runs
    }
}
```

Phase -1 returns entries like `(i, inner_hits)` where `inner_hits` is itself a list:

```
hits = [
    (0, []),         // i=0, check failed
    (1, [(1,)]),     // i=1, check passed
    (2, []),         // i=2, check failed
    (3, [(3,)]),     // i=3, check passed
]
```

Phase 0 has nested replicates:

```mlir
%a_final, %b_final = hir.replicate %hits
    iter_args(%a_iter = %a, %b_iter = %b) -> (u32, u32) {
    ^body(%i: u32, %inner_hits: !hir.opaque_list, %a_in: u32, %b_in: u32):
        %a_out = call @f(%a_in, %i)
        %b_out = hir.replicate %inner_hits
            iter_args(%b_inner = %b_in) -> (u32) {
            ^body(%i2: u32, %b_in2: u32):
                %b_next = call @g(%b_in2, %i2)
                hir.yield %b_next
        }
        hir.yield %a_out, %b_out
}
```

The outer replicate expands first, producing 4 copies.
Each copy's inner replicate has a now-known hits vector (empty or single-element) and expands in turn.
The worklist in `SpecializeFuncs` handles this transitively.

### Multiple phases

For deeply nested phasing (phase -2 → phase -1 → phase 0), each level of const nesting produces one level of replicate.
The `PhaseEvalLoop` iterates: phase -2 runs and produces hits vectors for phase -1, which runs and produces hits vectors for phase 0.
Each iteration peels off one phase.

### The `opaque_list` type

The hits vectors use an `!hir.opaque_list` type — a dynamically-sized, type-erased container for inter-phase data.
This is a distinct type from `!hir.opaque` (which carries single values), signaling "this is a hits vector for a replicate."

- The element structure is erased.
  The replicate body's block argument types serve as the schema for unpacking.
- Passes can identify replicate-related values by type without inspecting uses.
- The verifier can check that a replicate's input is an `opaque_list`.
- Serialization uses the existing `opaque_pack`/`opaque_unpack` machinery.

This avoids the need for complex recursive types like `!hir.const_vec<tuple<u32, !hir.const_vec<tuple<u32>>>>` in the type system.

## Dyn Control Flow: CFG to Dataflow

After all const control flow has been resolved (loops unrolled via replicate expansion, const ifs eliminated), the remaining control flow is `dyn`: conditions that are runtime hardware signals.
This must be lowered from a CFG (basic blocks with conditional branches) to a flat dataflow graph of muxes suitable for CIRCT.

### Why CFG rather than structured ops

We considered keeping structured ops like `hir.if`, but a CFG representation with basic blocks and conditional branches is more general:

- `break`, `continue`, and early `return` are just branches — no special representation needed.
- Nested and irregular control flow doesn't require pattern-matching on structured op nesting.
- Standard compiler analyses (dominance, SSA) work directly.

The tradeoff is that we need a pass to recover the dataflow structure, but for acyclic CFGs (which is all we have, since there are no dyn loops) this is well-studied.

### Converting phi nodes to muxes

MLIR represents merge points with block arguments (its version of phi nodes).
Each block argument at a merge point selects between values from different predecessors — this is exactly a mux.

**Example:**

```silicon
fn abs(dyn x: i32) -> dyn i32 {
    if x < 0 {
        return -x;
    }
    return x;
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
For chained branches (multiple early returns, nested ifs), predicates compose:

```silicon
if a { return x; }
if b { return y; }
return z;
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
Parse → AST → HIR (unified_func)
→ InferTypes, CheckCalls
→ SplitPhases (produces replicate ops, hits vector collection)
→ PhaseEvalLoop:
    → HIRToMIR
    → Canonicalize, CSE
    → Interpret (runs phase -1, produces hits vectors)
    → SpecializeFuncs (specializes functions AND expands replicates)
    → Canonicalize, CSE
    → (repeat until fixpoint)
→ All const control flow resolved, only dyn CFG remains
→ CFG-to-dataflow (flatten dyn ifs to muxes, compute path predicates)
→ Enable injection (gate side effects, thread enables through calls)
→ Lower to CIRCT (comb.mux, hw.module, etc.)
```

## Test Cases

The following examples exercise various aspects of the control flow design.
They are organized by category and escalate in complexity.

### Basic replicate

```silicon
// Simplest case: loop with no captured const, no threading.
fn emit_four() {
    for i in 0..4 {
        dyn { side_effect(); }
    }
}

// Threading a single value.
fn sum_ones(dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        dyn { x = x + 1; }
    }
    return x;
}

// Captured const loop variable.
fn sum_indices(dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        dyn { x = x + i; }
    }
    return x;
}

// Captured const from outer scope (not the loop variable).
fn add_n_times(const n: u32, const val: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn { x = x + val; }
    }
    return x;
}

// Multiple threaded values.
fn dual_accum(dyn a: u32, dyn b: u32) -> (dyn u32, dyn u32) {
    for i in 0..4 {
        dyn { a = a + i; b = b * i; }
    }
    return (a, b);
}
```

### Edge cases in iteration count

```silicon
// Zero iterations — replicate is a no-op, threaded values pass through.
fn zero_iters(dyn x: u32) -> dyn u32 {
    for i in 0..0 {
        dyn { x = x + 1; }
    }
    return x; // must be original x
}

// Single iteration — degenerate case, body inlined once.
fn one_iter(dyn x: u32) -> dyn u32 {
    for i in 0..1 {
        dyn { x = x + i; }
    }
    return x;
}

// Iteration count determined by const argument.
fn n_iters(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn { x = x + 1; }
    }
    return x;
}

// Iteration count determined by const computation.
fn computed_bound(const a: u32, const b: u32, dyn x: u32) -> dyn u32 {
    const n = a * b + 1;
    for i in 0..n {
        dyn { x = x + i; }
    }
    return x;
}
```

### If as 0-or-1 replicate

```silicon
// Simple conditional dyn block.
fn maybe_inc(const cond: bool, dyn x: u32) -> dyn u32 {
    if cond {
        dyn { x = x + 1; }
    }
    return x;
}

// If-else with dyn blocks in both branches.
fn branch_both(const cond: bool, dyn x: u32) -> dyn u32 {
    if cond {
        dyn { x = x + 1; }
    } else {
        dyn { x = x + 2; }
    }
    return x;
}

// Const value captured from the if condition's scope.
fn conditional_add(const n: u32, dyn x: u32) -> dyn u32 {
    if n > 3 {
        dyn { x = x + n; }
    }
    return x;
}

// Nested ifs.
// a=T,b=T → x+3; a=T,b=F → x+1; a=F → x.
fn nested_cond(const a: bool, const b: bool, dyn x: u32) -> dyn u32 {
    if a {
        dyn { x = x + 1; }
        if b {
            dyn { x = x + 2; }
        }
    }
    return x;
}
```

### Loop with conditional filtering

```silicon
// If inside loop — some iterations produce hits, others don't.
// n=6: x + 0 + 2 + 4.
fn filter_evens(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        if i % 2 == 0 {
            dyn { x = x + i; }
        }
    }
    return x;
}

// Dyn block both inside and outside the if.
fn mixed_filter(const n: u32, dyn a: u32, dyn b: u32) -> (dyn u32, dyn u32) {
    for i in 0..n {
        dyn { a = a + 1; }           // always runs
        if i % 2 == 0 {
            dyn { b = b + i; }       // only even iterations
        }
    }
    return (a, b);
}

// Multiple conditions filtering different dyn blocks.
fn multi_filter(const n: u32, dyn x: u32, dyn y: u32) -> (dyn u32, dyn u32) {
    for i in 0..n {
        if i % 2 == 0 {
            dyn { x = x + i; }
        }
        if i % 3 == 0 {
            dyn { y = y + i; }
        }
    }
    return (x, y);
}
```

### Nested loops

```silicon
// Inner loop with const bound.
fn nested_simple(dyn x: u32) -> dyn u32 {
    for i in 0..3 {
        for j in 0..4 {
            dyn { x = x + 1; }
        }
    }
    return x; // x + 12
}

// Inner bound depends on outer variable.
// i=0: nothing; i=1: +0; i=2: +0+1; i=3: +0+1+2.
// hits: [(0,[]), (1,[(0,)]), (2,[(0,),(1,)]), (3,[(0,),(1,),(2,)])].
fn triangular(dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        for j in 0..i {
            dyn { x = x + j; }
        }
    }
    return x;
}

// Both loop variables captured.
fn grid(dyn x: u32) -> dyn u32 {
    for i in 0..3 {
        for j in 0..3 {
            dyn { x = x + i * 3 + j; }
        }
    }
    return x;
}

// Deeply nested (3 levels).
fn cube(dyn x: u32) -> dyn u32 {
    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                dyn { x = x + i * 4 + j * 2 + k; }
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
fn break_early(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        if i == 3 { break; }
        dyn { x = x + i; }
    }
    return x;
}

// Continue skips iterations.
// n=6: hits = [(1,),(3,),(5,)].
fn skip_evens(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        if i % 2 == 0 { continue; }
        dyn { x = x + i; }
    }
    return x;
}

// Break with dyn block before the break condition.
// n=10: iterations 0,1,2 all execute the dyn block, then break.
fn partial_then_break(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn { x = x + i; }
        if i == 2 { break; }
    }
    return x;
}

// Continue with dyn blocks before and after.
// n=4: a hits all, b hits all except i=2.
fn continue_split(const n: u32, dyn a: u32, dyn b: u32) -> (dyn u32, dyn u32) {
    for i in 0..n {
        dyn { a = a + i; }              // always
        if i == 2 { continue; }
        dyn { b = b + i; }              // skipped for i=2
    }
    return (a, b);
}
```

### Early return

```silicon
// Early return from inside a loop.
// n=10: iterations 0,1 do x+1, iteration 2 returns x+2+2.
fn find_first(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        if i == 2 {
            dyn { return x + i; }
        }
        dyn { x = x + 1; }
    }
    return x;
}

// Early return from inside a const if (no loop).
fn clamp(const max: u32, dyn x: u32) -> dyn u32 {
    if max == 0 {
        dyn { return 0; }
    }
    return x;
}

// Multiple early return points.
fn cascade(const a: bool, const b: bool, dyn x: u32) -> dyn u32 {
    if a {
        dyn { return x + 1; }
    }
    if b {
        dyn { return x + 2; }
    }
    return x + 3;
}
```

### Dyn control flow inside dyn blocks

```silicon
// Dyn if inside a const loop — becomes muxes after expansion.
fn conditional_accum(const n: u32, dyn x: u32, dyn cond: bool) -> dyn u32 {
    for i in 0..n {
        dyn {
            if cond {
                x = x + i;
            }
        }
    }
    return x;
}

// Dyn if-else inside a const loop.
fn select_per_iter(const n: u32, dyn x: u32, dyn sel: bool) -> dyn u32 {
    for i in 0..n {
        dyn {
            if sel {
                x = x + i;
            } else {
                x = x - i;
            }
        }
    }
    return x;
}

// Complex dyn control flow: dyn early return becomes mux + enable.
fn dyn_cfg_in_loop(const n: u32, dyn x: u32, dyn limit: u32) -> dyn u32 {
    for i in 0..n {
        dyn {
            if x > limit {
                return x;
            }
            x = x + i;
        }
    }
    return x;
}
```

### Function calls across phases

```silicon
// Dyn block calls a dyn function.
fn helper(dyn a: u32, dyn b: u32) -> dyn u32 {
    return a + b;
}

fn caller(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn { x = helper(x, i); }
    }
    return x;
}

// Callee itself contains a const loop with replication.
// Transitive specialization: inner_loop gets specialized for each value of i,
// and each specialization has its own replicate that expands.
fn inner_loop(const m: u32, dyn x: u32) -> dyn u32 {
    for j in 0..m {
        dyn { x = x + j; }
    }
    return x;
}

fn outer_loop(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        x = inner_loop(i, x);
    }
    return x;
}

// Two levels of call nesting.
fn level2(const k: u32, dyn x: u32) -> dyn u32 {
    for j in 0..k {
        dyn { x = x + j; }
    }
    return x;
}

fn level1(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        x = level2(i * 2, x);
    }
    return x;
}

fn top(dyn x: u32) -> dyn u32 {
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
fn multi_phase(const const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        const m = i + 1;
        for j in 0..m {
            dyn { x = x + i * 10 + j; }
        }
    }
    return x;
}
```

### Interaction with enable threading

```silicon
// Side effects inside replicated dyn blocks need enable signals.
fn guarded_assert(const n: u32, dyn x: u32) -> dyn u32 {
    for i in 0..n {
        dyn {
            assert(x > 0);
            x = x + i;
        }
    }
    return x;
}

// Side effects gated by both const conditions (filtering the hits vector)
// and dyn conditions (enable signals from CFG-to-dataflow lowering).
fn double_guarded(const n: u32, dyn x: u32, dyn dbg: bool) -> dyn u32 {
    for i in 0..n {
        if i % 2 == 0 {                    // const: filters hits
            dyn {
                if dbg {                    // dyn: becomes enable
                    print(x);
                }
                x = x + i;
            }
        }
    }
    return x;
}
```

### Corner cases

```silicon
// Empty dyn block — replicate body is a no-op.
fn empty_body(dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        dyn { }
    }
    return x;
}

// Independent instances (no threading).
fn independent(const n: u32, dyn out: [u32; n]) -> dyn [u32; n] {
    for i in 0..n {
        dyn { out[i] = i * i; }
    }
    return out;
}

// Loop variable not used in dyn block — hits are [(),(),(),()].
fn ignore_index(dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        dyn { x = x + 1; }
    }
    return x;
}

// Many captured const values.
fn many_captures(const a: u32, const b: u32, dyn x: u32) -> dyn u32 {
    for i in 0..4 {
        const c = a + i;
        const d = b * i;
        const e = c ^ d;
        dyn { x = x + e; }
    }
    return x;
}

// Replicate body with complex dyn control flow (multi-block CFG).
fn complex_body(const n: u32, dyn x: u32, dyn y: u32) -> (dyn u32, dyn u32) {
    for i in 0..n {
        dyn {
            if x > y {
                x = x - 1;
            } else {
                y = y - 1;
            }
            x = x + i;
        }
    }
    return (x, y);
}
```

## Open Questions

- **Early `return` from inside const control flow.**
  How exactly does an early return from inside a const loop body interact with the replicate model?
  The return exits the function, not just the loop.
  If the return's value is dyn, the replicate body needs to communicate "I'm done, here's the result" — but the replicate expects to chain through all entries.
  One approach: the hits vector is truncated (phase -1 stops collecting after the return), and the return value flows out of the last replicate entry.
  But what if the return is conditional on a const condition that varies per iteration?

- **If-else with dyn blocks in both branches.**
  The current model uses two separate replicates (one for each branch).
  Should we instead have a single replicate that handles both branches, with the hits vector encoding which branch was taken?
  This might matter for threaded values that need to flow through whichever branch executes.

- **Independent vs threaded iterations.**
  The `iter_args` model assumes sequential threading.
  For truly independent iterations (each writes to a different array index), should there be a separate `hir.replicate_parallel` that doesn't thread state?
  Or is the threading model general enough, and the lack of data dependence is just an optimization opportunity?

- **Interaction with recursion.**
  Silicon does not currently support recursive functions, but if it ever does, a recursive function with dyn blocks would produce a dynamically-deep nesting of replicates.
  This is likely out of scope, but worth noting as a constraint.

- **Dyn early return inside a replicate body.**
  A `dyn { if cond { return x; } }` inside a const loop means the replicate body has a dyn early return.
  After replicate expansion, this becomes a dyn CFG with multiple potential exit points across the unrolled iterations.
  The CFG-to-dataflow lowering handles this (mux chain with "done" flag), but the interaction between replicate expansion and CFG construction needs careful design.

- **Representation of `while` loops.**
  `while` loops with const conditions (e.g., `while !done { ... done = check(); }`) fit the replicate model — phase -1 runs the loop and collects hits.
  But the iteration count is not syntactically bounded.
  Should we require the user to provide an upper bound for safety (to prevent infinite compile-time loops)?

- **Error reporting for non-terminating const loops.**
  A `for i in 0..n` with a very large `n` could produce enormous hardware.
  Should the compiler warn or error when a replicate expansion exceeds some threshold?
