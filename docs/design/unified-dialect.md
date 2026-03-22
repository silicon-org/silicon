# Unified Dialect (`uir`)

This document describes the **Unified IR dialect** (`uir`), which captures the high-level, structured representation of Silicon programs before phase splitting and CFG flattening.
The `uir` dialect holds all ops that represent the code "as the user wrote it": unified (cross-phase) functions, structured control flow with nested regions, expression grouping for phase inference, and the witness ops that bridge the unified and split representations.

All `uir` ops are lowered away by two passes:

- **SplitPhases** consumes unified function ops (`uir.func`, `uir.call`) and phase grouping ops (`uir.expr`, `uir.pin`), producing per-phase `hir.func` and `hir.call` ops.
  It also produces `uir.split_func` witness ops that record how a unified function was decomposed.
- **FlattenCF** consumes structured control flow ops (`uir.if`, `uir.loop`, etc.) and their terminators (`uir.yield`, `uir.break`, `uir.continue`, `uir.return`, `uir.unreachable`), producing block-based `cf.br`/`cf.cond_br` and `hir.return`.

After both passes, only `uir.split_func` witnesses may remain (for external linking).
Everything else from `uir` is gone.

The ops that are *not* in `uir` — pure ops like `hir.add`, type constructors like `hir.uint_type`, type inference utilities like `hir.unify` and `hir.inferrable`, and the flat per-phase function/call/return ops — stay in the `hir` dialect.
They coexist freely with `uir` ops before lowering and survive into the HIR-to-MIR conversion.

## Pipeline Placement

The `uir` dialect fits into the Silicon pipeline as follows:

1. **Codegen** produces `uir.*` + `hir.*` ops.
   Functions are `uir.func`; calls are `uir.call`.
   Control flow is structured: `uir.if`, `uir.loop`, etc.
   Subexpressions may be wrapped in `uir.expr` for phase grouping.
2. **CheckCalls** copies function signatures into call sites.
   Works on the mixed `uir.*` + `hir.*` IR.
3. **Canonicalize / CSE** fold constants, simplify.
4. **InferTypes** resolves type unifications.
5. **CheckTypes** verifies type consistency.
6. **SplitPhases** internally runs phase analysis (DFS on structured IR), then splits:
   - `uir.func` → per-phase `hir.func` + `uir.split_func` witness.
   - `uir.call` → per-phase `hir.call` ops (decomposed using the witness).
   - `uir.expr` / `uir.pin` → dissolved (phase grouping consumed by analysis).
   - Structured CF ops survive inside the split `hir.func` bodies.
7. **FlattenCF** converts structured CF to block-based CF:
   - `uir.if` → `cf.cond_br` + merge blocks.
   - `uir.loop` → `cf.br` back-edges + exit blocks.
   - `uir.return` → `cf.br` to function return block.
   - `uir.break` / `uir.continue` → `cf.br` to loop exit / header.
   - `uir.unreachable` → erased (dead block).
8. **PhaseEvalLoop** iteratively compiles and evaluates split functions.
9. **HIRToMIR** converts the flat `hir.*` IR to `mir.*`.

## Op Overview

The ops in this dialect are organized into four groups:

### Unified Functions

Ops for representing functions that span multiple phases and their call sites.
These are consumed by SplitPhases, which decomposes them into per-phase `hir.func` and `hir.call` ops.

| Op | Summary |
|---|---|
| `uir.func` | Unified function spanning multiple phases |
| `uir.call` | Call to a unified function |
| `uir.split_func` | Witness recording how a unified function was split |

### Structured Control Flow

Ops representing control flow as nested regions, preserving the expression tree structure for phase inference.
These are consumed by FlattenCF after phase splitting.

| Op | Summary |
|---|---|
| `uir.if` | Structured if/else expression |
| `uir.loop` | Structured loop expression |
| `uir.match` | Structured match expression (future) |

### Region Terminators

Terminators for regions inside structured CF and expression ops.
Each structured region must end with exactly one of these.

| Op | Summary |
|---|---|
| `uir.yield` | Normal region completion, returning values to parent op |
| `uir.break` | Exit enclosing loop with values |
| `uir.continue` | Restart enclosing loop iteration |
| `uir.return` | Return from enclosing function (structured early exit) |
| `uir.unreachable` | Marks unreachable control flow |

### Phase Grouping

Ops for grouping expressions and pinning phases during phase inference.
These are consumed by SplitPhases (the phase analysis uses them, then they are dissolved).

| Op | Summary |
|---|---|
| `uir.expr` | Block expression with optional phase shift |
| `uir.pin` | Pin values at a phase offset |

## Op Details

### Unified Functions

#### `uir.func`

A function whose body spans multiple phases.
This is the primary function representation after codegen, before phase splitting.

The body contains ops from multiple phases, intermixed.
Phase inference (inside SplitPhases) assigns each op to a phase, then the splitting logic distributes ops into per-phase `hir.func` ops.

The function has a signature region that describes argument and result types, and a body region with the actual computation.
Argument and result phase offsets are determined by `const`/`dyn` modifiers on the function's parameters and return type.

Corresponds to the current `hir.unified_func`.

#### `uir.call`

A call to a unified function.
Carries the full set of arguments for all phases.

SplitPhases decomposes this into per-phase `hir.call` ops using the callee's `uir.split_func` witness.
Each per-phase call is placed into the caller's corresponding phase function with the appropriate argument subset and opaque context threading.

Corresponds to the current `hir.unified_call`.

#### `uir.split_func`

A witness op that records how a `uir.func` was split into per-phase functions.
It maps the unified function's name and signature to the names of the per-phase `hir.func` ops and describes how arguments and results are distributed across phases.

Its sole purpose is to allow `uir.call` ops (from the same module or from external callers linking against precompiled library IR) to be decomposed into per-phase `hir.call` ops without re-running phase inference on the callee.

After all `uir.call` ops are decomposed, the witnesses can be dropped.

Corresponds to the current `hir.split_func`.

### Structured Control Flow

#### `uir.if`

A structured if/else expression with two regions: `then` and `else`.
Each region has a single block (no block arguments) terminated by `uir.yield` or an early-exit op (`uir.return`, `uir.break`, `uir.continue`).

The condition is a `!si.bool` operand.
The `!si.bool`-to-`i1` coercion happens during FlattenCF or HIRToMIR, not at the if op.

The op produces variadic results and carries **result type operands** — SSA values representing the type of each result.
Codegen creates `hir.inferrable` ops for these types.
Inside each region, `hir.unify` ops connect the yielded value's type to the corresponding result type operand, making type flow across the region boundary explicit SSA.
See `docs/design/phase-inference.md`, "Type inference across region boundaries."

If some regions exit early (via `uir.return` etc.), only the yielding regions contribute results.
If all regions exit early, the op produces no results and the subsequent op must be `uir.unreachable`.

The `else` region may be omitted for if-without-else, in which case the op produces no results.

```mlir
%r_ty = hir.inferrable : !hir.any
%r = uir.if %cond : %r_ty {
  %x_ty = ...                        // type operand from %x's defining op
  %x = hir.add %a, %c42 : %x_ty
  %u = hir.unify %r_ty, %x_ty
  uir.yield %x : %u
} else {
  %y_ty = ...
  %u2 = hir.unify %r_ty, %y_ty
  uir.yield %y : %u2
}
```

Early return from one branch:

```mlir
%r_ty = hir.inferrable : !hir.any
%r = uir.if %cond : %r_ty {
  uir.return %c42 -> (%int_ty)
} else {
  %y_ty = ...
  %u = hir.unify %r_ty, %y_ty
  uir.yield %y : %u
}
```

Early return from all branches:

```mlir
uir.if %cond {
  uir.return %a -> (%ty)
} else {
  uir.return %b -> (%ty)
}
uir.unreachable
```

For phase inference rules, see `docs/design/phase-inference.md`, Example 10.
For FlattenCF lowering, see `docs/design/control-flow.md`, "FlattenCF: Structured CF to Block-Based CF."

#### `uir.loop`

A structured loop expression with one region: the loop body.
The body has a single block terminated by `uir.yield` (which means "continue to next iteration").
`uir.break` inside the body (possibly nested in a `uir.if`) exits the loop with values.

The op produces variadic results and carries **result type operands**, same as `uir.if`.
`uir.break` carries type operands that are unified with the loop's result types.

```mlir
%r_ty = hir.inferrable : !hir.any
%r = uir.loop : %r_ty {
  %done = ...
  uir.if %done {
    %v_ty = ...                       // type operand from %value's defining op
    %u = hir.unify %r_ty, %v_ty
    uir.break %value : %u
  }
  uir.yield
}
```

`while` and `for` loops desugar to `uir.loop`:

```mlir
// while cond { body }
uir.loop {
  %c = <cond>
  uir.if %c {
    <body>
    uir.yield
  } else {
    uir.break
  }
  uir.yield
}

// for i in 0..n { body }  (const loop, unrolled at compile time)
uir.loop {
  %done = <check i < n>
  uir.if %done {
    <body>
    <increment i>
    uir.yield
  } else {
    uir.break
  }
  uir.yield
}
```

For phase inference rules, see `docs/design/phase-inference.md`, Example 10.
For FlattenCF lowering, see `docs/design/control-flow.md`, "FlattenCF: Structured CF to Block-Based CF."

#### `uir.match`

> **Future.** Not yet implemented in the language.

A structured match expression with N regions, one per arm.
Each arm region has a single block with matched bindings as block arguments.
Terminated by `uir.yield` or an early-exit op.

The scrutinee is a single operand.
Phase inference treats it like `uir.if`; see `docs/design/phase-inference.md`.

### Region Terminators

#### `uir.yield`

Normal region completion.
Returns values from a structured CF region or expression region to the parent op.
Carries **type operands** for each value, which are unified with the parent op's result type operands during codegen (via `hir.unify`).

This is the "happy path" terminator: "this region completed normally and produced these values."

```mlir
// %u is typically the result of hir.unify between the parent's result type and %a's type
uir.yield %a, %b : %u, %v
```

For void regions (e.g., loop body continuing):

```mlir
uir.yield
```

#### `uir.break`

Exits the nearest enclosing `uir.loop`, providing the loop's result values.
Must be transitively nested inside a `uir.loop` region (possibly inside `uir.if` etc.).
Carries **type operands** unified with the enclosing loop's result type operands.

```mlir
// %u is typically the result of hir.unify between the loop's result type and %value's type
uir.break %value : %u
```

For phase inference rules (`p(enclosing_block) = p(loop)` constraint), see `docs/design/phase-inference.md`, Example 10.

#### `uir.continue`

Restarts the nearest enclosing `uir.loop` iteration.
Must be transitively nested inside a `uir.loop` region.

```mlir
uir.continue
```

Same phase constraint as `uir.break`.

#### `uir.return`

Returns from the enclosing function.
This is the structured early-exit counterpart to `hir.return`.
Unlike `hir.return` (which terminates the function body's block in flat CFG), `uir.return` can appear inside structured CF regions at any nesting depth.

Carries the same operands as `hir.return`: return values and their type operands.

```mlir
uir.return %val -> (%ty)
```

For phase inference rules (`p(enclosing_block) = p(function_body)` constraint), see `docs/design/phase-inference.md`, Example 10.
For FlattenCF lowering, see `docs/design/control-flow.md`, "FlattenCF: Structured CF to Block-Based CF."

#### `uir.unreachable`

Marks a point in the block that control flow never reaches.
Emitted by codegen after structured CF ops where all regions exit early (e.g., an `uir.if` where both branches `uir.return`).

```mlir
uir.if %cond {
  uir.return %a -> (%ty)
} else {
  uir.return %b -> (%ty)
}
uir.unreachable
```

This avoids the need for dummy return values or placeholder ops after exhaustive early exits.
For FlattenCF lowering, see `docs/design/control-flow.md`.

### Phase Grouping

#### `uir.expr`

A block expression that groups a sequence of ops into a single expression.
Has one region with a single block, terminated by `uir.yield`.

A `uir.expr` can be **floating** or **pinned**:
- **Floating** (`uir.expr { ... }`): the expression's phase comes from its consumer (demand-driven). Used when grouping is needed but the block should float.
- **Pinned** (`uir.expr pin <offset> { ... }`): the expression is fixed at `p(enclosing_block) + offset`. Used for `const { ... }` / `dyn { ... }` blocks and for statement blocks with disconnected side effects.

Non-zero offsets are always pinned (the offset is meaningless if floating).
Zero offset can be either: `uir.expr { ... }` is floating, `uir.expr pin { ... }` is pinned at `p(block)`.

Syntax:
- `uir.expr { ... }` — floating block expression (grouping only)
- `uir.expr pin { ... }` — pinned at block phase
- `uir.expr pin -1 { ... }` — `const { ... }` block (pinned at `p(block) - 1`)
- `uir.expr pin 1 { ... }` — `dyn { ... }` block (pinned at `p(block) + 1`)

Like `uir.if`, the op carries **result type operands** and the inner `uir.yield` carries type operands unified with them.

```mlir
%r_ty = hir.inferrable : !hir.any
%r = uir.expr pin -1 : %r_ty {
  %v_ty = ...                        // type operand from %v's defining op
  %v = hir.add %a, %c42 : %v_ty
  %u = hir.unify %r_ty, %v_ty
  uir.yield %v : %u
}
```

For codegen inlinability optimization and phase inference rules, see `docs/design/phase-inference.md`, "Decisions" and "Codegen changes."
Dissolved by SplitPhases after phase analysis.

#### `uir.pin`

Pins one or more values at a phase offset relative to the enclosing block.
Lightweight alternative to `uir.expr` when no region is needed (clean use-def chain, no disconnected side effects).

```mlir
%pinned = uir.pin %val, 0 : !hir.any
```

Used for:
- `let` bindings: `uir.pin %val, 0` (pin at block phase).
- `const { expr }` after inlining: `uir.pin %val, -1`.
- `dyn { expr }` after inlining: `uir.pin %val, 1`.

May have multiple operands and results (e.g., for `let` bindings that destructure tuples).

For phase inference rules, see `docs/design/phase-inference.md`, "Phase inference on this IR."
Dissolved by SplitPhases after phase analysis.

## Interaction with HIR

The `uir` and `hir` dialects coexist freely in the IR.
MLIR's dialect mixing means no conversion is needed for the shared ops.

Ops that stay in `hir`:
- **Per-phase functions:** `hir.func`, `hir.call`, `hir.return` (flat CFG terminator).
- **Pure ops:** `hir.add`, `hir.sub`, `hir.mul`, `hir.neg`, etc.
- **Constants:** `hir.constant_int`, `hir.constant_bool`, `hir.constant_unit`.
- **Type constructors:** `hir.int_type`, `hir.uint_type`, `hir.bool_type`, `hir.unit_type`, `hir.type_type`, `hir.func_type`.
- **Type inference:** `hir.unify`, `hir.inferrable`, `hir.coerce_type`, `hir.type_of`.
- **Function structure:** `hir.signature`, `hir.next_phase`.
- **Cross-phase context:** `hir.opaque_pack`, `hir.opaque_unpack`, `hir.opaque_type`.
- **Phase evaluation:** `hir.multiphase_func`, `hir.mir_constant`.
- **Bridging:** `hir.coerce_to_i1`, `hir.let`, `hir.store`.
