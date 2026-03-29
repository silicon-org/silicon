# Phase Inference

## Rules

### Phase Assignment

Almost everything is assigned the *latest* phase, pushed top-down from the parent.
The only exception is **pure ops**, which may be adjusted to an earlier phase if slack permits.

| Construct | Phase | Direction |
| --- | --- | --- |
| ConstantLike ops (literals, etc.) | `-∞` | fixed |
| Function arguments | declared phase (`const` → -1, `dyn` → +1, stacking) | fixed |
| Function results | declared phase (same modifiers) | fixed |
| Pure ops | **earliest** = `max(p(operands))` | bottom-up |
| Calls | `p(enclosing_block)`, result values shifted by offsets | fixed |
| CF expressions (`if`, `loop`, `while`, `for`, `match`) | `p(enclosing_block)` | fixed |
| Blocks (`{ ... }`) | **latest** = inherited from parent expression | top-down |
| `const { ... }` / `dyn { ... }` blocks | `p(enclosing_block) + shift` (-1 / +1) | fixed |
| Type expressions | latest = `p(annotated_value) - 1` | top-down |
| Statements (`;`-terminated) | `p(enclosing_block)` | fixed |
| `let x = expr;` | name binding; `x` has the phase of `expr`'s result | — |
| Global declarations (`static`) | pinned at module body phase; value shifted by `const`/`dyn` | fixed |
| `return` / `break` / `continue` | checked against target phase | constraint |

### Phase Details

- **ConstantLike ops** are at phase `-∞`.
  They never constrain earliest and always satisfy any latest.
  Any op whose transitive operands are all at `-∞` is also at `-∞` ("trivially materializable") and can be cloned into each phase during splitting rather than threaded as context.
  This is useful for type constructors (e.g., `uint_type %N` where `%N` is a constant).

- **Pure ops** are the only expressions that may float to an earlier phase.
  Their earliest is `max(p(operands))`, their latest is pushed from the parent.
  If there is slack (earliest < latest), the op is scheduled at the earliest phase — eager evaluation.
  This is a small optimization within the otherwise top-down algorithm.

- **Calls** are anchored at `p(enclosing_block)`, like CF expressions.
  Arg values must be available at `p(call) + arg_offset_i` (feasibility check).
  Result values are at `p(call) + result_offset_j`.
  When a call needs to execute at a different phase (e.g., as a const call argument), codegen wraps it in a floating `uir.expr` — the `uir.expr` floats to the consumer's demand, and the call inside is anchored at the expr's block phase.
  At the language level, nested calls like `foo(bar(x))` appear to float seamlessly; at the IR level, codegen wraps each call-as-subexpression in a `uir.expr` which carries the demand.

- **CF expressions** (`if`, `loop`, `while`, `for`, `match`) are anchored at `p(enclosing_block)`.
  They are control flow decisions and execute at the phase of their surrounding block.
  The condition/iterator must be available at or before `p(CF)` (feasibility check).
  Branch/body blocks are at `p(CF)`.
  When a CF expression needs to float to a different phase (e.g., as a const call argument), codegen wraps it in a `uir.expr` — the `uir.expr` is what floats, not the CF op itself.

- **Side-effecting ops** in general are anchored at `p(enclosing_block)`.
  Calls and CF are the main categories, but any op with side effects (e.g., `func.call`, `hir.let`) follows the same rule.
  Only pure ops and ConstantLike ops may float to earlier phases; everything else is anchored.
  The `uir.expr` wrapper is the universal mechanism for moving a group of side-effecting ops to a different phase.

- **Blocks** (`{ ... }`) inherit their phase from the parent expression's constraint.
  Statements within pin at `p(block)`.
  Phase constraints imposed on the block from the outside (e.g., from a call arg demand or a return type) are forwarded transparently through the yield onto the block's result expression.
  The result value flows through the yield at whatever phase it resolves to — the block phase does not constrain it.

- **`const { ... }` / `dyn { ... }` blocks** are pinned at `p(enclosing_block) + shift` (-1 for `const`, +1 for `dyn`).
  The shift is relative to the enclosing block, not to the parent expression's demand.
  The shift determines where side-effecting ops inside are anchored and where `uir.yield` logically executes.
  However, the block's **result** is at the phase of the yielded value, not the block's phase.
  Values flow through yields as phase-transparent conduits: a `const` block yielding a phase -2 value produces a -2 result, and a `dyn` block yielding a phase +2 value produces a +2 result.
  This enables nested `dyn` blocks to produce "futures" — values from later phases that can be passed around symbolically in earlier phases.
  Inside, the same rules apply.

- **Statements** pin the execution context at `p(block)`.
  The `;` means "evaluate here."

- **`let` bindings** are name bindings: `let x = expr` assigns the name `x` to the result of `expr`.
  The `let` itself has no phase effect — it does not pin or constrain the expression's phase.
  The phase of `x` is entirely determined by the RHS expression:
  - `let x = a + 42;` → x at `max(p(a), -∞)` (pure op, floats to earliest)
  - `let x = const { a + 42 };` → x at `p(block) - 1` (explicit const block)
  - `let x = dyn { b + 1 };` → x at `p(block) + 1` (explicit dyn block)
  - `let x = foo(...);` → x at `p(call) + result_offset` (call anchored at block phase)
  For pure RHS expressions, the phase of `x` may also be influenced by usage sites: if multiple consumers demand `x` at different phases, the pure op floats to the earliest that satisfies all consumers.
  At the IR level, `let` emits no wrapping ops (`uir.pin` or `uir.expr`).
  The RHS expression's ops are emitted directly into the enclosing block.
  Phase shifts require explicit `const { ... }` / `dyn { ... }` blocks on the RHS.

- **Type-constructing ops** (`hir.int_type`, `hir.uint_type %N`, `hir.type_type`, etc.) are just pure ops, like `hir.add`.
  (These are `hir` dialect ops; they coexist with the `uir` structured CF ops before flattening.)
  They compute a value that happens to be a type.
  Phase = `max(p(operands))`, resolved eagerly. No special treatment.

- **Type-consuming points** enforce the "one phase before" rule.
  The `-1` constraint is not a property of type ops — it's a property of the points where types are consumed as type annotations:
  - **`hir.signature` terminator**: pushes `latest = p(arg_i) - 1` to each arg type operand and `latest = p(result_j) - 1` to each result type operand. (The signature region is inside `uir.func`; `hir.signature` is an `hir` op that coexists with the `uir` function op.)
  - **`let` type ascription** (`let x: T = expr;`): pushes `latest = p(value) - 1` to the type expression `T`. The expression is processed first to determine `p(value)`.

  This enforces that types are known one phase before the values they describe, because the phase's program must be compiled using those types.
  Concrete types (e.g., `int`) are ConstantLike at `-∞` and trivially satisfy any constraint.
  Type-determining values must be at a strictly earlier phase than the annotated value — enforcing `const` on dependent type parameters.

- **`return` / `break` / `continue`** impose a constraint: `p(enclosing_block)` must equal the target's phase (`p(function_body)` for `return`, `p(loop)` for `break`/`continue`).
  Since all enclosing phases are already resolved top-down when these are encountered, the check happens immediately during the downward pass.
  Violation means the expression containing this CF statement can't float to the required phase.

## Computation

The algorithm is a single DFS that pushes phases top-down.
The downward pass resolves almost everything.
The upward pass only adjusts pure ops and checks feasibility.

### Function entry point

Processing a function definition follows a symmetric sequence:

1. **Fix arg and result phases** from the signature declarations. All known immediately from the `const`/`dyn` modifiers.
2. **Process signature region.** The signature region is like the body: entry block args correspond to function args with their fixed phases.
   Type-constructing ops inside (`hir.int_type`, `hir.uint_type %N`, etc.) are just pure ops — process normally.
   The `hir.signature` terminator is the type-consuming point: it pushes `latest = p(arg_i) - 1` to each arg type operand and `latest = p(result_j) - 1` to each result type operand. (The signature region is inside `uir.func`; `hir.signature` is an `hir` op.)
3. **Process body block.** `p(block) = 0`. Arg phases are available as already-resolved leaves for any references in the body.

The body block's result expression gets `latest = p(result)` (the declared result phase), **not** `latest = p(block)`.
This is a slight difference from arbitrary block expressions, where the result expression has `latest = p(block)`.
For function bodies, the block phase (0) governs statements and control flow, but the result is constrained by the declared return phase.
For example, `-> const int` (result at -1) means statements pin at 0 but the return expression must be at -1.
This matches intuition: the body runs at phase 0, and the return value must satisfy the declared result phase.

### Block traversal order

Within a block, children (statements and result expression) are processed **sequentially, top to bottom**.
Each child's full DFS (downward + upward) completes before the next child starts.
This mirrors lexical order and guarantees that `let` bindings have their phase fully resolved before any subsequent use:

1. Enter block, set `p(block)`.
2. For each statement (top to bottom): run full DFS (down + up). `let` bindings resolve their value's phase.
3. Process result expression: run full DFS. References to earlier bindings find already-resolved phases.

When a use of a `let`-bound variable is encountered (e.g., `x` in `const { x }`), it is a leaf with a known phase — just check `p(x) ≤ latest`, same as checking a function arg.
This follows SSA dominance: definitions dominate all uses.
Forward references (using a name before its `let`) are not allowed.

### Downward pass (pre-order)

Push the *latest* phase from parent to children, resolving as we go:

- **Function body**: `p(block) = 0`. Result expression gets `latest = p(declared_result)`, which may differ from `p(block)` (see above).
- **Blocks**: `p(block) = latest` from parent. Push to children.
- **`const` / `dyn` blocks**: `p(block) = p(enclosing_block) + shift`. Push to children.
- **Statements**: pin root expression at `p(block)`. Push to children.
- **`let` bindings**: pin at `p(block)`. Process expression's full DFS first to determine `p(value)`. Then push `latest = p(value) - 1` to type expression and process its DFS.
- **Calls**: `p(call) = p(enclosing_block)`. Push `p(call) + arg_offset_i` to each arg. Result values at `p(call) + result_offset_j`.
- **CF expressions**: `p(CF) = p(enclosing_block)`. Push `p(CF)` as condition latest and branch block phase.
- **`return`/`break`/`continue`**: check `p(enclosing_block) = p(target)` immediately. Error if mismatch.
- **Leaves** (variable references, literals): check `p(leaf) ≤ latest`. ConstantLike at `-∞` always passes. Function args and `let`-bound variables check their already-resolved phase ≤ latest.

### Upward pass (post-order)

Compute *earliest* and verify feasibility:

- **Pure ops**: `earliest = max(p(resolved_operands))`. If `earliest ≤ latest`, resolve to earliest (the only adjustment to the top-down assignment). Otherwise error.
- **Calls**: `earliest = max_i(p(resolved_arg_i) - arg_offset_i)`. Check `earliest ≤ p(call)` (feasibility).
- **CF expressions**: `earliest = max(condition_resolved, ...)`. Check `earliest ≤ p(CF)` (feasibility).

## Implementation Structure

### Components

- **`PhaseAnalysis` struct** (`HIR/Analysis/PhaseAnalysis.h`, `.cpp`): takes a `uir.func` op, has a `LogicalResult run()` method that runs the DFS and populates a phase map (`DenseMap<Operation*, int16_t>`).
  On phase errors, emits user-facing diagnostics directly via `mlir::emitError()` and friends (using the const/dyn vocabulary), and returns `failure()`.
  On success, returns `success()` and the phase map is ready for consumption.
  This is not a formal MLIR analysis (not cached, not invalidated) — just factored-out C++ code.

- **`SplitPhases` pass**: constructs `PhaseAnalysis` for each `uir.func`, calls `run()`.
  If failure: signals pass failure (diagnostics already emitted by the analysis).
  If success: uses the phase map to split the function into per-phase `hir.func` ops.
  Single pass that does inference + error reporting + splitting.

- **`AnnotatePhases` test pass**: constructs `PhaseAnalysis`, calls `run()`, writes the phase map back as attributes on ops (e.g., `{phase = -1}`).
  This allows lit tests to check phase assignment independently of splitting:

  ```mlir
  // RUN: silicon-opt %s --annotate-phases | FileCheck %s
  // CHECK: {phase = -1}
  %0 = hir.add %x, %c42
  ```

### How splitting consumes the phase map

The phase map (`DenseMap<Operation*, int16_t>`) tells the splitting pass which phase each op belongs to.
Splitting creates a `hir.func` for each distinct phase, moves/clones ops into the appropriate function, and wires up cross-phase context.

Key aspects (see `docs/design/phase-splits.md` for full details):
- Ops are placed into the split function for their assigned phase.
- Trivially materializable ops (phase `-∞`) are cloned into each phase that needs them.
- Cross-phase values are threaded via opaque context: if a value is defined at phase P and used at phase Q, it is carried through the context boundaries at P/P+1, P+1/P+2, ..., Q-1/Q. The context for each split function can be computed in a post-processing step.
- `uir.call` ops are decomposed into per-phase `hir.call` ops using the callee's `uir.split_func` witness. Each split call is placed into the caller's split function for the corresponding phase, not kept together — this distributes the calls across distinct per-phase programs. Each call is extended with opaque context arguments and results. Call sites don't see the individual values flowing between phases — only the opaque context structure.
- Structured CF ops (`uir.if`, `uir.loop`, etc.) survive splitting — they are moved into the appropriate phase function as a whole. The **FlattenCF** pass lowers them to `cf.br`/`cf.cond_br` after splitting.

### Pipeline placement

Phase inference and splitting run once, after type checking and call checking, before phased evaluation begins:

1. **CheckCalls** — copies function signatures into call sites. Required so that phase inference knows the arg/result offsets at each call.
2. **Canonicalize / CSE** — fold constants, simplify.
3. **InferTypes** — resolves type unifications. Required so that type expressions have concrete phases.
4. **CheckTypes** — verifies type consistency.
5. **SplitPhases** — internally runs `PhaseAnalysis` (DFS on `uir.func` ops), emits phase errors if any, then splits unified functions into per-phase `hir.func` ops using the phase map. Produces `uir.split_func` witnesses.
6. **FlattenCF** — converts `uir.if`/`uir.loop`/etc. to `cf.br`/`cf.cond_br` + `hir.return`. After this, no `uir` ops remain except `uir.split_func` witnesses.
7. **PhaseEvalLoop** — iteratively compiles and evaluates split functions. Does not re-run phase inference.

The phase map is ephemeral — computed, checked, used for splitting within the single `SplitPhases` pass, then discarded.
No pipeline restructuring is needed compared to the current structure; only the internals of step 5 change (new DFS-based analysis replacing the old three-stage approach), and step 6 is a new pass.

## IR Representation

The phase inference DFS assumes an expression tree structure: parents push latest to children, children bubble up earliest.
This requires the IR to preserve expression nesting and structured control flow until after phase splitting.
The current codegen flattens control flow to block-based CF (`cf.br`, `cf.cond_br`) immediately, which loses the structure phase inference needs.

We considered running phase inference on the AST instead of the IR.
The AST has natural expression tree structure, but lacks type information from CheckCalls/InferTypes and doesn't benefit from canonicalization/folding.
As the compiler matures, we'd also want IR-level phase inference for inlining and optimization.
Therefore, we run phase inference on the IR and ensure the IR preserves enough structure for the DFS rules to apply cleanly.

### The `uir` dialect

The structured representation lives in a dedicated **Unified IR dialect** (`uir`), separate from `hir`.
See `docs/design/unified-dialect.md` for the full dialect design.

The `uir` dialect holds:
- **Unified functions:** `uir.func`, `uir.call`, `uir.split_func` — cross-phase function representation.
- **Structured CF:** `uir.if`, `uir.loop`, `uir.match` — control flow with nested regions.
- **Region terminators:** `uir.yield`, `uir.break`, `uir.continue`, `uir.return`, `uir.unreachable`.
- **Phase grouping:** `uir.expr`, `uir.pin` — expression grouping and phase pinning.

The `hir` dialect retains the flat, per-phase ops: `hir.func`, `hir.call`, `hir.return` (block-level terminator), pure ops (`hir.add`, etc.), type constructors, type inference utilities, and cross-phase context ops.
Both dialects coexist freely in the IR — MLIR's dialect mixing requires no conversion for the shared ops.

All `uir` ops are lowered away by two passes:
- **SplitPhases:** consumes `uir.func`/`uir.call`/`uir.expr`/`uir.pin`, produces `hir.func`/`hir.call` + `uir.split_func` witnesses.
- **FlattenCF:** consumes `uir.if`/`uir.loop`/`uir.break`/`uir.continue`/`uir.return`/`uir.unreachable`, produces `cf.br`/`cf.cond_br` + `hir.return`.

### Decisions

- **Floating is the default.**
  Ops directly in a block are floating — their phase is determined by use-def chains from consumers.
  This is the common case: most ops in a block are subexpressions feeding into a result.

- **Pinning via `uir.pin` and `uir.expr`.**
  Two ops handle pinning:
  - **`uir.pin %val, <offset>`** — lightweight, no region. Pins one or more SSA values at `p(block) + offset`. May have multiple operands and results (e.g., for let bindings that destructure tuples). Used for let bindings and const/dyn expressions where the subexpression tree has a clean use-def chain.
  - **`uir.expr <offset> { ... }`** — has a region. Groups ops into a block expression. A non-zero offset pins the expression at `p(block) + offset`. Used for `const { ... }` / `dyn { ... }` blocks and for grouping disconnected side effects.

  A `uir.expr` can be **floating** or **pinned**, controlled by an optional `pin` keyword:
  - **Pinned**: block phase is fixed at `p(enclosing_block) + offset`. Side-effecting ops inside anchor at this phase.
  - **Floating**: block phase is determined by the tightest constraint from all consumers of the expr's results.
    Consumer constraints propagate through the yield to inner values.
    This works transitively through pure ops: if a yield operand is a pure op consuming a call result, the constraint propagates through the pure op to the call, which then determines the block phase.
    If a yielded value comes (directly or transitively) from a side-effecting op (call, CF), the constraint determines the block phase via the op's result offset: `p(block) = constraint - result_offset`.
    Multiple results may impose different constraints; the block floats to the **latest** phase satisfying all of them.
    If the block phase tightens as new constraints arrive, the block is re-traversed to re-anchor side-effecting ops.
    If the expr contains only pure ops, the block phase is unconstrained (nothing anchors to it).
    A floating `uir.expr` is conceptually an inline call whose result phase offsets emerge from its contents.
  Non-zero offsets are always pinned (the offset is meaningless if floating).
  Zero offset can be either: `uir.expr { ... }` is floating, `uir.expr pin { ... }` is pinned at `p(block)`.

  Syntax:
  - `uir.expr { ... }` — floating block expression (grouping only)
  - `uir.expr pin { ... }` — pinned at block phase
  - `uir.expr pin -1 { ... }` — pinned at block phase - 1 (const)
  - `uir.expr pin 1 { ... }` — pinned at block phase + 1 (dyn)
  - `uir.pin %val, 0` — pin value at block phase
  - `uir.pin %val, -1` — pin value at block phase - 1 (const)

- **Codegen wraps defensively, then immediately checks inlinability.**
  Codegen wraps call args and subexpressions in `uir.expr` whenever one *may* be needed.
  Once codegen finishes emitting into the wrapper, it checks on the spot whether the `uir.expr` can be inlined:
  - Floating `uir.expr` with a clean use-def chain (no disconnected side effects) → inline the contents into the parent block.
  - Pinned `uir.expr pin N` with a result and clean use-def chain → replace with `uir.pin %result, N` and inline contents.
  - Otherwise (disconnected side effects, no result, etc.) → keep the `uir.expr` as-is.

  This avoids leaving simplification to a separate canonicalization pass, which would need to walk the entire expr body and may not scale well.

- **Zero-use ops in a block are expression statements**, implicitly pinned.
  An op with no uses that's directly in a block is a side-effecting statement.
  Its phase is the block phase — identified by having no consumers.

- **Structured CF ops live in `uir`.**
  `uir.if`, `uir.loop`, `uir.match` replace block-based CF with region-based ops.
  `uir.return`, `uir.break`, `uir.continue` are early-exit terminators for regions.
  `uir.unreachable` marks unreachable code after exhaustive early exits.
  The DFS naturally enters/exits these regions during phase inference.
  Flat block-based CFG would require block splitting at phase boundaries, loop/region analysis to identify which blocks move together, and branch reconstruction across phases — all of which is error-prone and could easily produce illegal or unsplittable CFGs.
  Structured ops make each CF construct a self-contained unit that can be moved to a phase function as a whole.
  The **FlattenCF** pass converts structured CF to block-based CF (`cf.br`, `cf.cond_br`) after phases are assigned and split functions created.

### Phase inference on this IR

The block-level DFS works as follows:

1. **Identify roots** in the block: `uir.pin` ops, pinned `uir.expr` ops, zero-use ops (expression statements), and the terminator.
2. **Process roots in block order** (top to bottom). Each root is pinned at `p(block) + offset`. Its operands are reached via use-def.
3. **Process the terminator's operands** via use-def DFS. Push latest from the block's consumer.
4. **Floating ops** (not `uir.pin`, not pinned `uir.expr`, not zero-use) are only reached via use-def from roots or the terminator. Their phase comes from their consumer.
5. **When the DFS encounters a `uir.expr`** (floating or pinned), enter its region — it's a nested block with its own roots and result.
6. **When the DFS encounters a `uir.if` / `uir.loop` / `uir.match`**, enter each region as a nested block at `p(CF)`.

### Type inference across region boundaries

*(Decision: explicit type operands + results on region ops, unify at yield, optimistic hoisting in InferTypes.)*

#### Explicit type operands and results

`uir.if`, `uir.expr`, and `uir.loop` carry **result type operands** (SSA values representing the expected type of each result) and produce **type results** alongside value results.
`uir.yield` and `uir.break` also carry **type operands** for each value they return.
Codegen creates `hir.inferrable` ops for the result types and `hir.unify` ops inside each region to connect the yield's value types to the parent op's result types.

This makes type flow across region boundaries explicit SSA, and makes `getTypeOf` trivial during codegen and lowering — just grab the type result from the op.

Example — `uir.if` as a call arg:

```mlir
%r_ty = hir.inferrable : !hir.any
%r, %r_ty_out = uir.if %cond : %r_ty {
  %x_ty = ...                        // type operand from %x's defining op
  %x = hir.add %a, %c42 : %x_ty
  %u = hir.unify %r_ty, %x_ty        // connects inner type to outer inferrable
  uir.yield %x : %u
} else {
  %y_ty = ...                         // type operand from %y's defining op
  %u2 = hir.unify %r_ty, %y_ty
  uir.yield %y : %u2
}

// CheckCalls clones foo's signature, unifies %r_ty with foo's expected arg type
uir.call @foo(%r, %z) : (%r_ty, %z_ty) -> (%res_ty)
```

The chain: `foo`'s signature says arg 0 is `uint<8>` → CheckCalls unifies `%r_ty` with `uint<8>` → InferTypes resolves `%r_ty` → the `hir.unify %r_ty, %x_ty` inside the then-region propagates to `%x_ty` → the literal's inferrable type resolves.
Both branches unify with the same `%r_ty`, so "both branches must produce the same type" falls out for free.

The key: `%r_ty` is defined *outside* the region but MLIR's SSA scoping makes it visible *inside*.
The `hir.unify` ops inside the regions directly reference it — standard SSA, no magic.

#### Optimistic hoisting for cross-region RAUW

When InferTypes resolves a `hir.unify %outer_inferrable, %inner_type` where the concrete type is inside a region and the inferrable is outside, a direct RAUW is invalid (the inner value doesn't dominate outer uses).
InferTypes handles this by **hoisting** the inner type's transitive op tree outside the region, then performing the RAUW at the same scope.

This hoisting is always legal — even for impure ops — because of a structural guarantee from the phase system:

- `uir.expr` regions have **no block arguments**. Every value inside is either captured from the enclosing scope or computed from captured values.
- Type computations must be at a **strictly earlier phase** than the values they describe (`p(type) ≤ p(value) - 1`).
  This means every op in the type's transitive dependency tree is at an earlier phase than the region's phase.
- Ops at earlier phases can only depend on ops at **even earlier phases** — never on same-phase or later-phase ops inside the region.
  Their entire dependency chain terminates at outer-scope values and constants.

Therefore, the full op tree can be moved to the enclosing block without changing any phase assignments.
The phases of these ops are determined by type constraints and operand phases, not by the enclosing block phase — both are invariant under hoisting.

**Phase neutrality:** hoisting can neither make a valid phase error disappear nor introduce a new one.
If the program is valid, hoisting was legal by construction (the ops were at an earlier phase and would have been separated by splitting anyway).
If the program has a phase error, the same error is detected after hoisting (same constraints, same operand phases).
Phase inference runs after InferTypes and validates the program independently.

#### Summary

Two complementary mechanisms handle type flow across region boundaries:

1. **Explicit type results** on `uir.if`/`uir.expr`/`uir.loop` — make `getTypeOf` trivial for codegen and lowering. The type result is always available at the outer scope.
2. **Optimistic hoisting** in InferTypes — resolves cross-region unification when the inner type computation needs to propagate to constrain outer inferrables. Sound because the phase system guarantees the type tree is at an earlier phase.

### Pass adaptation

*(Decision: passes need recursive traversal, but no algorithm changes.)*

With type flow handled explicitly via SSA type operands and results (see above), the existing passes need only **recursive region traversal** — no algorithm changes:

- **InferTypes**: Walk into all regions to find `hir.unify` ops. The unify ops inside `uir.if`/`uir.expr` regions connect inner types to outer inferrables. When RAUW would cross a region boundary, hoist the type op tree outside first.
- **CheckCalls**: Walk into regions to find calls nested inside structured CF. Insert callee signature ops at the call's scope (inside the region). Unify call type operands with signature types as usual.
- **CheckTypes**: Walk into regions. Type checks are local to each op. Add a check that `uir.yield`/`uir.break` type operands are consistent with the parent op's result types.
- **Canonicalization**: MLIR's canonicalization framework already recurses into regions. Existing HIR patterns work unchanged. New patterns for `uir` ops (e.g., folding `uir.if` with constant condition) are added in the `uir` dialect.

The "most `uir.expr` get inlined" property limits the impact: after codegen's inlinability check, the only surviving region ops are `uir.if`, `uir.loop`, and rare `uir.expr` with disconnected side effects.

### Codegen changes

Codegen currently produces block-based CF immediately.
It needs to produce `uir` structured CF ops (`uir.if`, `uir.loop`, `uir.return`, `uir.break`, `uir.continue`, `uir.unreachable`) and wrap subexpressions in `uir.expr`.

For each `uir.if`/`uir.expr`/`uir.loop` with results, codegen:
1. Creates `hir.inferrable` ops for the result types.
2. Passes them as the op's result type operands.
3. Inside each region, creates `hir.unify` between the outer inferrable and the yielded value's type operand (grabbed directly from the defining op, no `hir.type_of` needed).
4. Uses the unify result as the `uir.yield`'s type operand.

The op produces both value results and type results.
The type results make `getTypeOf` trivial: just grab the type result from the op.

After emitting into each `uir.expr`, codegen immediately checks inlinability and simplifies to `uir.pin` or flat ops where possible.

## Error Reporting

### User-facing phase terminology

Errors never mention numeric phases.
Instead, they use the const/dyn vocabulary that matches what the user writes in code:

| Phase | User-facing term | Example phrasing |
| --- | --- | --- |
| -3 | const const const | "expression `a` must be const const const" |
| -2 | const const | "expression `a` must be const const" |
| -1 | const | "expression `a` must be const" |
| 0 | *(current phase)* | "expression `b` is dyn but must be available at the current phase" |
| +1 | dyn | "expression `x` must be dyn" |
| +2 | dyn dyn | "expression `x` must be dyn dyn" |

The terms are not put in backticks — they describe high-level language concepts, not verbatim code extracts.
Errors state the **absolute** requirement (what the value must be), not the delta.
This makes errors directly actionable: "must be const" tells the user to add `const` to the declaration.

### Error DFS

When a value's earliest phase exceeds its latest (required) phase, we report the error by propagating the required phase downward through the expression tree (DFS):

1. At each **leaf** that can't satisfy the required phase, emit an error pointing at the use site (e.g., the argument expression inside the call): "expression `b` must be const".
2. When **popping back up** to a call, emit a note on the callee name explaining why the args had that phase requirement: "result of `add` must be const, which requires argument 2 to be const".

This reports errors at the root cause (the specific values that are too late) and explains the reasoning chain on the way back up.

### Example error chain

```
error: expression `b` must be const
 --> main
  |   add(a, b)
  |          ^
note: result of `add` must be const, which requires all arguments to be const
 --> main
  |   add(a, b)
  |   ~~~
note: because it is used as a const argument to `foo`
 --> main
  |   foo(add(a, b), c)
  |   ~~~
```

No phase numbers anywhere.
For deeper cases: "expression `a` must be const const" — rare in practice, but clear.
For type errors: "type of `x` requires `N` to be const" — directly says what to fix.

## Examples

### Example 1: Plain function, no const

```silicon
pub fn add(
  x: int,   // 0
  y: int,   // 0
) -> int {   // 0
  x + y      // 0  (pure; earliest = max(0, 0) = 0, no slack)
}
```

Args and result are declared at phase 0.
The body executes at phase 0.
`x + y` is a pure op; earliest `max(0, 0) = 0`, latest 0 (return).
No slack — earliest equals latest.

### Example 2: One const arg

```silicon
pub fn add(
  const x: int,  // -1
  y: int,        // 0
) -> int {        // 0
  x + y           // 0  (pure; earliest = max(-1, 0) = 0, no slack)
}
```

`x` is declared at phase -1 (one phase earlier than the body due to `const`).
`y` and the result are at phase 0.
`x + y` is pure; earliest `max(-1, 0) = 0`, latest 0 (return).
No slack — `x` is available early but `y` pins the addition to phase 0.

### Example 3: Slack between phases

```silicon
pub fn add(
  const x: int,  // -1
  y: int,        // 0
) -> int {        // 0
  (x + 42)        // [-1, 0] → -1  (pure; slack, resolved to earliest)
    + y            // 0  (pure; earliest = max(-1, 0) = 0, no slack)
}
```

`42` is a literal at `-∞`.
`x + 42` is pure; earliest `max(-1, -∞) = -1`, latest 0 (consumed by the outer `+` at phase 0).
Slack `[-1, 0]`; resolved to -1 (eager).
This is the first example where a sub-expression can be evaluated one phase earlier than the body.
During splitting, `x + 42` would go into the phase -1 function, and `_ + y` into phase 0.

### Example 4: Calls with all-zero offsets

```silicon
fn add(x: int, y: int) -> int { x + y }

pub fn main1(a: int, b: int) -> int {              // a: 0, b: 0, return: 0
  add(a, b)                                         // 0
}

pub fn main2(const a: int, b: int) -> int {         // a: -1, b: 0, return: 0
  add(a, b)                                         // 0
}

pub fn main3(const a: int, const b: int) -> int {   // a: -1, b: -1, return: 0
  add(a, b)                                         // 0  (latest; earliest would be -1)
}
```

When the callee's arg and result offsets are all zero, a call is at `p(block)`.

`main1`: call at 0 (block phase). Arg feasibility: `max(0, 0) = 0 ≤ 0`. ✓.

`main2`: call at 0. Arg feasibility: `max(-1, 0) = 0 ≤ 0`. ✓.

`main3`: call at 0. Arg feasibility: `max(-1, -1) = -1 ≤ 0`. ✓.
Both args are const, available one phase earlier — the call could in principle execute earlier, but it's anchored at the block phase.

### Example 5: Phase errors on return

```silicon
fn add(x: int, y: int) -> int { x + y }

pub fn main4(const a: int, b: int) -> const int {   // a: -1, b: 0, return: -1
  add(a, b)                                          // 0  (block phase); result at 0, return needs -1 → ERROR
}

pub fn main5(a: int, b: int) -> const int {          // a: 0, b: 0, return: -1
  add(a, b)                                          // 0  (block phase); result at 0, return needs -1 → ERROR
}
```

Both have infeasible constraints: the call is at block phase 0, result at 0 (all-zero offsets), but the `const` return requires -1.

`main4` error — `b` is the sole bottleneck:
```
error: expression `b` must be const
 --> main4
  |   add(a, b)
  |          ^
note: result of `add` must be const, which requires argument 2 to be const
 --> main4
  |   add(a, b)
  |   ~~~
```

`main5` error — both `a` and `b` are bottlenecks:
```
error: expression `a` must be const
 --> main5
  |   add(a, b)
  |       ^
error: expression `b` must be const
 --> main5
  |   add(a, b)
  |          ^
note: result of `add` must be const, which requires arguments 1 and 2 to be const
 --> main5
  |   add(a, b)
  |   ~~~
```

`main6` error — nested call, the error chain traces through two calls:
```silicon
pub fn main6(const a: int, b: int) -> const int {   // a: -1, b: 0, return: -1
  add(
    a,              // -1
    add(b, 42),     // -1  (latest); earliest 0 → ERROR
  )                 // -1  (latest); earliest 0 → ERROR
}
```

Inner `add(b, 42)` has earliest `max(0, -∞) = 0`.
Outer `add(a, ...)` has earliest `max(-1, 0) = 0`.
Return requires -1, but earliest is 0 — infeasible.

The DFS propagates the required phase -1 downward: the outer `add` result must be at -1, so arg 2 (the inner call) must be at -1.
The inner `add` result must then be at -1, so its args must be at -1.
`b` at phase 0 is the leaf that fails; `42` at `-∞` is fine.

```
error: expression `b` must be const
 --> main6
  |     add(b, 42),
  |         ^
note: result of `add` must be const, which requires argument 1 to be const
 --> main6
  |     add(b, 42),
  |     ~~~
note: result of `add` must be const, which requires argument 2 to be const
 --> main6
  |   add(
  |   ~~~
```

### Example 6: Calls with non-zero offsets

```silicon
fn foo(
  const a: int,   // -1
  b: int,          // 0
  dyn c: int,      // +1
) -> (
  const int,       // -1
  int,             // 0
  dyn int,         // +1
) {
  (
    a + 1,          // -1  (pure; no slack)
    b + 1,          // 0   (pure; no slack)
    c + 1,          // +1  (pure; no slack)
  )
}
```

Inside `foo`, each computation lands exactly at its result's phase.
No slack.

#### Aligned caller — no slack

```silicon
pub fn main(
  const x: int,   // -1
  y: int,          // 0
  dyn z: int,      // +1
) -> (const int, int, dyn int) {   // -1, 0, +1
  foo(x, y, z);    // 0  (pinned)
  foo(x, y, z)     // 0
}
```

For the statement `foo(x, y, z);`:
Pinned at body phase 0.
Arg constraints pushed from the pin: const arg must be ≤ `0 + (-1) = -1`, regular arg ≤ `0`, dyn arg ≤ `0 + 1 = +1`.
`x` at -1, `y` at 0, `z` at +1 — all satisfied.
Results would be at -1, 0, +1 (discarded).

For the expression `foo(x, y, z)`:
latest = `min((-1)-(-1), 0-0, 1-1)` = 0. Snaps to 0.
Feasibility: earliest = `max((-1)-(-1), 0-0, 1-1)` = 0. 0 ≤ 0. ✓.

#### Early caller — slack on expression, statement pinned

```silicon
pub fn main_early(
  const const x: int,   // -2
  const y: int,          // -1
  z: int,                // 0
) -> (const int, int, dyn int) {   // -1, 0, +1
  foo(x, y, z);    // 0  (pinned)
  foo(x, y, z)     // 0  (latest; earliest would be -1)
}
```

For the statement `foo(x, y, z);`:
Pinned at body phase 0.
Arg constraints pushed from the pin, all satisfied (same as aligned caller above).

For the expression `foo(x, y, z)`:
latest = `min((-1)-(-1), 0-0, 1-1)` = 0. Snaps to 0.
Feasibility: earliest = `max((-2)-(-1), (-1)-0, 0-1)` = -1. -1 ≤ 0. ✓.

Results at `p(call) = 0`:
- const result: `0 + (-1) = -1`. Return requires -1. ✓
- regular result: `0 + 0 = 0`. Return requires 0. ✓
- dyn result: `0 + 1 = 1`. Return requires +1. ✓

Both the statement and expression end up at phase 0, but for different reasons: the statement is pinned, the expression snaps to latest.

#### Error caller — const arg pushes call too late

```silicon
pub fn main_error(
  x: int,    // 0
  y: int,    // 0
  z: int,    // 0
) -> (const int, int, dyn int) {   // -1, 0, +1
  foo(x, y, z)                     // 0  (latest); earliest 1 → ERROR
}
```

latest = `min((-1)-(-1), 0-0, 1-1)` = 0. Snaps to 0.
Feasibility: earliest = `max(0-(-1), 0-0, 0-1)` = 1. 1 > 0. **ERROR.**

The DFS pushes latest 0 to the call, then `0 + (-1) = -1` to the const arg.
`x` at phase 0 can't satisfy phase -1 — error on `x`:
```
error: expression `x` must be const
 --> main_error
  |   foo(x, y, z)
  |       ^
note: result of `foo` must be available at the current phase, which requires argument 1 to be const
 --> main_error
  |   foo(x, y, z)
  |   ~~~
```

### Example 7: Nested calls with dyn result

```silicon
fn bar(const x: int) -> int { x + 1 }
fn foo(const x: int, y: int) -> dyn int { x + y }

pub fn main(
  const const const a: int,   // -3
  const c: int,                // -1
) -> int {                     // 0
  foo(          // 0  (pinned)
    bar(a),     // -1  (latest; earliest -2, slack)
    c,
  );
  foo(          // -1
    bar(a),     // -2
    c,
  )
}
```

The `dyn` result on `foo` (offset +1) means `p(foo_result) = p(foo) + 1`.
For the result to satisfy the return at phase 0: `p(foo) + 1 ≤ 0`, so `p(foo) ≤ -1`.

**Statement** `foo(bar(a), c);` — pinned at 0:
`p(foo) = 0`. Result would be at `0 + 1 = 1` (discarded).
Arg 1 latest = `0 + (-1)` = -1. Arg 2 latest = `0 + 0` = 0.
`bar(a)`: latest = -1 (snaps to -1). Feasibility: earliest = `(-3) - (-1)` = -2. -2 ≤ -1. ✓. Slack — bar waits until needed.
`c` at -1 ≤ 0. ✓.

**Expression** `foo(bar(a), c)`:
latest(foo) = `0 - 1` = -1 (the `dyn` result pushes foo one phase earlier). Snaps to -1.
Arg 1 latest = `-1 + (-1)` = -2. `bar(a)` snaps to -2. Feasibility: earliest -2. ✓.
Arg 2 latest = -1. `c` at -1 ≤ -1. ✓.
Feasibility: foo earliest = `max((-2)-(-1), (-1)-0)` = -1. -1 ≤ -1. ✓.
Result at `-1 + 1 = 0`. Return ≤ 0. ✓.

Key contrast: the `dyn` result shifts the **expression** form of foo from 0 to -1, cascading through bar to `a`.
The **statement** pins foo at 0, so the `dyn` offset doesn't cascade — bar has slack and snaps to -1 (latest).

### Example 8: `const { ... }` and `dyn { ... }` blocks

#### `const { ... }` shifts evaluation earlier

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  const { a + 42 }    // -1  (pinned: body phase 0, shifted by -1)
    + b                // 0  (pure; earliest = max(-1, 0) = 0, no slack)
}
```

`const { a + 42 }` is pinned at `p(body) - 1 = -1`.
Inside: `a + 42` has latest -1, earliest `max(-1, -∞) = -1`. No slack. ✓.
The block's output is at -1 (pinned).
`_ + b` is pure; earliest `max(-1, 0) = 0`, latest 0. No slack. ✓.

#### `const { ... }` as a const call arg

```silicon
fn foo(const x: int, y: int) -> int { x + y }

pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  foo(const { a + 42 }, b)                    // 0
}
```

`const { a + 42 }` is pinned at -1.
foo's const arg requires ≤ `p(call) + (-1)`.
Call at `p(block) = 0`. Arg 0 needs `0 + (-1)` = -1.
`const` block at -1 ≤ -1. ✓.

Compare with a plain subexpression: `foo(a + 42, b)` — codegen wraps `a + 42` in a floating `uir.expr`, which floats to -1 (from the const arg demand), arriving at the same result.
The `const { ... }` is more explicit: it *pins* the block at -1 regardless of the consumer's demand.
The difference shows when the enclosing context and the call's demand disagree (see below).

#### `const { ... }` vs `{ ... }` — where they diverge

```silicon
fn bar(const const x: int, y: int) -> int { x + y }

pub fn main(const const a: int, b: int) -> int {   // a: -2, b: 0, return: 0
  bar(const { a + 42 }, b)    // const block at -1, but bar needs -2. ERROR!
  bar({ a + 42 }, b)          // regular block at -2 (from bar's demand). OK.
}
```

`const { a + 42 }` is pinned at `0 - 1 = -1`.
bar's const-const arg requires ≤ `p(call) + (-2)` = -2.
But -1 > -2. **ERROR.**

The regular block `{ a + 42 }` inherits `p(block) = -2` from bar's demand.
Inside: `a + 42` has latest -2, earliest `max(-2, -∞) = -2`. ✓.
The regular block adapts; the const block is fixed.

```
error: const block produces a const value, but `bar` requires argument 1 to be const const
 --> main
  |   bar(const { a + 42 }, b)
  |       ~~~~~
note: result of `bar` must be available at the current phase, which requires argument 1 to be const const
 --> main
  |   bar(const { a + 42 }, b)
  |   ~~~
```

#### Error inside a `const { ... }` block

```silicon
pub fn main(a: int, b: int) -> int {   // a: 0, b: 0, return: 0
  const { a + 42 }    // -1  (pinned)
    + b                // 0  (pure; no slack)
}
```

`const { a + 42 }` at -1.
Inside: `a + 42` latest -1, earliest `max(0, -∞) = 0`.
Earliest 0 > latest -1. **ERROR.**
`a` is at phase 0 but the const block requires phase -1.

```
error: expression `a` must be const
 --> main
  |   const { a + 42 }
  |           ^
```

#### `dyn { ... }` shifts evaluation later

```silicon
pub fn main(a: int, dyn b: int) -> dyn int {   // a: 0, b: +1, return: +1
  a + dyn { b + 1 }
}
```

`dyn { b + 1 }` is pinned at `p(body) + 1 = +1`.
Inside: `b + 1` has latest +1, earliest `max(+1, -∞) = +1`. No slack. ✓.
Block result at +1.
`a + _` is pure; earliest `max(0, +1) = +1`, latest +1 (return). No slack. ✓.

#### `dyn { ... }` with eager internal computation

```silicon
pub fn main(a: int, dyn b: int) -> dyn int {   // a: 0, b: +1, return: +1
  dyn { a + 1 }
}
```

`dyn { a + 1 }` at +1.
Inside: `a + 1` has latest +1, earliest `max(0, -∞) = 0`.
`[0, +1] → 0` (pure; slack, resolved to earliest).
The computation happens at phase 0, but the block's result is pinned at +1.
During splitting, `a + 1` is computed in the phase-0 function, but the value is presented at phase +1 — like a pipeline register.

#### `const { ... }` as a statement

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  const { side_effect(a) };    // -1  (const block pinned, `;` discards result)
  b
}
```

The `const { ... }` block is pinned at -1 regardless of being a statement.
The `;` discards the result but doesn't change the execution phase.
Inside: `side_effect(a)` has latest -1. `a` at -1 ≤ -1. ✓.
The side effect executes at phase -1.

#### ConstantLike ops inside `const { ... }`

```silicon
pub fn main() -> int {   // return: 0
  const { 42 } + 1
}
```

`const { 42 }` at -1.
Inside: `42` is ConstantLike at `-∞`. Latest -1. `-∞ ≤ -1`. ✓.
Block result at -1.
`_ + 1` is pure; earliest `max(-1, -∞) = -1`, latest 0. `[-1, 0] → -1` (slack, resolved to earliest).

The literal `42` is at `-∞` and will be cloned into whatever phase needs it during splitting.
The `const { ... }` block pins the result at -1, giving it a concrete phase — the user explicitly asked for a const value.

### Example 9: `let` bindings

#### Simple `let` — pinned at block phase

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  let x = a + 42;     // 0  (pinned: let is a statement). x at 0.
  x + b                // 0  (pure; no slack)
}
```

`let x = a + 42;` pins the execution at phase 0.
`a + 42` is a pure op; its earliest would be -1 (from `a`), but the statement pins it at 0.
`x` is at phase 0.
`x + b` has earliest `max(0, 0) = 0`, latest 0. No slack.

Compare with the inlined version (Example 3): `(a + 42) + b` lets `a + 42` float to -1.
Naming the value with `let` pins it.
This is intentional — `let` means "evaluate here."

#### `let` with `const { ... }` — named value at earlier phase

```silicon
fn foo(const x: int, y: int) -> int { x + y }

pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  let x = const { a + 42 };    // 0  (pinned). const shifts → x at -1.
  foo(x, b)                     // 0
}
```

The `let` pins the execution context at 0.
The `const { ... }` shifts the result to `0 - 1 = -1`, so `x` is at phase -1.
Inside the const block: `a + 42` has latest -1, earliest `max(-1, -∞) = -1`. ✓.
`foo(x, b)`: foo's const arg needs ≤ `0 + (-1)` = -1. `x` at -1 ≤ -1. ✓.

Without `const { ... }`, `x` would be at 0, and foo's const arg would fail:

```silicon
  let x = a + 42;    // x at 0.
  foo(x, b)           // foo needs const arg at -1. x at 0. ERROR!
```

```
error: expression `x` must be const
 --> main
  |   foo(x, b)
  |       ^
```

#### `let` with call — side effects stay pinned

```silicon
fn print_and_return(x: int) -> int { /* prints x, returns x */ }

pub fn main(const a: int) -> int {   // a: -1, return: 0
  let z = print_and_return(a);    // 0  (pinned). z at 0.
  const { z }                      // -1  (pinned). Needs z at -1. ERROR!
}
```

The `let` pins `print_and_return(a)` at phase 0 — the print happens at phase 0, as the user expects from reading the code top-to-bottom.
`z` is at phase 0.
`const { z }` is at -1 and needs `z` at -1. But `z` is at 0. **ERROR.**

```
error: expression `z` must be const
 --> main
  |   const { z }
  |           ^
```

To explicitly run the print at phase -1, the user writes:

```silicon
  let z = const { print_and_return(a) };    // context at 0, const shifts → z at -1.
  const { z }                                // z at -1 ≤ -1. ✓.
```

Now the intent is clear: the `const { ... }` around the call explicitly says "execute this at phase -1."

#### `let` with `dyn` result — value at later phase

```silicon
fn deferred() -> dyn int { ... }

pub fn main(a: int) -> dyn int {   // a: 0, return: +1
  let y = deferred();    // 0  (pinned). dyn result → y at +1.
  y + a                   // +1  (pure; no slack)
}
```

The `let` pins the call at phase 0.
`deferred()` returns `dyn int` (result offset +1), so `y` is at `0 + 1 = +1`.
`y + a` has earliest `max(+1, 0) = +1`, latest +1 (dyn return). ✓.

#### Multiple uses of a `let` binding

```silicon
fn foo(const x: int, y: int) -> int { x + y }

pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  let x = const { a + 42 };    // x at -1
  foo(x, x + b)                 // 0
}
```

`x` at -1 is used in two places:
- As foo's const arg: needs ≤ `0 + (-1)` = -1. `x` at -1 ≤ -1. ✓.
- In `x + b`: pure op, earliest `max(-1, 0) = 0`. ✓.

Both consumers are satisfied.
No conflict — the value was computed once at -1 and is available to all later phases.

#### Summary

`let x = expr;` pins the execution context at `p(block)`, ensuring side effects happen "here."
The value `x` can be at a different phase:
- Same as `p(block)` for plain expressions (`let x = a + 42;`)
- Shifted by `const { ... }` or `dyn { ... }` (`let x = const { ... };`)
- Shifted by call result offsets (`let x = foo();` where foo returns `dyn int`)

The user controls the value's phase explicitly through `const`/`dyn` annotations.
Without annotation, `let` pins at the block phase — safe and predictable.

### Example 10: Control flow

#### `if` expression — resolves to latest

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  if a > 0 {     // 0
    a + 42        // [-1, 0] → -1  (pure; slack)
  } else {
    b + 1          // 0  (pure; no slack)
  }
}
```

`if` is anchored at `p(block) = 0` (function body phase).
Both branch blocks at 0.
Condition `a > 0` is pure; earliest `max(-1, -∞) = -1`, latest 0. `[-1, 0] → -1` (slack).
`a + 42` is pure; `[-1, 0] → -1` (slack, floats to -1 inside the branch).
`b + 1` is pure; no slack, stays at 0.
Feasibility: if earliest = condition resolved = -1. -1 ≤ 0. ✓.

#### `if` as a const call arg — floats via `uir.expr`

```silicon
fn foo(const x: int, y: int) -> int { x + y }

pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  foo(
    if a > 0 { a + 42 } else { a - 1 },    // -1
    b,
  )                                           // 0
}
```

At the language level, the `if` expression floats to -1 (from foo's const arg).
At the IR level, codegen wraps the `if` in a floating `uir.expr`:

```
%r = uir.expr : %t {
  %v = uir.if %cond : %t { ... } else { ... }
  uir.yield %v : %t
}
```

The `uir.expr` floats to -1 (from the call's const arg demand).
The `uir.if` inside is anchored at the expr's block phase, which is -1.
Both branch blocks at -1.
Condition `a > 0` is pure; earliest -1, latest -1. No slack. ✓.
`a + 42` and `a - 1` are pure; both have earliest -1. No slack. ✓.
Feasibility: condition earliest = -1 ≤ -1. ✓.

#### `if` as a dyn call arg — floats to later phase via `uir.expr`

```silicon
fn bar(x: int, dyn y: int) -> dyn int { x + y }

pub fn main(a: int, dyn b: int) -> dyn int {   // a: 0, b: 1, return: 1
  bar(
    a,
    if b > 0 { b + 1 } else { b - 1 },    // 1
  )                                           // 1
}
```

Symmetric to the const case: codegen wraps the `if` in a floating `uir.expr`.
The `uir.expr` floats to 1 (from bar's dyn arg demand).
The `uir.if` inside is anchored at the expr's block phase, which is 1.
Both branch blocks at 1.
Condition `b > 0` is pure; earliest 1 (from `b`), latest 1. No slack. ✓.
`b + 1` and `b - 1` are pure; both have earliest 1. No slack. ✓.
Feasibility: condition earliest = 1 ≤ 1. ✓.

#### `if` with `return` — pinned by constraint

```silicon
fn foo(const x: int, y: int) -> int { x + y }

pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  foo(
    if a > 0 { return 0; a } else { a - 1 },   // ERROR
    b,
  )
}
```

Codegen wraps the `if` in a floating `uir.expr` (as a call arg to foo).
The `uir.expr` floats to -1 (from foo's const arg demand).
The `uir.if` inside is anchored at the expr's block phase, which is -1.
The `return` requires `p(enclosing_block) = p(function_body) = 0`.
But `p(enclosing_block) = p(if) = -1`. **-1 ≠ 0. ERROR.**

```
error: `return` cannot be used in a const context
 --> main
  |     if a > 0 { return 0; a } else { a - 1 },
  |                ~~~~~~
note: this block is const because it is a const argument to `foo`
 --> main
  |   foo(
  |   ~~~
```

The `return` prevents the `uir.expr` from floating to -1, because the return's target (function body) is at phase 0.
Without the `return`, the `uir.expr` and its `uir.if` would float to -1 without conflict.

#### `if` with `return` — no conflict when at body phase

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  if a > 0 {        // 0
    return a + 42;   // p(block) = 0 = p(function_body). ✓.
  }
  b
}
```

The `if` is the block result, latest = 0 (from return type).
`p(if) = 0`. Branch block at 0.
`return` checks `p(block) = 0 = p(function_body) = 0`. ✓. No conflict.

#### `loop` with `break`

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  loop {             // 0
    if a > 0 {       // 0  (statement, pinned)
      break a + 42;  // p(block) = 0 = p(loop). ✓.
    }
  }
}
```

`p(loop) = 0`. Loop body block at 0.
`if` is a statement, pinned at 0. Branch block at 0.
`break` checks `p(block) = 0 = p(loop) = 0`. ✓.
`break` value `a + 42` is pure; `[-1, 0] → -1` (slack).
The loop's result is at `p(loop) = 0`.

#### `break` from a floated context — error

```silicon
fn foo(const x: int) -> int { x }

pub fn main(a: int) -> int {   // a: 0, return: 0
  loop {                        // 0
    foo(
      if true {                 // -1  (from const arg, via uir.expr)
        break 42;               // p(block) = -1 ≠ p(loop) = 0. ERROR!
        0
      } else { 0 },
    );
  }
}
```

`p(loop) = 0`. Codegen wraps the `if` in a floating `uir.expr` (call arg).
The `uir.expr` floats to -1 (from foo's const arg). The `uir.if` is anchored at -1.
`break` checks `p(block) = -1 = p(loop) = 0`. **ERROR.**
The `break` can't cross the phase boundary created by foo's const arg.

#### `while` loop

```silicon
pub fn main(const n: int) -> int {   // n: -1, return: 0
  let i = const { n };               // i at -1
  while i > 0 {    // 0  (statement, pinned)
    // loop body at phase 0
  }
  i
}
```

`while` is a statement, pinned at 0.
Condition `i > 0` is pure; earliest `max(-1, -∞) = -1`, latest 0. `[-1, 0] → -1` (slack).
The condition can be evaluated early, but the loop executes at phase 0.

#### `match` expression

```silicon
pub fn main(const a: int, b: int) -> int {   // a: -1, b: 0, return: 0
  match a {          // 0
    0 => b + 1,      // 0  (pure; no slack)
    _ => a + 42,     // [-1, 0] → -1  (pure; slack)
  }
}
```

`match` is anchored at `p(block) = 0`. All arm blocks at 0.
Matched expression `a`: at -1, latest 0. ✓.
Each arm follows the same rules as `if` branches.
`a + 42` floats to -1 (pure, eager). `b + 1` stays at 0.

#### IR representation

These control flow constructs map to `uir` dialect ops (see `docs/design/unified-dialect.md`):
- `if`/`else` → `uir.if` with `then`/`else` regions.
- `loop`/`while`/`for` → `uir.loop` with a body region. `while` and `for` desugar to `uir.loop` + `uir.if`.
- `match` → `uir.match` with per-arm regions.
- `break` → `uir.break`, `continue` → `uir.continue`, early `return` → `uir.return`.
- After exhaustive early exits: `uir.unreachable`.

`uir.break`/`uir.continue`/`uir.return` are region terminators (alternatives to `uir.yield`).
Phase inference works on the structured `uir` ops; the **FlattenCF** pass converts them to block-based `cf.br`/`cf.cond_br` after phase splitting.

**CF ops are anchored at their block phase.**
`uir.if`, `uir.loop`, and `uir.match` always execute at `p(enclosing_block)`.
They do not float on their own.
When a CF expression needs to float to a different phase (e.g., as a const or dyn call argument), codegen wraps it in a floating `uir.expr { ... }` — the `uir.expr` is what floats, and the CF op inside is anchored at the expr's block phase.
This keeps CF ops simple and predictable: their phase is always determined by the enclosing block, never by consumer demand.

#### Summary

At the **language level**, calls and control flow expressions (`if`, `loop`, `while`, `for`, `match`) appear to resolve to **latest** (lazy, top-down) — when used as subexpressions in call arguments or assignments, they snap to the consumer's demanded phase.

At the **IR level**, this is implemented by anchoring all side-effecting ops at their enclosing block phase:
- `uir.call`, `uir.if`, `uir.loop`, `uir.match` are always at `p(enclosing_block)`.
- When the language requires floating (e.g., a call or `if` as a const call arg), codegen wraps the op in a floating `uir.expr`, which carries the consumer demand.
- The wrapped op is then anchored at the `uir.expr`'s block phase, which is the consumer's demanded phase.
- Only pure ops and ConstantLike ops may float to earlier phases on their own; everything else requires a `uir.expr` wrapper.

Sub-blocks of CF ops are at the CF op's phase.
Call args must be available at `p(call) + arg_offset_i` (feasibility check).
The condition/iterator of a CF op must be available at or before the CF op's phase.
`uir.return`/`uir.break`/`uir.continue` impose a constraint: `p(enclosing_block) = p(target)`.
This constraint prevents the enclosing `uir.expr` from floating to a phase where the CF transfer would cross a phase boundary.

**`uir.yield` and `uir.break` are phase-transparent conduits.**
They have an execution phase (for CFG ordering), but they relay values and phase constraints without imposing their own phase onto the values.
Phase constraints from the consumer of a region op's result are forwarded transparently through the yield onto the yielded value.
The region op's result is at the yielded value's actual phase, not the region op's execution phase.
This enables nested `dyn` blocks to produce "futures" — values from later phases that flow through earlier-phase yields as opaque handles.
Symmetrically, `const` block results at earlier phases flow naturally through later-phase yields (as they are already available).

### Example 11: Types and dependent types

#### Concrete types — trivially satisfied

```silicon
fn foo(x: int, y: int) -> int { x + y }
//         ~~~     ~~~     ~~~
//     type `int`: -∞. Latest = p(x) - 1 = -1. -∞ ≤ -1. ✓.
```

Concrete types like `int` are ConstantLike at `-∞`.
They always satisfy the "one phase before" requirement.
No user impact for non-dependent types.

#### Simple dependent type — `const` satisfies the constraint

```silicon
fn foo(
  const N: int,    // -1.  Type `int`: -∞. Latest = -1 - 1 = -2. ✓.
  x: uint<N>,      // 0.   Type `uint<N>`: [-1, -1] → -1. Latest = 0 - 1 = -1. ✓.
) -> uint<N> {     // 0.   Type `uint<N>`: [-1, -1] → -1. Latest = 0 - 1 = -1. ✓.
  x
}
```

`uint<N>` is a pure op; earliest `max(p(N)) = -1`. Latest = `p(x) - 1 = -1`. No slack.
The type is known at phase -1, one phase before `x` arrives at phase 0.
During splitting, the type computation goes into the phase -1 function.

#### Missing `const` — error

```silicon
fn foo(
  N: int,          // 0
  x: uint<N>,      // 0.   Type `uint<N>`: earliest 0, latest -1. ERROR!
) -> uint<N> {     // 0.   Same error.
  x
}
```

`uint<N>` needs to be at phase -1 (one before `x` at 0), but `N` is at phase 0.
The type can't be known in time to compile the phase-0 program.

```
error: `N` must be const to be used in the type of `x`
 --> foo
  |   N: int,
  |   ^
note: consider making `N` constant: `const N: int`
```

#### Nested dependent types

```silicon
fn foo(
  const const M: int,   // -2
  const N: int,          // -1
  x: uint<M>,            // 0.   Type `uint<M>`: [-2, -1] → -2. ✓.
  y: uint<N>,            // 0.   Type `uint<N>`: [-1, -1] → -1. ✓.
) -> uint<N> {           // 0.   Type `uint<N>`: [-1, -1] → -1. ✓.
  x + y
}
```

`M` at -2 and `N` at -1 both satisfy the "one phase before" rule for their respective types.
`uint<M>` resolves to -2 (pure, eager — even more slack than needed).
`uint<N>` resolves to -1 (no slack).

#### `let` with type ascription

```silicon
pub fn main(const N: int) -> int {   // N: -1, return: 0
  let x: uint<N> = const { ... };    // x at -1. Type `uint<N>` latest = -1 - 1 = -2.
                                      // uint<N> earliest = -1. [-1, -2] → ERROR!
  0
}
```

Here `x` is at -1 (const shifts the let).
The type needs to be at -2.
But `uint<N>` can only be at -1 (from `N`). Error!

This is correct: to compile the phase -1 program that produces `x`, we need to know `x`'s type at phase -2.
The fix: make `N` even earlier with `const const N: int` (at -2).

```silicon
pub fn main(const const N: int) -> int {   // N: -2, return: 0
  let x: uint<N> = const { ... };           // x at -1. Type latest = -2.
                                             // uint<N> earliest = -2. ✓.
  0
}
```

#### Type ascription on a plain `let` — common case

```silicon
pub fn main(const N: int) -> int {   // N: -1, return: 0
  let x: uint<N> = some_expr;        // x at 0. Type `uint<N>` latest = -1.
                                      // uint<N> earliest = -1. ✓.
  0
}
```

When `x` is at phase 0 (normal let, no const shift), the type needs to be at -1.
`N` at -1 satisfies this — the standard dependent type pattern works naturally.

#### Computed type width

```silicon
fn foo(
  const A: int,    // -1
  const B: int,    // -1
  x: uint<A + B>,  // 0.   Type `uint<A + B>`: A + B is pure, earliest max(-1, -1) = -1.
                   //      Latest = 0 - 1 = -1. ✓.
) -> uint<A + B> { // 0.   Same. ✓.
  x
}
```

`A + B` is a pure op at -1. `uint<A + B>` is a pure op at -1. Both satisfy the -1 requirement.
Type arithmetic works naturally — the entire type expression is a pure op tree that resolves eagerly.

#### Type depending on a call result

```silicon
fn compute_width(const a: int, const b: int) -> int { a + b }

fn foo(
  const A: int,                  // -1
  const B: int,                  // -1
  x: uint<compute_width(A, B)>,  // 0.   Type: compute_width is a call.
) -> int {                       //      Type needs phase -1. Call in uir.expr floats to -1.
  0                               //      Arg feasibility: -1 + (-1) = -2, args at -1. ERROR!
}                                  //
```

The call to `compute_width(A, B)` in the type expression is wrapped in a `uir.expr` (subexpression of the type annotation).
The type context demands phase -1. The `uir.expr` floats to -1. The call inside is at -1.
Arg feasibility: `compute_width` has const args (offset -1), so args need `-1 + (-1) = -2`. `A` and `B` are at -1. `-1 > -2`: **ERROR!**
The args aren't available early enough for the call to execute at -1.

The fix: `compute_width` would need to return `const int` (result offset -1):
```silicon
fn compute_width(const a: int, const b: int) -> const int { a + b }
```
Now the `uir.expr` floats to 0 (from the type context: `-1 - (-1) = 0`).
The call is at 0. Args need `0 + (-1) = -1`. `A` and `B` at -1: ✓.
Result at `0 + (-1) = -1`. The type is at -1. ✓.

Or alternatively, use a pure expression instead of a call: `uint<A + B>` works directly because pure ops are eager.

#### Two args sharing a dependent type

```silicon
fn foo(const N: int, x: uint<N>, y: uint<N>) -> uint<N> {
  x + y
}
```

`N` at -1, `x` at 0, `y` at 0, return at 0.
Three occurrences of `uint<N>`:
- Type of `x`: latest = `0 - 1 = -1`. `uint<N>` earliest = -1. ✓.
- Type of `y`: latest = `0 - 1 = -1`. Same. ✓.
- Return type: latest = `0 - 1 = -1`. Same. ✓.

All three are the same pure expression `uint<N>`, and all resolve to -1.
No conflict.

Inside the body: `x + y` is pure, earliest `max(0, 0) = 0`, latest 0. No slack.
The `+` needs to know the type of its operands to compile — the types are at -1, one phase before the body at 0. ✓.

#### Const arg with dependent type — deepens the requirement

```silicon
fn foo(const N: int, const x: uint<N>, y: uint<N>) -> uint<N> {
  x + y
}
```

`N` at -1, `x` at -1, `y` at 0.
- Type of `x`: latest = `-1 - 1 = -2`. `uint<N>` earliest = -1. `[-1, -2]` → **ERROR!**

`N` at -1 can't produce a type at -2.
The fix: `const const N: int` (at -2).
This makes sense — if `x` is const, its type must be known even earlier.

#### Dyn arg with dependent type — relaxes the requirement

```silicon
fn foo(const N: int, dyn x: uint<N>) -> dyn uint<N> {
  x
}
```

`N` at -1, `x` at +1, return at +1.
- Type of `x`: latest = `+1 - 1 = 0`. `uint<N>` earliest = -1. `[-1, 0] → -1` (pure, eager). ✓.
- Return type: latest = `+1 - 1 = 0`. Same. ✓.

The `dyn` shift gives the type *more* room.
The type needs to be at phase 0, but `uint<N>` can be at -1. Plenty of slack.
This is the opposite of the const case: `dyn` makes type requirements easier, `const` makes them harder.

#### Dyn arg with dyn type parameter — error

```silicon
fn foo(dyn N: int, dyn x: uint<N>) -> dyn uint<N> {
  x
}
```

`N` at +1, `x` at +1.
- Type of `x`: latest = `+1 - 1 = 0`. `uint<N>` earliest = `max(+1) = +1`. `[+1, 0]` → **ERROR!**

`N` at +1 can't produce a type at phase 0.
Even with `dyn` on both, the type-determining value must be at a strictly earlier phase than the typed value.
The fix: make `N` a regular arg (phase 0) or const.

#### Summary

Type expressions have `latest = p(annotated_value) - 1`: types must be known one phase before the values they describe, so the phase's program can be compiled.
In the DFS, the expression is processed first to determine `p(value)`, then `latest = p(value) - 1` is pushed to the type expression.

- **Concrete types** (`int`, `bool`): at `-∞`, always satisfied.
- **Dependent types** (`uint<N>`): pure op, resolves eagerly. The type-determining value (`N`) must be at a strictly earlier phase than the annotated value.
- **Type arithmetic** (`uint<A + B>`): pure ops compose naturally.
- **Call results in types** (`uint<compute_width(A, B)>`): calls are anchored at the `uir.expr`'s block phase (which floats to the type's demand), but arg offset feasibility can conflict with the type's deadline. Use pure expressions or `const` result modifiers.
- **`let` with const shift**: shifting a value earlier also shifts the type requirement earlier, potentially requiring deeper `const const` on dependencies.
- **`const` on typed arg**: deepens the type requirement by one phase. `const x: uint<N>` needs the type at `p(x) - 1`, which is one phase earlier than for a regular arg.
- **`dyn` on typed arg**: relaxes the type requirement. The type has more time to be computed.
- **General rule**: the type-determining value must always be at least one phase ahead of the typed value, regardless of other modifiers.

### Example 12: Globals

#### Global constants — ConstantLike

```silicon
const static PI: float = 3.14159;
const static MAX_WIDTH: int = 64;
```

Simple constant globals are ConstantLike at `-∞`.
They can be used in any phase, any context, cloned into whatever phase needs them during splitting.
Same as literals.

#### Globals with computation

The module level is a block at phase 0.
Global declarations are statements in that block.
`const`/`dyn` are phase modifiers (not mutability — that's a separate concept like `mut`).

```silicon
const static WIDTH: int = compute(42);     // pinned at 0. const shifts → WIDTH at -1.
static COUNT: int = compute(42);           // pinned at 0. COUNT at 0.
dyn static DEFERRED: int = deferred(42);   // pinned at 0. dyn shifts → DEFERRED at +1.
```

The expressions are pinned at the module body phase (0), same as `let` bindings.
The `const`/`dyn` modifier shifts the resulting value's phase.
Inside, the same DFS rules apply: calls and CF are anchored at the block phase, pure ops float eagerly, types must be one phase before the value.

#### Multi-phase call in a global

```silicon
fn foo(x: int) -> (const int, int, dyn int) { ... }

static (a, b, c) = foo(42);
// Call pinned at 0 (module body phase).
// a at 0 + (-1) = -1
// b at 0 + 0 = 0
// c at 0 + 1 = +1
```

The call is pinned at phase 0.
Each result is at `p(call) + result_offset`.
During module-level phase splitting:
- Phase -1: `a` becomes available.
- Phase 0: `b` becomes available.
- Phase +1: `c` becomes available.

Each global becomes available in its corresponding phase, exactly like how local values in a function body get assigned to split functions.

#### Type constraints on globals

```silicon
const static WIDTH: uint<N> = const { compute(N) };
// WIDTH at -1. Type `uint<N>` latest = -1 - 1 = -2.
// Same rules as `let` type ascriptions.
```

The type rules for globals are identical to `let` bindings: `latest(type) = p(value) - 1`.

#### Open questions for implementation

- How are globals with non-trivial computation represented in MLIR?
  The computation may need its own unified region (similar to a function body) that gets split by phase splitting into separate regions for each phase.
- How does module-level phase splitting interact with function-level splitting?
  The module's phase -1 program compiles the phase 0 program, which may itself contain functions that split across phases.

#### `const fn` / `dyn fn` — not needed

We considered `const fn` / `dyn fn` as function-level phase modifiers that would shift the entire call by -1 / +1.
After analysis, these don't bring distinct semantics beyond arg/result modifiers and `const { ... }` / `dyn { ... }` blocks at call sites.
The existing tools cover all use cases:
- "Call at earlier phase" → `const { foo(...) }` at call site
- "Result at later phase" → `-> dyn int` on the function
- "All args earlier" → `const` on each arg

See TODO.md for the note to remove `const fn` / `dyn fn` from the parser and codegen.

### Example 13: Other constructs

#### Short-circuit operators (`&&`, `||`)

`a && b` desugars to `if a { b } else { false }`.
`a || b` desugars to `if a { true } else { b }`.
They are CF expressions and snap to latest, same as `if`.
No new phase rules needed.

#### Method calls (`x.foo(y)`)

Syntactic sugar for `foo(x, y)` (or `Trait::foo(x, y)`).
`self` is the first argument with its own phase offset from the function signature.
Identical to regular calls for phase purposes.

#### Recursion

A recursive call uses the same function's signature.
Phase offsets come from the signature, which is known.
No circular dependency — each call is analyzed independently.

```silicon
fn factorial(
  const n: int,     // -1
) -> int {           // return: 0, body: 0
  if n <= 1 {        // 0
    1                // -∞
  } else {           // block: 0
    n                // -1
      * factorial(   // 0 (call, block phase)
          n - 1      // [-1, -1] → -1 (pure; no slack)
        )            // result at 0
  }                  // 0 (pure `*`; earliest = max(-1, 0) = 0, no slack)
}
```

The recursive `factorial(n - 1)` call needs its const arg at -1.
`n - 1` is pure at -1 (from `n`). ✓.
Each recursive call specializes for a different `n`, expanding at compile time.
Mutual recursion works the same way — each call uses the callee's signature.
