# Inference

This document specifies the design of type and value inference in Silicon.
It covers the core inference mechanism (inferrables and unification), the pass pipeline that implements it, provenance tracking for error diagnostics, and open design questions.

> [!WARNING]
> This document is a work in progress.

## Core Concepts

### Types as SSA Values

All HIR values have MLIR type `!hir.any`; the "actual" type of a value is represented as a separate SSA value, threaded alongside data through the IR.
Type constructor ops (`hir.int_type`, `hir.uint_type`, `hir.ref_type`, `hir.func_type`, `hir.type_type`, `hir.unit_type`, `hir.anyfunc_type`, `hir.opaque_type`) produce type values.
All are `Pure` but not `ConstantLike`.

Because types are SSA values rather than MLIR types, type inference and value inference are the same mechanism: both reduce to unifying SSA values.
For example, unifying `uint_type %a` with `uint_type %b` produces a derived constraint `unify(%a, %b)` on the width values, using the exact same algorithm that handles type-level constraints.

### Inference Variables

`hir.inferrable` creates a free inference variable — a placeholder for an unknown value.
It carries no constraints initially; its value is determined entirely by unification.

Inference variables arise in two situations:

1. **Compiler-generated:** Codegen emits `hir.inferrable` for call result types, since the callee's signature hasn't been inspected yet at that point.
   CheckCalls later resolves most of these by directly replacing inferrables with the callee's signature types.
2. **User-written `_` placeholders:** The `_` token in user code lowers to `hir.inferrable`, allowing the user to request inference for any value position.
   Examples:
   - Type annotations: `let x: uint<_> = some_uint8_expr` — the width is inferred from context.
   - Const function arguments: `foo(_, a)` where the callee's signature threads the const argument into a type constraint that resolves it.

In both cases the mechanism is the same: an inferrable is created, constraints flow into it via `hir.unify` ops, and the InferTypes pass resolves it.

### Unification

`hir.unify %a, %b` asserts that two SSA values must be equal.
It produces a single result that replaces both operands in subsequent uses.
The op is `Commutative`, `Pure`, `SameOperandsAndResultType`, and folds when both operands are identical.

Unify ops are created by:

- **Codegen:** to constrain operand types of binary ops (e.g., `hir.add` unifies its two operand types).
- **CheckCalls:** to constrain call argument types against the callee signature's declared parameter types, and to constrain body return types against declared result types.

The semantics of `hir.unify` are purely declarative: it asserts equality.
It is up to the InferTypes pass to resolve this assertion by replacing one operand with the other, or to leave it for CheckTypes to report as an error.

## Pass Pipeline

The inference-related passes execute in the following order:

```
Canonicalize + CSE
→ CheckCalls
→ Canonicalize + CSE + Canonicalize
→ InferTypes
→ Canonicalize + CSE
→ CheckTypes
→ SplitPhases
→ ...
```

### CheckCalls Pass

Processes functions in callee-before-caller (post-order) traversal of the call graph within function signatures.
Detects and reports recursion errors in function signatures.

For each call site, CheckCalls:

1. Clones the callee's signature region into the caller.
2. Replaces signature block arguments with actual call arguments.
3. Inserts `hir.coerce_type` ops on body block arguments to connect them to declared parameter types.
4. For inferrable placeholders on call type-of-args and type-of-results: directly replaces them with the corresponding signature type and erases the inferrable op.
5. For concrete (non-inferrable) type operands on calls: emits `hir.unify` between the declared signature types and the actual call type operands.
6. Unifies body return types with declared result types from the signature.

Step 4 is a **dominance workaround**: when CheckCalls inlines signature ops, they appear after inferrable placeholders created by codegen.
If CheckCalls emitted `hir.unify` instead, InferTypes would fail its dominance check because the concrete value (from the inlined signature) does not dominate the inferrable.
The direct replacement avoids this, but couples CheckCalls tightly to the inference algorithm.

> **Open question:** Can we remove this workaround?
> One option is for InferTypes to clone side-effect-free concrete ops to the inferrable's location when the concrete value doesn't dominate.
> This is partially implemented (see "cloning" case in InferTypes) but may not cover all scenarios.
> Another option is to restructure the IR so that signature ops are inserted before the call site's inferrables, but this requires changes to codegen.

### InferTypes Pass

Collects all `hir.unify` ops into a worklist and processes them in reverse order.
The pass is best-effort and silent: it resolves what it can and leaves everything else for CheckTypes to report.

Four cases:

1. **Both operands inferrable:** Keep the one earlier in dominance order, replace all uses of the other, erase both the later inferrable and the unify op.
2. **One inferrable, one concrete (concrete dominates):** Replace all uses of the inferrable with the concrete value, erase both.
3. **One inferrable, one concrete (concrete does not dominate):** If the concrete op is side-effect-free, has no regions, and all its operands dominate the inferrable, clone it to the inferrable's location and resolve. Otherwise, skip silently — CheckTypes will report the error.
4. **Both concrete:** Check structural equivalence using `OperationEquivalence` (ignoring value identity, both side-effect-free with a single result and no regions). If equivalent, keep the dominating op, erase the other, and create new `hir.unify` ops for each pair of differing operands (added to the worklist for recursive processing). If not structurally equivalent, skip — CheckTypes will report the error.

The pass uses `DominanceInfo` throughout to ensure replacements produce valid SSA.

When creating derived unify ops (case 4), the pass must propagate provenance from the parent unify op to the new child unify ops — see [Provenance Tracking](#provenance-tracking) below.

### CheckTypes Pass

Runs after InferTypes and inspects all remaining `hir.unify` and `hir.inferrable` ops.
This is the single location for user-facing type and value error diagnostics.

CheckTypes reports errors for:

1. **`hir.unify` with two different concrete type constructors:** A type mismatch that inference could not resolve. Reports "type mismatch: X is not compatible with Y" using the op locations and provenance chain to explain *why* the constraint exists.
2. **Unresolved `hir.inferrable`:** A free variable that no constraint resolved. Reports "cannot infer value of ..." using the inferrable's location to identify what couldn't be inferred (e.g., "placeholder `_` at argument 1 of call to `foo`").

> **Open question:** Should CheckTypes also report `hir.unify` ops where one or both operands are non-concrete and non-inferrable (e.g., `coerce_type`, `type_of`, block arguments)?
> Currently these are skipped under the assumption that later pipeline stages may resolve them.
> But if nothing resolves them, they silently survive into HIRToMIR where they become internal assertions rather than user-facing errors.
> We need to determine the exact set of patterns that CheckTypes should flag versus defer.

> **Open question:** Error recovery.
> Currently, if one `hir.unify` is an error, the pass reports it but leaves the broken op in the IR.
> The rest of the function body may contain cascading errors from the same root cause.
> Should CheckTypes attempt to continue (e.g., by replacing the failed unify with one of its operands) and report multiple independent errors?
> Or should it stop at the first error per function?

## Provenance Tracking

When inference fails, the user needs to understand *why* two values were constrained to be equal and *where* the conflicting constraint originated.
Silicon uses MLIR's location system for this rather than custom attributes, because locations survive transformations (canonicalization, CSE, folding) automatically.

### Location Mechanisms

Three MLIR location types are relevant:

- **`NameLoc`**: Attaches a descriptive string to a location. Used when creating `hir.inferrable` and `hir.unify` ops to describe their origin, e.g., `NameLoc("call result type of 'foo'", <source loc>)`.
- **`CallSiteLoc`**: Pairs a callee location with a caller location. Used by CheckCalls when inlining signature constraints: the resulting `hir.unify` ops get a `CallSiteLoc` pointing from the parameter type in the signature back to the call site in the caller.
- **`FusedLoc`**: Combines multiple locations with optional metadata. Used by InferTypes when creating derived constraints: a new `unify(%a, %b)` derived from structurally matching `unify(uint_type %a, uint_type %b)` gets a fused location combining the child position ("width operand") with the parent unify's location.

### What Gets Tagged

- **Codegen** sets `NameLoc` on compiler-generated inferrables to describe their role (e.g., "result type of call to `foo`").
- **Codegen** sets `NameLoc` on user-written `_` placeholders to indicate the source position and syntactic context (e.g., "placeholder `_` for argument 1 of call to `foo`").
- **CheckCalls** sets `CallSiteLoc` on unify ops it creates, linking the callee's signature constraint to the call site.
- **InferTypes** propagates or fuses locations when creating derived unify ops from structural matching.

### Diagnostic Construction

CheckTypes walks the location chain on a failed `hir.unify` to construct a multi-part diagnostic:

1. The primary error: "type mismatch: `uint` is not compatible with `int`".
2. Notes tracing the provenance: "required because argument 2 of `foo` has type `uint<N>`", "in call at line 42".

> **Open question:** The exact format and depth of provenance traces needs to be determined through experimentation.
> How many levels of "because ..." are useful before they become noise?
> Should the trace include the full chain, or only the most relevant link?

> **Open question:** When CSE merges two structurally identical ops, MLIR fuses their locations into a `FusedLoc`.
> This is generally desirable, but could it produce confusing diagnostics if a type error points to two merged locations?
> Need to verify that CSE-induced location fusion doesn't degrade error quality.

## Value Inference

Because types are SSA values, value inference falls out of the existing type inference mechanism without any additional algorithm.

### How It Works

Consider `hir.uint_type %width`, where `%width` is an SSA value representing the bit width.
If two uses require `uint<N>` and `uint<M>`, codegen or CheckCalls emits `hir.unify(uint_type %n, uint_type %m)`.
InferTypes recognizes both as `uint_type` ops (structural equivalence) and emits `hir.unify(%n, %m)`.
If one of `%n`, `%m` is an inferrable (from a `_` placeholder), it gets resolved to the other.
If both are concrete constants, they must match or CheckTypes reports an error.

### Const Argument Inference

The `_` placeholder enables inference of const function arguments from type constraints:

```silicon
fn foo(const N: int, x: uint<N>) -> uint<N> { ... }
fn main(a: uint<42>) {
  foo(_, a);  // _ is inferred as 42
}
```

The mechanism:

1. Codegen lowers `_` to `hir.inferrable` at the call site.
2. CheckCalls inlines `foo`'s signature, which declares parameter `x` has type `uint_type %N` where `%N` is the first argument.
3. CheckCalls emits `hir.unify` between the actual type of `a` (`uint_type(const 42)`) and the declared type `uint_type %N`.
4. InferTypes structurally matches the two `uint_type` ops and emits `hir.unify(inferrable_for_N, const 42)`.
5. The inferrable resolves to `42`.

This generalizes: any const argument that flows into a type position in the callee's signature can be inferred from the actual argument types at the call site.

### Partial Const Argument Inference

When a function has multiple const parameters, the user can infer some and specify others:

```silicon
fn bar(const N: int, const M: int, x: uint<N>, y: uint<M>) { ... }
fn main(a: uint<8>, b: uint<16>) {
  bar(_, _, a, b);    // infer both N=8, M=16
  bar(8, _, a, b);    // specify N, infer M=16
}
```

Each `_` becomes an independent inferrable, resolved (or not) by its own constraint chain.

### Limitations of Structural Matching

Structural matching handles direct constructor operands: `unify(uint_type %a, uint_type %b)` recurses to `unify(%a, %b)`.
It does **not** handle computed or derived values:

- `uint_type(add %a, const 1)` vs `uint_type %b` where `%b = add %a, const 1` — these are structurally different DAGs even though they compute the same value.
- `uint_type %a` vs `uint_type %b` where `%a` and `%b` are block arguments with no visible definition to match structurally.

**Decision:** Silicon does **not** perform symbolic reasoning or constraint solving on values.
Unification is strictly structural: if two ops are not `OperationEquivalence`-equivalent, they do not unify.
Users must be explicit about computed types.
This keeps the inference algorithm simple and predictable.
If symbolic width reasoning is needed in the future (e.g., for automatic bit-widening), it would require a separate constraint solver — see [Open Design Questions](#open-design-questions).

## Current State of the Implementation

### What Works

- `hir.inferrable` ops are created by codegen for call result types and if-expression result types.
- `hir.unify` ops are created by CheckCalls for argument type constraints and return type constraints, and by codegen for binary op operand types.
- InferTypes resolves all four cases (both inferrable, one inferrable + one concrete with or without dominance, both concrete with structural equivalence).
- CheckTypes reports type mismatches between two concrete type constructors.
- `hir.type_of %v` extracts the type of a value as an SSA value.
  Has a folder (extracts type from `unified_call` results) and a canonicalizer (creates `type_of` ops from call type operands).
- `hir.coerce_type %v, %t` annotates a value with a declared type (used at function entry for arguments, and at call sites for results).

### What Is Missing

- **`_` placeholder syntax:** The lexer and parser do not yet support `_` as an expression that lowers to `hir.inferrable`.
- **Provenance tracking:** No `NameLoc`, `CallSiteLoc`, or `FusedLoc` annotations are set by codegen, CheckCalls, or InferTypes. All ops use the default source location from the parsed token.
- **Unresolved inferrable errors:** CheckTypes does not yet report unresolved `hir.inferrable` ops; it only checks `hir.unify` mismatches.
- **Non-constructor unify operands:** CheckTypes skips `hir.unify` ops where either operand is not a concrete type constructor. Need to determine which of these represent real errors vs. deferred constraints.
- **Error recovery:** No mechanism to continue checking after a type error and report multiple independent errors.

## HIR-to-MIR Lowering Interactions

The HIRToMIR pass interacts with inference in several ways:

- `shouldLower()` gates which functions are lowered: a function is only lowered when all type operands are resolvable to concrete MIR types.
  Functions with unresolved types are skipped and deferred to a later pipeline iteration.
- `isResolvableType()` checks type values: simple type constructors (`int_type`, `unit_type`, `type_type`, `anyfunc_type`, `opaque_type`) always resolve; `uint_type` needs a constant integer width; `func_type` needs all arg/result types resolvable; `ConstantLike` ops with a non-`!hir.any` `TypeAttr` resolve.
- `hir.inferrable` is lowered to a dummy `mir.constant #mir.type<!hir.any>` — by this point they should have been resolved, so only dead metadata remains.
- `hir.type_of` is similarly lowered to a dummy `mir.constant` — it is consumed by binary ops, return ops, etc. and then discarded.
- `hir.unify` emits an error if its two operands differ after conversion ("hir.unify survived to HIR-to-MIR lowering with different operands"); if they are the same, the operand is forwarded.
- `hir.coerce_type` verifies the input type matches the type operand's constant value, then forwards the input directly.
- Binary ops: arithmetic/bitwise ops discard the type operand and forward lhs/rhs; comparison ops pass the result type explicitly since the result type may differ from the operand type.

The `shouldLower()` gate acts as an implicit backstop: if inference and CheckTypes somehow miss an unresolved type, the function simply won't lower, and the PhaseEvalLoop will eventually report a progress error.
This is a safety net, not the intended error reporting mechanism.

## Open Design Questions

### Subtype and Coercion Relationships

Unification is strict equality only.
There is no notion of widening (e.g., `uint<8>` to `uint<16>`), implicit coercion, or subtyping.
`hir.coerce_type` exists in the IR but is a no-op in lowering.

> When is implicit widening allowed?
> Should `hir.coerce_type` insert runtime conversion ops when the types are related but not identical?
> Or should all conversions be explicit in the source language?
> This is a language design question that affects the inference algorithm: if subtyping exists, `hir.unify` (equality) is insufficient and we need a directional constraint (e.g., "`%a` is assignable to `%b`").

### `hir.coerce_type` Semantics

Currently a pure annotation erased during lowering (input forwarded after type check).
CheckCalls inserts it at function entry to associate body arguments with their declared types from the signature.

> Should `hir.coerce_type` create an implicit `hir.unify` constraint between the input's type and the declared type?
> Currently it does not, which means the constraint only exists if CheckCalls explicitly emits a unify op.
> Making `hir.coerce_type` imply unification would simplify CheckCalls but change the semantics of the op.

### Unify Op Result Semantics

`hir.unify` produces a result with `SameOperandsAndResultType`.
But since all HIR types are `!hir.any`, this trait is trivially satisfied; the _value-level_ equality is what matters.
The result is used as the "resolved" type, but the semantics of which operand "wins" is left to the InferTypes pass.

> Should the unify op be purely a side-effect constraint (no result), or is having a result important for the IR structure?
> Having a result is convenient because it lets downstream ops consume the "resolved" value directly, but it also means the unify op must remain in the IR until resolution.

### Per-Op Type Rules for Binary Ops

All binary ops (`hir.add`, `hir.sub`, `hir.eq`, etc.) take three operands: `lhs`, `rhs`, and `resultType`.
Codegen currently unifies the types of both operands (requiring them to be equal) for all binary ops.

> Different ops may need different type rules:
> - `hir.add` could widen the result (`uint<N> + uint<N> → uint<N+1>`).
> - `hir.eq` produces a boolean result type, different from the operand types.
> - A hypothetical `hir.concat` would not unify its operand types at all.
>
> Where should per-op type rules be specified?
> In codegen (when emitting the op and its unify constraints)?
> In a dedicated type-rule pass?
> In the op's verifier?
> Codegen is the most natural place, since the rules determine *which* unify ops are emitted.

### Occurs Check and Cycle Detection

The InferTypes algorithm doesn't check for cyclic constraints (e.g., `%t = hir.ref_type %t`).
SSA dominance likely prevents cycles in practice: a value cannot use itself as an operand because it doesn't dominate its own definition.

> Is SSA dominance sufficient to guarantee no cycles, or are there edge cases (e.g., through block arguments or region ops) that could introduce them?
> Should InferTypes include an explicit occurs check as a safety measure?

### Generics and Parametric Polymorphism

There is no mechanism for universally quantified type variables (e.g., `fn id<T>(x: T) -> T`).
Specialization handles monomorphization, but the inference system has no concept of type schemes, instantiation, or generalization.

> How do generic type parameters interact with phased execution?
> Are they always resolved in an earlier phase via const arguments and specialization?
> Or do we need first-class type variables with quantification?
> The `_` placeholder combined with const argument inference may cover many use cases that other languages handle with generics, since `fn foo(const N: int, x: uint<N>) -> uint<N>` is effectively generic over `N`.

### Inference Across Control Flow

The InferTypes pass operates on a flat worklist of unify ops within a function body.
It does not reason about types across branches, loops, or phi-like constructs.
`hir.expr` regions could introduce scoping challenges for inference.

> If an inferrable is used in multiple branches, each branch may produce a different concrete constraint.
> Currently, the if-expression codegen creates a single inferrable for the result type and unifies it with both branches' result types.
> This works for simple cases but may not scale to more complex control flow.

### Integer and Bitwidth Inference

`hir.uint_type` takes a width as an SSA operand, but there is no mechanism to infer widths from arithmetic or assignment context beyond structural matching.

> For hardware, bitwidth inference is critical (e.g., result of `a + b` where `a: uint<8>` and `b: uint<8>` could be `uint<8>` or `uint<9>` depending on overflow policy).
> This requires per-op type rules (see above) and potentially a constraint solver for width arithmetic.
> The current structural matching approach cannot express width relationships like "result width = max(lhs width, rhs width) + 1".

### Trait and Interface Constraints

No way to express that a type must implement a particular interface (e.g., `Add`, `Eq`).
Each binary op (`hir.add`, `hir.eq`, etc.) does not yet check that the operand types support that operation.

> Trait bounds interact with inference: if a type variable is constrained to implement `Add`, that constrains which concrete types can be inferred for it.
> This is orthogonal to the current unification algorithm and would require a separate mechanism.

### Where Does Inference Run in the Pipeline?

Currently InferTypes runs once, before SplitPhases and PhaseEvalLoop.
The phase-splits.md design doc puts inference (Canonicalize + Unify) inside the phased evaluation loop, implying inference is re-run after each phase of execution.

> Is inference expected to produce new information after specialization?
> For example, a generic function gets specialized with concrete types, then its body needs re-inference.
> The current single-run approach may be insufficient if specialization introduces new type constraints.
> However, running inference inside the PhaseEvalLoop adds complexity and may interact badly with partially-lowered IR.

### Interaction Between Type Inference and Phase Splitting

Type constructors (`hir.int_type`, etc.) are `Pure` but not `ConstantLike`, so they are not automatically available in all phases.
Only `hir.mir_constant` is `ConstantLike`.
Computed types (e.g., `hir.uint_type %width` where `%width` is a runtime value) are pinned to the phase where `%width` is available.

> Can type inference introduce cross-phase dependencies?
> What happens when a type can only be resolved in a later phase?
> This is related to the question of whether inference should run inside the PhaseEvalLoop.
