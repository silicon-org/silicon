# Inference

Bullet-point outline of the current state, inconsistencies, missing features, and open design questions around type and value inference in Silicon.
To be expanded into a proper design document later.

## Current State

### Types as SSA Values

- All HIR values have MLIR type `!hir.any`; the "actual" type of a value is represented as a separate SSA value, threaded alongside data through the IR.
- Type constructor ops (`hir.int_type`, `hir.uint_type`, `hir.ref_type`, `hir.func_type`, etc.) produce type values.
- `hir.type_of %v` extracts the type of a value as an SSA value.
- `hir.coerce_type %v, %t` annotates a value with a declared type (used at function entry for arguments, and at call sites for results).

### Inference Variables

- `hir.inferrable` creates a free type variable (placeholder for an unknown type/value).
- Codegen emits these for call result types, since the callee's signature hasn't been inspected yet at that point.
- CheckCalls later resolves most of them by inlining the callee's signature at the call site and replacing the inferrables with the signature's computed types.

### Unification

- `hir.unify %a, %b` asserts that two SSA values must be equal; produces a result that replaces both.
- The op is `Commutative`, `Pure`, `SameOperandsAndResultType`, and folds when both operands are identical.
- Codegen emits unify ops for binary operand types (`hir.unify %lhsType, %rhsType`), and CheckCalls emits them to constrain argument/result types against the callee signature.

### InferTypes Pass (`lib/HIR/Transforms/InferTypes.cpp`)

- Collects all `hir.unify` ops into a worklist and processes them in reverse order.
- Three cases:
  1. Both operands inferrable: keep the one earlier in dominance, erase the other.
  2. One operand inferrable, one concrete: replace inferrable with concrete (if concrete dominates).
  3. Both concrete: if the defining ops are structurally equivalent (`OperationEquivalence` ignoring value identity), keep the earlier op, erase the later, and recursively unify their operands.
- Uses `DominanceInfo` throughout to ensure replacements are legal.

### CheckCalls Pass (`lib/HIR/Transforms/CheckCalls.cpp`)

- Processes functions in callee-before-caller (post-order) traversal.
- Clones the callee's signature region into the caller at each call site.
- Replaces signature block arguments with actual call arguments.
- Emits `hir.unify` between declared parameter types and actual argument types, and between declared result types and the call's result type placeholders.
- Replaces `hir.inferrable` placeholders with the unified types directly, avoiding a dominance problem where inlined signature ops would appear after the inferrable.

### HIR-to-MIR Lowering (`lib/Conversion/HIRToMIR.cpp`)

- `hir.inferrable`, `hir.type_of`, `hir.unify` are all lowered to dummy `mir.constant #mir.type<!hir.any>` — by this point they should have been resolved, so only dead metadata remains.
- `hir.coerce_type` is erased; the input value passes through directly.
- Type operands on `hir.binary`, `hir.call`, etc. are consumed to determine the materialized MIR types and then discarded.
- The lowering requires type operands to be `ConstantLike` ops so they can be folded into MIR attributes.

### Phased Evaluation Pipeline (design doc, `phase-splits.md` step 1–2)

- Step 1 (Canonicalize): propagate and fold constants, including pushing unify ops through trivially equivalent ops and substituting inferrables.
- Step 2 (Unify): verify all unify ops are resolved; emit errors for incompatible types. This is the main user-facing type error mechanism.

## Bugs and Implementation Issues

### InferTypes: swap uses null pointers in concrete-concrete case

- Lines 106–107 of `InferTypes.cpp`: when both operands are concrete (not inferrable), the swap assigns `keepOp = inferrableRhs` and `eraseOp = inferrableLhs`, but both are null at that point.
  This is a latent bug that would crash if the eraseOp dominates the keepOp in the concrete-concrete case.

### Inference failures are silent

- When the InferTypes pass cannot resolve a unify op (e.g., concrete doesn't dominate inferrable, or the ops aren't structurally equivalent), it silently skips the constraint.
- The design doc says step 2 of the pipeline ("Unify") should emit user-facing type errors for unresolved unify ops, but this pass does not exist yet.
- There is no diagnostic trail explaining _why_ two types are incompatible or _where_ the conflicting constraints originated.

### Dominance workaround in CheckCalls

- SESSION.md note: when CheckCalls inlines signature ops, they appear after inferrable placeholders created by codegen, causing InferTypes dominance checks to fail.
- The current fix is for CheckCalls to replace inferrables directly rather than creating a unify with them.
- This works but couples CheckCalls tightly to the inference algorithm and may not generalize to more complex scenarios (e.g., multiple call sites constraining the same inferrable).

## Missing Features

### No user-facing type error pass

- The design doc specifies a "Unify" step that emits errors for unresolved unify ops.
  This pass does not exist; there is no way to report type mismatches to the user.
- Need to design: what does a good type error message look like? How do we trace back from a failed unify to the source-level types and expressions involved?

### No type error recovery

- If inference fails for one unify op, the pass moves on but leaves broken IR.
  No mechanism to continue checking the rest of the function body and report multiple errors.

### No subtype or coercion relationships

- Unification is strict equality only (`hir.unify`).
- No notion of widening (e.g., `uint<8>` to `uint<16>`), implicit coercion, or subtyping.
- `hir.coerce_type` exists in the IR but is a no-op in lowering; it carries no runtime or compile-time semantics beyond documentation.
- Need to decide: should coerce_type insert runtime conversion ops, or should it remain a type annotation? When is implicit widening allowed?

### No value inference (only type inference)

- The current unification only operates on type values.
- There is no mechanism to infer _non-type_ values, such as integer widths in `uint<N>` where `N` could be inferred from context.
- Example: `let x: uint<_> = some_uint8_expr` — inferring that `_` is 8.
- The design doc mentions "value inference" in passing (phase-splits.md line 8) but doesn't specify it.

### Dependent types in signatures

- TODO.md lists "dependent types in signatures (type depends on argument value)" as an open item.
- The infrastructure exists (`hir.uint_type %width` takes an SSA operand), and codegen already converts `uint<b>` to use the argument's SSA value.
- But there is no mechanism to propagate value constraints across function boundaries: if a callee declares `fn foo(a: int) -> uint<a>`, the caller needs to know that the result width equals the argument value.
- CheckCalls inlines the signature, which handles this in principle, but the unification algorithm only handles type equality, not value-level constraints like "this integer equals that integer".

### No occurs check or cycle detection in unification

- The InferTypes algorithm doesn't check for cyclic constraints (e.g., `%t = hir.ref_type %t`).
- In practice this may not arise due to SSA dominance, but it's worth specifying whether this is guaranteed or needs a check.

### No generics / parametric polymorphism

- No mechanism for universally quantified type variables (e.g., `fn id<T>(x: T) -> T`).
- Specialization handles monomorphization, but the inference system has no concept of type schemes, instantiation, or generalization.
- How do generic type parameters interact with phased execution? Are they always resolved in an earlier phase?

### No inference across control flow

- The InferTypes pass operates on a flat worklist of unify ops.
- It does not reason about types across branches, loops, or phi-like constructs.
- `hir.expr` regions could introduce scoping challenges for inference.

### No integer/bitwidth inference

- `hir.uint_type` takes a width as an SSA operand, but there is no mechanism to infer widths from arithmetic or assignment context.
- For hardware, bitwidth inference is critical (e.g., result of `a + b` where `a: uint<8>` and `b: uint<8>` could be `uint<8>` or `uint<9>` depending on overflow policy).
- Need to decide: does the language support width inference, and if so, what are the rules?

### No trait / interface constraints

- No way to express that a type must implement a particular interface (e.g., `Add`, `Eq`).
- Currently `hir.binary` takes a generic binary kind attribute but has no mechanism to check that the operand types support that operation.

## Design Questions

### Unification vs. constraint solving

- The current approach is direct unification (Hindley-Milner style, no constraint generation + solving phase).
- Is this sufficient for Silicon's goals, or do we need a more general constraint solver (e.g., for subtyping, width inference, or trait bounds)?
- If we need constraints beyond equality, what does the constraint language look like?

### Where does inference run in the pipeline?

- Currently: codegen → CheckCalls → InferTypes → SplitPhases → PhaseEvalLoop.
- The design doc puts inference (Canonicalize + Unify) inside the phased evaluation loop, implying inference is re-run after each phase of execution.
- Is inference expected to produce new information after specialization? (e.g., a generic function gets specialized with concrete types, then its body needs re-inference.)
- The current InferTypes pass runs once before splitting; is that sufficient?

### Interaction between type inference and phase splitting

- Type constructors (`hir.int_type`, etc.) are `ConstantLike` and float to any phase.
- But what about computed types (e.g., `hir.uint_type %width` where `%width` is a runtime value)? These are pinned to the phase where `%width` is available.
- Need to specify: can type inference introduce cross-phase dependencies? What happens when a type can only be resolved in a later phase?

### Error provenance and diagnostics

- When unification fails, how do we trace back to the user's code?
- Unify ops don't carry information about _why_ the constraint exists (e.g., "because you passed X to parameter Y of function Z").
- Should unify ops carry source-level provenance attributes? Or should the error pass reconstruct this from the IR structure?

### `hir.coerce_type` semantics

- Currently a pure annotation erased during lowering.
- In the design doc, it appears at function entry to associate arguments with their declared types, and at call sites.
- Should it produce a runtime cast when the types are not identical?
- Should it emit a type error if the types are provably incompatible?
- Should it create a unify constraint implicitly?

### Unify op result semantics

- `hir.unify` produces a result with `SameOperandsAndResultType`.
- But since all HIR types are `!hir.any`, this trait is trivially satisfied; the _value-level_ equality is what matters.
- The result is used as the "resolved" type, but the semantics of which operand "wins" is left to the InferTypes pass.
- Should the unify op be purely a side-effect constraint (no result), or is having a result important for the IR structure?

### Replacing `hir.binary` with dedicated ops

- TODO.md mentions replacing `hir.binary` with dedicated `hir.add`, `hir.sub`, etc.
- This affects inference because each dedicated op could have different type rules (e.g., `hir.add` could widen, `hir.concat` doesn't unify its operand types).
- The current "unify both operand types" strategy in codegen would need per-op type rules.

### Value-dependent unification

- When a type depends on a value (e.g., `uint<N>`), unifying two such types requires unifying the width values.
- The current structural-equivalence check in InferTypes handles `hir.uint_type %a` vs `hir.uint_type %b` by recursively unifying `%a` with `%b`.
- But this only works for direct operands, not for derived values (e.g., `uint<a + 1>` vs `uint<b>` where `b = a + 1`).
- Do we need symbolic reasoning, or is operand-level structural matching sufficient?
