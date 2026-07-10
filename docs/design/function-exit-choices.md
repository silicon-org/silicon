# Early Exits Across Phase Boundaries

Result of the design interview on function-level early returns (2026-07-10).
Extends {{< page-link "/design/sampling-model" >}} §6/§7, into which it will be folded; the implementation plan's milestones change accordingly.

## The model

1. **Receiving phases inherit no control flow.**
   Producing-phase CF (ifs, loops, returns, breaks) is never replicated in later phases.
   What crosses a boundary is only data: records (per loop activation) and choices (per branch path).
   The receiving phase consumes them with `replicate`/`pick` ops placed at the producing CF op's position, with the continuation following *in program order* after the pick — never nested into alternatives.
   Nesting the continuation would duplicate it when a branch point has several rejoining arms, e.g. a match with mixed terminating and falling-through arms.

2. **Chains are constructed at the exits, not at the branch points.**
   Each `return` site builds the boundary's nested choice bottom-up, in place: every branch disposition *along its own path* is statically known there, and every path-visible crossing value is in scope.
   Nothing is constructed early, and nothing needs undef/poison filler.
   The per-return context pack is `(shared-prefix values…, chain)`; the flat straight-line context is the degenerate single-return case.

3. **Arm-local divergence is the only case needing accumulation.**
   When an arm contains later-phase content and falls back out (its samples leave scope), the `if` yields a *sub-choice* covering just that arm-local divergence (`ran(samples…)` / `not_ran()`); downstream returns embed that SSA value as a payload field in the chains they build.
   This is the same device extraction already uses for in-body picks.

4. **Pick alternatives yield iff the selection rejoins, and return iff the path exits.**
   Multiple returns are legal in every phase; there is no yield-plus-single-return normalization (it is never *necessary*, so it is never done).
   A `return` inside a pick alternative terminates the receiving phase's function; pick results on exit paths never materialize and their consumers are unreachable.
   Rejoining selections must yield — that is how the selected value reaches the shared continuation.

   ```
   // if d { let a=..; dyn{print"a"}; return dyn{a} }
   // if e { let b=..; dyn{print"b"}; return dyn{b} }
   // let c=..; dyn{print"c"}; return dyn{c}
   //
   // phase 0 (CF untouched, each return packs its own path):
   if d { …; return choice 0(a) }
   if e { …; return choice 1(choice 0(b)) }
   …; return choice 1(choice 1(c))
   // phase 1 (no CF mirroring):
   %v = pick ch [
     0(a): { print("a"); yield a }
     1(rest): {
       %w = pick rest [ 0(b): { print("b"); yield b }
                        1(c): { print("c"); yield c } ]
       yield %w
     }
   ]
   return %v
   ```

5. **Totality is a cheap structural scan, not a local IR property.**
   A function body must end in `return` or `unreachable`; any other returns are just additional exits.
   That some return always fires is inherited from the producing phase (guarded by source-level definite-return checking); it is not re-provable per split function, and that is accepted.

6. **Returns under producing-phase loops ride the records.**
   Each region's exit structure travels on that region's own boundary object: hit-list records for loop iterations (the `return` sits inside the replicate's record-pick and, if it fires, terminates the receiving phase), the context channel for the function body.
   Partial activations put exit-path choices inside records, per the sampling model §6.

7. **The machinery is strictly forward-flowing.**
   Earlier phases never receive a choice.
   Phase −1 is blind to phase-0 exits (D-rule): all its content runs unconditionally, and its function is exit-free and straight-line.
   Earlier-phase result operands on multi-returns are either the same SSA value (delivered a phase earlier, dropped from the later return) or a floor-rule error (PhaseAnalysis step 3b, already implemented).

8. **Returns never resolve cross-phase selections.**
   A const-condition `if` stays anchored at its block phase; returning distinct earlier-phase values under it is rejected by the floor rule even when the selection is semantically decidable earlier (conservative, per the sampling model §7's no-late-errors stance).
   The escape hatch is a value-if at the deciding phase: `let const r = const { if p { j2 } else { j } }`.

## Related decisions from the same session

- **SortPhases merges into SplitPhases**: one pass owns PhaseAnalysis, expr/pin dissolution, call decomposition, sorting (hoist + extract), and distribution — the phase map never crosses a pass boundary and is maintained incrementally for created ops.
- **Dissolve `uir.expr`/`uir.pin` right after analysis**, before sorting.
- **Decompose `uir.call` into per-phase calls before sorting** (the callee's split witness exists — callees are processed before callers), so call halves are ordinary ops for hoisting and extraction.
- **Choices stay `!hir.any`** at the HIR/UIR level; a new `hir.choice_type` constant op (mirroring `hir.opaque_type`) provides the type operand for rebuilt CF results and yields; `!si.choice` materializes at MIR, like opaque does.
- **Function-level picks and choices are born at distribution time** (choice construction at returns, pick in the next-phase function); they need no unified-IR representation.
