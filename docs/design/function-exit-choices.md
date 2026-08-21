# Early Exits: Normalization to Single-Exit Form

Extends {{< page-link "/design/sampling-model" >}} §6/§7 and composes with the split cf/reach record model of {{< page-link "/design/record-replay" >}}.

## The model

1. **Early exits are normalized away before phase analysis.**
   A phase-blind structural pre-pass rewrites every exit that is not its region's terminator into value form, bottom-up per region.
   It runs once per function region, on the body and the signature alike: `uir.signature` is the signature region's return-like exit and is normalized exactly like `uir.return`, except that a signature site has no type operands and so contributes nothing to unification.
   An exit-carrying divergence is rebuilt with extra trailing results — one slot per function result carrying the exit path's returned value, plus a payload-free tag choice (`#exit()` / `#thru`) — and a real pick on the tag dispatches: the exit alternative consumes the value slots directly, the single fall-through alternative holds the continuation.
   Slots a path does not provide are filled with `uir.unreachable_value`, never omitted, per the exact-arity yield/break invariant.
   The slot types are one shared `hir.inferrable` per exit slot, created at the region top — the function results for a body, the argument and result types for a signature: every rebuilt op uses them as its exit-slot types, every exit site's own type operands unify into them, and the final return is typed by them directly — floating, they satisfy the type operand rule's one-phase-earlier demand at any phase, where a pinned recovery like `hir.type_of` would manufacture backward crossings for earliest-phase values.
   One `uir.return` remains, as the body terminator.

2. **Every SSA value carries data of a single phase.**
   The exit *decision* and the exit *values* have different phases in general — the tag is knowable at the divergence's phase and gates same-phase continuation content, while a returned value may only materialize later — so they travel as separate values and are never fused into one.
   A payload-carrying exit choice would floor the tag to the payload's phase and destroy the earlier gate; splitting them is the same move that split the cf choice from the reach link in loop records.
   The value slots are ordinary CF results, so the standard floor, forwarding, and choice-ification rules deliver each at its own providers' phase, and per-slot phase diversity across one op is expressed slot by slot, never inside one value.

3. **Returns under loops become break-carried tags.**
   A `uir.return` inside a loop rewrites to a `uir.break` providing the loop's trailing exit-value slots and the tag (existing breaks gain the unreachable slots and the `#thru` tag), and the post-loop pick dispatches between function exit and ordinary completion.
   Nesting composes bottom-up: each region level provides its own slots and wraps its own tag, producing a linear chain of binary picks after each divergence, so continuations are never duplicated — not even for multi-arm divergences with mixed exit and fall-through arms, the case that makes naive if-else nesting explode.

4. **The pipeline order is normalize → analyze → sort.**
   Normalization consults no phase information, so it runs before PhaseAnalysis, and analysis input never contains early returns.
   The reverse order would invalidate the phase map mid-flight and reintroduce the incremental map maintenance that the pass split of {{< page-link "/design/phase-ops" >}} exists to avoid.

5. **After normalization there is no function-exit machinery.**
   The normalized body is ordinary value CF, so branch and loop extraction produce the cross-phase records, valued cf/reach picks, and value threading of the split record model without any exit-specific code.
   "Chain construction at return sites" is not a mechanism; it is what extraction already does to nested value CF.
   The function's later-phase result is a demanded later result like any other; its consumer happens to be the final `uir.return`.

6. **Receiving-phase replay is data-only everywhere.**
   Replayed content contains no exit terminators, for loop bodies and function bodies alike: recorded exit paths gate content through cf and reach picks, and exit values thread up through the picks' yields.
   Consequently a `uir.phase` region has exactly one `uir.phase_return`, and the splitter's multi-return plumbing disappears.

7. **The D-rule is not a rule.**
   Earlier phases are blind to later-phase exits because earlier-phase content inside a later-phase pick alternative runs unconditionally at its own phase — the same treatment any earlier-phase content inside later-phase CF receives under the sampling model.
   Result-phase errors (a result provided by several exit paths cannot deliver earlier than the deciding phase) come from the general CF-result floor rule applied to the pick, not from a return-specific check; PhaseAnalysis's multi-return collection goes dead once its input is single-exit.

8. **Totality is source-level.**
   Definite-return checking happens at the language level, before codegen; normalization preserves whatever the source guarantees.
   A body whose paths all exit normalizes to a pick whose fall-through alternative terminates in `uir.unreachable`; sorting must accordingly treat unreachable-terminated alternatives as ordinary non-providing arms.

9. **Choices and picks are executable single-phase CF.**
   Choice construction and pick dispatch are legal input to phase analysis and sorting, not just sorted-output vocabulary: analysis models a pick exactly like a `uir.if` whose condition is the selector (selector frame-consumed, arms yield-terminated, results under the standard floor), and a choice as a pure region-free constructor.
   This is required independently: a receiving phase's picks are the next boundary's producing CF, so re-extraction at phase p+2 needs them analyzable anyway.

## Example

```
// if d { dyn { print("a") } }            -- rejoins
// if e { return dyn { print("b"); 1 } }  -- exits
// dyn { print("c") }; return dyn { 2 }
```

Normalization touches only the exit (the first `if` has no return and stays); the rebuilt `if` gains a value slot and a payload-free tag:

```
uir.if %d { dyn { print("a") } }
%st = hir.inferrable
%if:2 = uir.if %e : %st, %ct {
  %v = dyn { print("b"); 1 }
  hir.unify %st, <type-of-%v>
  yield %v, #exit() : %st, %ct
} else {
  yield unreachable_value, #thru() : %st, %ct
}
%r = pick %if#1 [
  #exit(): yield %if#0
  #thru(): { dyn { print("c") }; yield dyn { 2 } }
]
uir.return %r : %st
```

This is plain value CF, and every value has a single phase: the tag `%if#1` stays at phase 0 (where `%e` decides), the value slot `%if#0` floors to phase 1 (its provider), the pick anchors at the tag's phase 0 with only its result floored to 1, and the return's type operand is the floating `%st`.
The existing branch extraction then lifts the phase-1 arm content out with a cf choice, yielding the split-model records, and choice-ifies the phase-1 value slot; the phase-1 function replays picks in program order and returns from its single `uir.phase_return`.

## Why exits are not kept in the IR

The alternative is to carry early exits through sorting and discover them at distribution time, building a nested choice chain at each return site from the path's statically-known dispositions and letting receiving phases hold several `uir.return`s.
It loses on every count against the split cf/reach model of {{< page-link "/design/record-replay" >}}:

- A chain that dispatches on exit dispositions first duplicates the replay of a *preceding* rejoining divergence into every chain alternative — the same cloning that rules out fused loop records.
- Multi-returns in receiving phases require non-local exits (a `return` inside a pick alternative, or inside a replicate body for returns under loops), reintroducing exactly the terminator-scoping complexity the split model eliminates for break/continue.
- With returns in replay, an exit divergence's reach slot is structurally present but never read on the exit path — a latent unread-slot invariant nothing machine-checks.
- Value threading through picks is machinery that has to exist anyway; multi-returns are an extra rule on top of it.

Normalizing exits to value form *before* analysis makes every downstream stage single-exit and dissolves the exit machinery into the general extraction rules.
