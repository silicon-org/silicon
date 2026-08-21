# Phase Ops: An IR Contract Between Sorting and Splitting

The pass architecture that implements {{< page-link "/design/sampling-model" >}}: how a cross-phase `uir.func` becomes one `hir.func` per phase, and what IR the stages hand each other.
It supersedes the merged single-pass plan in {{< page-link "/design/implementation-plan" >}} (its stages 5–6 and the topological-order driver); the sorting algorithm itself — dissolve, call decomposition, hoist, extract — carries over unchanged.

## Why the stages talk through IR

A merged sort-and-split pass has no expressible interface between "sorted" and "split": it lives in PhaseAnalysis side tables, in manually tracked `Value` handles that must be patched at every RAUW, and in `uir.split_func` witnesses that sorting would need before it can decompose a caller's calls.
That witness dependency forces callee-before-caller topological processing and structurally rejects recursion, both stages end up duplicating machinery (call decomposition, expr/pin dissolution), the interface cannot be lit-tested, and the splitter has to re-consult phase information everywhere.
Naming the intermediate state in IR — `uir.phase`, `uir.phase_return`, `uir.phase_call` — removes all of it.

## The design

Five stages, each owning a single concern:

1. **Exit normalization** rewrites early `uir.return`s into choice/pick value form, bottom-up per region, leaving one `uir.return` as the body terminator ({{< page-link "/design/function-exit-choices" >}}).
   It is phase-blind and runs before PhaseAnalysis, so analysis and sorting input is always single-exit.
2. **PhaseAnalysis** (unchanged) computes phases for ops and values.
   It is consumed only by the sorting pass; the phase map never crosses a pass boundary.
3. **Phase sorting** rewrites the internals of a `uir.func`.
   Before touching a function, it snapshots the declaration: it creates the `uir.split_func` under the function's original name, holding the original unsorted signature region verbatim and an empty link list, moves the implementation func to a fresh uniquified name, and records that name in an implementation-reference attribute on the split_func.
   Existing `uir.call`/`uir.phase_call` references keep resolving, since the split_func is a legal callee for both.
   Output contract: the body and signature regions contain only `uir.phase` ops, floating ops, and the region's terminator (`uir.unreachable` for the body, `uir.signature` for the signature), at most one `uir.phase` per phase, in increasing phase order.
   Values computed inside a phase and consumed in a later one cross as phase-op results: yielded by `uir.phase_return`, consumed by any later phase, at any distance.
   Values already visible at the region top level — function arguments and floating ops — are consumed directly wherever needed, without routing through a phase op.
   Calls are decomposed into `uir.phase_call` ops targeting a specific phase of the callee.
   Hoisting and single-provider forwarding run to a joint fixpoint first, since each can expose work for the other; branch and record extraction then run to a joint fixpoint of their own, so a rule that emits a receiving pick or replicate re-extracts it at the next boundary and a crossing of any width sorts in one sweep.
   Both fixpoints cover the body and the signature region alike.
   This is the only pass that runs or consumes PhaseAnalysis.
4. **Threading normalization** is a cheap structural pre-pass, run inside the function splitter or as a small pass immediately before it.
   It brings the sorted form into a fully threaded shape: it materializes `uir.phase` ops for externally visible phases that have no content and for intermediate hops of skipping crossings, threads every crossing through adjacent phase ops (function arguments starting at their declared phase), and sinks floating ops into the phase regions that use them.
   It delivers each `uir.signature` operand as a result of the phase op one before the argument or result it ascribes, so the signature region's phase range is the declared span shifted one phase earlier.
   Its output contract is purely structural, with no notion of "floating" left: regions hold exactly `uir.phase` ops and the region terminator, every value crossing between phase ops is yielded by the immediately preceding one, and every function result rides the `uir.phase_return` of its phase.
5. **Function splitting** is phase-blind: it never asks what phase an op has.
   It expects the normalized contract, creates one per-phase `hir.func` per phase present in either region (never synthesizing any) whose body is the signature region's content for that phase followed by the body region's, converts phase-op result flow into opaque context pack/unpack, rewrites `uir.phase_call` to `hir.call` against the callee split_func's link list, reconstructs per-phase signatures, fills the split_func's links (all link lists before lowering any call), emits `hir.eager_func` for pre-span chains, and deletes the implementation func and reference.
   It also diagnoses `uir.phase` ops later than the declared span as a user error (runtime work past the last declared phase has no executor), which catches call-implied late phases uniformly since calls are already decomposed into `uir.phase_call` entries by this point.

## New ops

- `uir.phase`: a region op holding one phase's content, letting multiple per-phase function bodies coexist inside one `uir.func`.
  Its results are the values consumed by later phases.
  The function's original `uir.return` is replaced by a `uir.unreachable`.
- `uir.phase_return`: terminator of a `uir.phase` region.
  It carries both the function results belonging to that phase and the values yielded as the phase op's results.
  Exactly one `uir.phase_return` terminates each phase region: early exits are normalized to value form before analysis, so exit values arrive threaded through picks rather than on extra returns.
- `uir.phase_call`: calls one specific phase of a `uir.split_func`, with the callee's args and results filtered down to that phase, plus an opaque context value chaining consecutive phase calls.
  The caller-visible phase entries are one per phase in the callee's *declared span* — the range from the earliest to the latest declared argument/result phase, always including phase 0 — with pass-through entries for phases that carry no arguments or results.
  The span is derivable from the callee's declaration alone: no split witness is needed at sort time, so there is no callee-before-caller processing order, and recursion is no longer structurally rejected.

## Design decisions

- **Phase span and internal phases.**
  A function's phase span is the range from its earliest to its latest declared argument/result phase, always including phase 0.
  Callers thread one `uir.phase_call` entry per phase in the span, including pass-through entries for phases without arguments or results, so mid-span internal phases execute lock-step in the caller's timeline and are never hidden behind a callee-side vehicle.
  Content *later* than the span is a user error: runtime work past the last declared phase has no executor, and the fix is declaring it (e.g. a `dyn` result or effect marker).
  Content *earlier* than the span is legal: it is argument-free by construction (no declared value exists that early), prefixes into the earliest span entry, and is eagerly evaluated at definition time.
  The asymmetry is deliberate: compile-time work can always be pulled earlier to definition time, later runtime work cannot.
  Sorting itself stays span-agnostic: it distributes content into a `uir.phase` op for every phase with content, and the span error and eager absorption are splitting/evaluation concerns.
- **Eager evaluation is explicit policy, never inferred from zero-argument-ness.**
  A zero-argument phase-0 function is runtime code; the compiler is not entitled to interpret it.
  Pre-span phases are compile-time by construction, and `hir.eager_func` — the successor of `hir.multiphase_func` — is the op that carries the eagerness policy: `hir.eager_func @f(args) -> (results) [@pre0, ..., @tail]`, where the head link is always zero-argument, consecutive links chain an opaque context, and the tail takes the earliest span entry's declared arguments.
  The op's external signature is exactly the earliest span entry's signature, so `uir.split_func` entries and `hir.call`s target it like any function; the `first`/`last` argument split of `hir.multiphase_func` disappears.
  The evaluation driver keys on the op's existence: it evaluates the head link when it becomes evaluable and specializes the next link forward; when a single link remains, the op is replaced by a same-name `hir.func` shim that tail-calls it, so no symbol use is ever rewritten.
  Expressing the eager chain as plain calls and SSA use-def chains instead of a dedicated op is deferred exploration.
- **`uir.split_func` is the stable late-binding interface.**
  Its purpose is to combine the post-splitting function symbols back into a single unsplit symbol that `uir.call` can target interchangeably with a `uir.func`; this is what enables library precompilation.
  Its signature region holds the original, unsorted signature — the type interface that call checking inlines at call sites, folding toward constants as eager evaluation publishes results; pre-span signature computation is additionally relocated into the eager chain's bodies as re-evaluable clones.
  The flat signature is preserved, never reconstructed: sorting creates the split_func before mutating the function (see stage 3), and the splitter only fills the links, so signature computations may contain calls without any de-phase-ification being possible or needed.
  The op has three verifier states: *declared* (no implementation reference, no links), *sorted* (implementation reference, no links), and *split* (links, no implementation reference); the declared state doubles as a natural "not yet compiled" form for incremental compilation.
  The old "decomposition witness" role is obsolete: call decomposition is declaration-derived and never consults the split_func.
- **Dependent types across the split are delivered by checker inlining.**
  Call checking runs before sorting and inlines the callee's signature computation into the caller — from the signature region of whichever of `uir.func` or `uir.split_func` the callee symbol resolves to, identically — and unifies the inlined types with the caller's.
  A split caller thus recomputes a dependent callee type from arguments it already holds and threads it through its *own* opaque context; the callee's context is private plumbing, never read for types, and the caller supplies call type operands independently.
  Agreement between the caller-side and callee-side type computation is structural: checking either rejects the program with a user diagnostic or guarantees both sides pick the same types.
  A type disagreement observed at evaluation time, when a per-phase func is specialized, is therefore an internal compiler error, and the evaluator must trap on it loudly.
  The `check-calls` pass implements this, between exit normalization and sorting: it copies the declared computation into every consumer — ahead of each `uir.call`, and into the declaring function's own body, where the arguments are additionally constrained by `hir.coerce_type` — and ties each copy to the local types with `hir.unify`.
  Signatures are copied whole, control flow included, and are therefore consumed in the single-exit shape exit normalization leaves behind; a signature with several blocks or exits has no single computation to copy and is rejected.
  Every signature region is checked alike, a `uir.split_func`'s as much as a `uir.func`'s: a declaration's signature is the interface every caller copies, so a call left unchecked in one lands unchecked in user code.
  Nothing in the IR marks a signature as checked, and checking is not idempotent: `hir.unify` folds only when both operands are the same value, so every re-check lengthens the unify chain on each call in the declaration, and a signature that itself calls is re-inlined whole.
  Where checked-ness lives is unsettled, and `uir.split_func` is precisely the form that gets re-checked on every downstream compile.
  Bodies belong to `uir.func` alone.
  Checking is ordered post-order over the call graph restricted to signature calls, so a callee's signature is resolved before it is copied; recursion through signatures is a user error.
- **Phase boundaries pack uniformly.**
  Every phase boundary packs exactly one fresh opaque context bundling all its crossings: no special case for a single crossing, for a crossing that is already an opaque context, or for a contentless gap phase (whose func unpacks and repacks unchanged).
  Packing must stay symmetric with the receiving phase's unpacking, so skipping a "redundant" pack is not a local decision.
  Collapsing degenerate transcodes is canonicalization — `OpaquePackOp::canonicalize` rewrites `opaque_pack(opaque_unpack(x))` to `x` — never splitter logic.
- **Per-phase function names are cosmetic.**
  Split_func link lists, eager_func chains, and the implementation-reference attribute are the authoritative connections; nothing resolves by name convention, and names may be uniquified freely.
  A link list is positional: entry *k* is the function implementing the *k*-th phase of the declared span, so an entry's meaning is a pure function of the declaration and the verifier can cross-check each entry's arity against the declared per-phase interface.
  Nothing on a per-phase `hir.func` says which phase of which declaration it implements — that would be a second source of truth — so a producer that emits two entries with matching arity in the wrong order is not caught.
- **No empty phases in sorted output.** Sorting creates a `uir.phase` op only for phases that have content; it never materializes empty ones.
  A `uir.phase_return` carrying function results or yields is content: a result-only phase gets a phase op holding just its `uir.phase_return`.
  Empty and passthrough-only phase ops are threading normalization's vocabulary: it materializes them freely for gap phases and threading hops, so the splitter emits exactly one function per phase op and never synthesizes.
- **Crossings in sorted output may jump.** Sorting yields a value once, from the phase computing it; consumers at any later phase use that result directly, and function arguments and floating ops are consumed directly at any phase.
  Hop-by-hop threading is threading normalization's job, not sorting's: no sorting rule depends on whether an intermediate phase op happens to exist.
  Floating ops are never assigned a phase by sorting: a floating pure value is a phase-compatible provider at any phase, so sorting delivers, forwards, and extracts it as-is — cloning its region-free pure tree to the use side where dominance demands it — and floats stay at the region top level, unmoved.
  Sinking floats into the phase regions that use them is exclusively threading normalization's job.
- **Body and signature regions follow one set of rules.** The same control flow and exit arrangements are legal in both; `uir.signature` is the signature region's return-like exit and may sit under control flow at any depth, with its position and arity checked by its own verifier.
  Exit normalization, PhaseAnalysis, and sorting therefore treat the two regions identically: normalization runs per region, a signature in a phase-shifted block is a user error exactly as a return or break would be, and sorting's hoist, forward, and extract walks cover both.
  Phase ops and their verifiers do not distinguish the regions, and sorting moves ops by phase without caring which region it rewrites.
  In particular, which function result values a `uir.phase_return` must carry is part of the splitting contract (diagnosed there), not an op verifier rule, since no single verifier rule covers both regions.
- **Signature region.**
  The signature undergoes the same phase sorting and contains its own `uir.phase` ops.
  A type ascribing a phase-P argument or result is phase P-1 content by definition: the per-phase function's MIR signature must be constructible after monomorphization and folding without evaluating anything, so a per-phase `hir.func` signature region can only unpack the incoming context and use `hir.opaque_type`; it never touches the function's own arguments.
  Contract after threading: every `uir.signature` operand typing a phase-P argument or result is a result of `uir.phase P-1`, without exception — floating ops (`hir.int_type` typing a phase -1 argument) sink into P-1, creating phase -2 if needed; leaving trivially foldable floats in the phase-P signature for post-specialization folding is a later optimization, not a rule.
  Phases before the declared span that arise this way are ordinary pre-span phases: arg-free by construction, evaluated eagerly via the `hir.eager_func` chain, after which the receiving signature's unpack folds and the per-phase function is self-describing.
  Splitting merges, never relocates: the phase-P function body is the signature region's phase-P content followed by the body region's phase-P content (either may be absent and counts as empty), its outgoing context is the union of both yields, and its signature region is built solely from unpacks of the slots the `uir.signature` operands name.
  The splitter performs no deduplication between the two regions: identical pure computations, calls included, are CSE's business, and the merge is a plain concatenation with a union context.
  A `uir.phase_call` whose callee phase differs from its enclosing phase is an offset call (offsets live on the enclosing block, not on the call) and lowers to the callee's phase-relative function like any other entry.
- **Module-ness lives on the declaration.**
  `isModule` is a declaration attribute: sorting's declaration snapshot copies it onto `uir.split_func`, and the single-link collapse shim inherits it, so after evaluation the consumer still finds it on the resulting function.
  A module's hardware is the link at the maximum phase of its span (a starting point; entry points and what exactly evaluates to hardware may refine this), resolved through the link list like any other callee resolution; the splitter never stamps per-phase functions and knows nothing about modules.
  A `dyn` module is therefore legal: its earlier links are type computations, its span-maximum link the hardware.
- **`uir.phase_call` targets only `uir.split_func`.**
  `uir.call` is the late-binding vocabulary and resolves to `uir.func` or `uir.split_func` interchangeably; sorting snapshots every declaration in the module before decomposing any call, so a decomposed entry always names the interface, whether the callee is fresh user code or already-split library code.
  A `uir.phase_call` naming a `uir.func` would bypass the interface and call an implementation directly, so the verifier rejects it.
- **Pass order.**
  Exit normalization, `check-calls`, canonicalize, CSE, canonicalize, `sort-phases`, `thread-phases`, `split-funcs`.
  The second canonicalization is not redundant: where a body computes its declared type itself, the two computations stay distinct values until CSE merges them, and only then is the `hir.unify` between them an identity the canonicalizer folds.
  The stage is load-bearing, not cosmetic: checking coerces every body argument to its declared type, so an `hir.type_of %x` the user wrote about an argument becomes `hir.type_of` of the coercion's result, a phase-P value whose type the sorter would have to place at phase P-1 — a backward crossing it rejects.
  Folding `type_of(coerce_type(_, %ty))` to `%ty` answers the query with the declared type the signature copy already computes, one phase earlier.
  The identity function `fn f(x: int) -> int { x }` is the smallest program that needs it, and the largest one this settles: the fold only reaches queries whose answer is recoverable in the same region, so `hir.type_of` in a branch arm or of a call result still reaches sorting, where a phase-P value's type demanded at phase P-1 stays unresolved.
  Sorting and splitting still run back-to-back: effect modeling and canonicalization interactions there (CSE across phase regions, constant hoisting out of them) are deferred until a canonicalizer needs to run in between.
- **The phase-op-only contract is not verifier-enforced.**
  The function-splitting pass errors out with a diagnostic when it encounters content that violates the expectation.
- **Function-level exits** ({{< page-link "/design/function-exit-choices" >}}) never reach sorting: exit normalization rewrites them to value form up front, so there are no exit chains to surface, and the splitter sees exactly one `uir.phase_return` per phase region.
  Choice construction and pick dispatch are consequently legal *input* vocabulary for analysis and sorting (a pick is modeled like a `uir.if` whose condition is the selector), not just sorted-output vocabulary.

## What this dissolves, what it does not

Dissolved: the cross-phase fixup machinery (`fixupCrossPhaseRefs`, hop-by-hop value mappings), the tracked-`Value` handle patching at every RAUW site, the duplicated call decomposition, the topological order and its cycle error, and the `sort-only` test flag (the sorted form is now a first-class, verifiable IR that lit tests check directly).

Not dissolved: the extraction machinery (hoisting, branch/loop extraction, records, break-value choices, partial activations) is intrinsic to the semantics and ported nearly verbatim from the old sorter; only its packaging changes (`uir.phase_call` instead of witness-based calls, output wrapped in `uir.phase` ops, `uir.phase_return` terminators).

## Remaining migration

`silc` still runs the separate, older HIR-level split (`hir.split_func`, `SplitPhases`, `PhaseEvalLoop`); it switches over to this pipeline at parity, after which the HIR-level stack is deleted.
Evaluation is not wired up yet: nothing drives `hir.eager_func` chains or collapses a single-link chain into its shim.
