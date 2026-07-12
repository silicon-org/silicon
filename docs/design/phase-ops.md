# Phase Ops: An IR Contract Between Sorting and Splitting

Result of the design discussion on pass complexity (2026-07-11).
Revises the pass architecture in {{< page-link "/design/implementation-plan" >}}: the merged single SplitPhases pass (its stages 5–6 and the topological-order driver) is replaced by two passes connected through explicit IR.
The sorting algorithm itself (stages 1–4: dissolve, call decomposition, hoist, extract) is unchanged and carries over.

## Problem

The current implementation (`SortPhases.cpp` + `SplitPhases2.cpp`) tangles several concerns in each stage, and the interface between them is invisible: it lives in the PhaseAnalysis side tables, in manually tracked `Value` handles that must be patched at every RAUW, and in `uir.split_func` witnesses that sorting needs before it can decompose a caller's calls.
The witness dependency forces callee-before-caller topological processing and structurally rejects recursion.
Because the intermediate state between "sorted" and "split" cannot be expressed in IR, both stages duplicate machinery (call decomposition, expr/pin dissolution), the interface cannot be lit-tested, and the splitter must re-consult phase information everywhere.

## The design

Four stages, each owning a single concern:

1. **PhaseAnalysis** (unchanged) computes phases for ops and values.
   It is consumed only by the sorting pass; the phase map never crosses a pass boundary.
2. **Phase sorting** rewrites the *internals* of a `uir.func` and leaves the function op untouched.
   Output contract: the body and signature regions contain only `uir.phase` ops, floating ops, and the region's terminator (`uir.unreachable` for the body, `uir.signature` for the signature), at most one `uir.phase` per phase, in increasing phase order.
   Values computed inside a phase and consumed in a later one cross as phase-op results: yielded by `uir.phase_return`, consumed by any later phase, at any distance.
   Values already visible at the region top level — function arguments and floating ops — are consumed directly wherever needed, without routing through a phase op.
   Calls are decomposed into `uir.phase_call` ops targeting a specific phase of the callee.
   This is the only pass that runs or consumes PhaseAnalysis.
3. **Normalization** is a cheap structural pre-pass, run inside the function splitter or as a small pass immediately before it.
   It brings the sorted form into a fully threaded shape: it materializes `uir.phase` ops for externally visible phases that have no content and for intermediate hops of skipping crossings, threads every crossing through adjacent phase ops (function arguments starting at their declared phase), and sinks floating ops into the phase regions that use them.
   Its output contract is purely structural, with no notion of "floating" left: regions hold exactly `uir.phase` ops and the region terminator, every value crossing between phase ops is yielded by the immediately preceding one, and every function result rides the `uir.phase_return` of its phase.
4. **Function splitting** is phase-blind: it never asks what phase an op has.
   It expects the normalized contract, creates one per-phase `hir.func` per `uir.phase` op (never synthesizing any), converts phase-op result flow into opaque context pack/unpack, rewrites `uir.phase_call` to `hir.call`, reconstructs signatures, and emits the structural ops (`uir.split_func`, `hir.multiphase_func`).

## New ops

- `uir.phase`: a region op holding one phase's content, letting multiple per-phase function bodies coexist inside one `uir.func`.
  Its results are the values consumed by later phases.
  The function's original `uir.return` is replaced by a `uir.unreachable`.
- `uir.phase_return`: terminator of a `uir.phase` region.
  It carries both the function results belonging to that phase and the values yielded as the phase op's results.
  Multiple `uir.phase_return`s per region are legal (early exits); all must agree on the yielded signature.
- `uir.phase_call`: calls one specific phase of a `uir.func`, with the callee's args and results filtered down to that phase, plus an opaque context value chaining consecutive phase calls.
  The caller-visible phase entries are exactly the callee's external phases, which are derivable from the callee's *declared* arg/result phases (plus phase 0).
  No split witness is needed at sort time, so there is no callee-before-caller processing order, and recursion is no longer structurally rejected.

## Settled decisions

- **Internal phases.** Sorting distributes content into a `uir.phase` op for every phase, including phases not visible in the function's declaration — leading, mid-range, and trailing internal phases, such as effect-only `dyn` content past the last declared phase.
  Callers cannot target internal phases: the `uir.phase_call` entries are exactly the declaration-derived external phases.
  How internal phases end up executing is a function-splitting and evaluation concern (multiphase funcs are the likely vehicle), not a sorting concern; sorting stoically wraps them like any other phase.
- **No empty phases in sorted output.** Sorting creates a `uir.phase` op only for phases that have content; it never materializes empty ones.
  A `uir.phase_return` carrying function results or yields is content: a result-only phase gets a phase op holding just its `uir.phase_return`.
  Empty and passthrough-only phase ops are normalization's vocabulary: it materializes them freely for gap phases and threading hops, so the splitter emits exactly one function per phase op and never synthesizes.
- **Crossings in sorted output may jump.** Sorting yields a value once, from the phase computing it; consumers at any later phase use that result directly, and function arguments and floating ops are consumed directly at any phase.
  Hop-by-hop threading is normalization's job, not sorting's: no sorting rule depends on whether an intermediate phase op happens to exist.
- **Body and signature regions follow one set of rules.** The same control flow and exit arrangements are legal in both, phase ops and their verifiers do not distinguish the regions, and sorting looks at op phases and moves ops around without caring which region it rewrites.
  In particular, which function result values a `uir.phase_return` must carry is part of the splitting contract (diagnosed there), not an op verifier rule, since no single verifier rule covers both regions.
- **Signature region.** The signature undergoes the same phase sorting and contains its own `uir.phase` ops.
  At function-splitting time, values the signature carries between phases must be computed in the function *body* of the previous phase (so they can be passed into the next phase call as specialization parameters), not in the signature.
  That relocation is purely a function-splitting concern; PhaseAnalysis and sorting do not know about it.
  Consequence: signature computations (including `uir.phase_call`s) must be re-evaluable, since they are cloned between signature and body — already the status quo assumption, now stated.
- **No canonicalization between the passes** for now; sorting and splitting run back-to-back.
  Effect modeling and canonicalization interactions (CSE across phase regions, constant hoisting out of them) are deferred until a canonicalizer needs to run in between.
- **The phase-op-only contract is not verifier-enforced.**
  The function-splitting pass errors out with a diagnostic when it encounters content that violates the expectation.
- **Function-level exit chains** ({{< page-link "/design/function-exit-choices" >}}) surface on `uir.phase_return` instead of being discovered at distribution time.
  This localizes the chain per phase region and turns the splitter's part into ordinary result plumbing; the design is worked up incrementally after the core port.

## What this dissolves, what it does not

Dissolved: the cross-phase fixup machinery (`fixupCrossPhaseRefs`, hop-by-hop value mappings), the tracked-`Value` handle patching at every RAUW site, the duplicated call decomposition, the topological order and its cycle error, and the `sort-only` test flag (the sorted form is now a first-class, verifiable IR that lit tests check directly).

Not dissolved: the extraction machinery (hoisting, branch/loop extraction, records, break-value choices, partial activations) is intrinsic to the semantics and ports nearly verbatim from `SortPhases.cpp`; only its packaging changes (`uir.phase_call` instead of witness-based calls, output wrapped in `uir.phase` ops, `uir.phase_return` terminators).

## Migration strategy

Ops and hand-written target IR come first: the expected phase-op form of the canonical examples is written by hand before any pass code, forcing the `uir.phase_return` design decision while it is cheap, and doubling as the lit-test contract between the passes.
Sorting is then rebuilt case-by-case (one PR per case, porting from `SortPhases.cpp`), followed by the new exit-chain territory, and the phase-blind splitter last — sorting output is inspectable on its own, and looking at it feeds back into the earlier transform before the splitter exists.
The old `SortPhases`/`SplitPhases2` stack stays intact and green throughout as the porting source and behavioral reference; `silc` switches over at parity, and the old passes are deleted in one removal PR.
