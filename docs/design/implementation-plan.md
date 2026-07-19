# Implementation Plan: Sampling Model (2026-07-10)

Companion to `sampling-model.md` and `function-exit-choices.md`. Maps the model onto the existing codebase.
Ground truth from the code inventory: the new UIR stack (`uir.func`, PhaseAnalysis, SplitPhases2) is silicon-opt-only; codegen still targets the old HIR stack; assignment and loops do not exist in the language surface at all; the interpreter cannot evaluate opaque lists.
Two later revisions apply: the merged-pass architecture of stages 5–6 was replaced on 2026-07-11 by the explicit phase-op contract of `phase-ops.md`, and the early-exit mechanism on 2026-07-19 by the exit normalization of the revised `function-exit-choices.md`; the affected statements below are corrected in place.

## Op changes

**New (HIR dialect):**
- `hir.choice_type`: constant type-constructor op mirroring `hir.opaque_type`.
  Choice values stay `!hir.any` at the HIR/UIR level; `hir.choice_type` provides the type operand for rebuilt CF-op choice results and their yields, and `!si.choice` materializes at MIR lowering, exactly like opaque.
- MIR twins for choice ops (construction op + constant attr) so the interpreter can produce them, and MIR twins for var ops (`mir.var`/`read_var`/`write_var`) plus interpreter mutable-cell support: compile-time phases contain vars (replicate bodies communicate through them), so they must be executable long before mem2reg exists.
  Interpreter reference values are `(cell, home)` pairs; the interpreter enforces write@home, read@home-or-later, and errors on deref of not-yet-materialized cells (sampling-model §11).
  HIRToMIR conversion patterns for all new ops (var, choice construction; pick/replicate never lower — they are expanded by SpecializeFuncs; unexpanded ones reaching HIRToMIR are a hard error).

**Existing (from M0):** `hir.var`/`read_var`/`write_var`, `hir.choice`/`hir.pick`, `hir.replicate` (no threading), `hir.opaque_list_create`/`push`, opaque pack/unpack.

**Changed:**
- `hir.pick` verifier: alternatives may terminate in `uir.return` instead of `hir.yield` (exit paths); pick results simply never materialize on such paths.
  Same for `hir.replicate` bodies.
- `uir.if`/`uir.loop`: rebuilt by the sorting stage with an extra choice result whose type operand is an `hir.choice_type` value; no verifier exemptions needed — the choice result is an ordinary result with an ordinary type operand.

## Pass architecture

Pipeline: codegen → CheckCalls/InferTypes/CheckTypes → **SplitPhases** (one merged pass) → FlattenCF → PhaseEvalLoop (HIRToMIR + Interpret + SpecializeFuncs + **FlattenCF**, extended) → ...

There is no separate SortPhases pass: sorting invalidates the phase analysis it consumes (dissolved exprs/pins cannot be re-analyzed), so one pass owns the analysis and every transformation that depends on it, and the phase map never crosses a pass boundary.
The pass processes functions in topological order (callees before callers; call cycles are an error) and maintains the phase map incrementally for every op it creates.

Per function, the merged pass runs these stages:

1. **Analyze**: run PhaseAnalysis (unchanged from M1).
2. **Dissolve**: inline `uir.expr` bodies and replace `uir.pin` outputs with inputs.
   The phase map is the single source of truth from here on; extraction never has to move content across expr region boundaries or rewire expr yields.
3. **Decompose calls**: split each `uir.call` into per-phase `hir.call`s using the callee's `uir.split_func` witness (which exists — callees are processed first), assigning phases from the call's map entry plus arg/result offsets.
   Call halves are ordinary ops for the following stages; a callee earlier-half under later-phase CF hoists like any other earlier-phase op.
4. **Sort**: per boundary p (earliest first), hoist then extract:
   - **Hoist up, to a fixpoint**: ops at phase ≤ p transitively inside regions of CF ops at phase > p move, in program order, to immediately before the (outermost such) enclosing CF op.
     Extraction below asserts its regions contain no earlier-phase content (extract-first changes multiplicity of earlier-phase ops under later CF — a D-rule violation; the orders are not confluent).
   - **Extract down**, per CF region at phase p with any later-phase content, innermost first:
     - Loop regions get one hit list created before the op; each region exit packs the activation's record (`opaque_pack` + `opaque_list_push` before the terminator), with exit-path choices as trailing record fields for partial activations.
     - Branch regions get per-arm `hir.choice` construction yielded as an extra CF-op result (rebuild the op; materialize the else region of an else-less `if` for the not-taken tag); arms with later-phase content that fall back out yield a sub-choice covering the arm-local divergence.
     - One `hir.replicate`/`hir.pick` after the CF op receives all of the region's later-phase content in program order; record fields and choice payloads become block args.
     - Replicates/picks count as CF regions for the next boundary; content more than one phase later rides along and extracts again.
     - Delete replicates/picks only when all bodies/alternatives are empty, along with their lists/choices (materialize-then-DCE).
   - Function-level `uir.return`s never reach this stage: exit normalization (`function-exit-choices.md`, revised 2026-07-19) rewrites them into single-exit choice/pick value form before phase analysis runs.
   - Anything the stage cannot handle is a hard error, never a silent skip.
5. **Distribute**: create per-phase `hir.func` ops and distribute straight-line ops by phase map (whole CF ops move as units — after sorting they are single-phase).
   Function-level early exits do not materialize here: exit normalization removes them before analysis (`function-exit-choices.md`, revised 2026-07-19), every phase is single-exit, and exit values arrive threaded through ordinary picks.
   The floor rule survives in general form: a result provided by several exit paths cannot deliver earlier than the deciding phase, emitted as the CF-result floor error on the pick (the same-SSA-value exemption stays dropped, sampling-model.md §7 as revised).
6. **Signatures and witness**: reconstruct per-phase signatures, emit `uir.split_func` and `hir.multiphase_func` structural ops.
   Prefer threading return/signature scaffolding through explicit IR over side tables of tracked `Value`s that must be manually RAUW'd (a known fragility of the current implementation).

A `sort-only` pass option stops after stage 4 so lit tests can check the sorted unified IR in isolation.

FlattenCF skips replicate/pick regions (their bodies keep structured CF — replicate/pick are single-block; lowering nested `uir.if` to `cf.cond_br` inside them would break that).
Expansion in SpecializeFuncs re-introduces structured CF into flat functions, so FlattenCF joins PhaseEvalLoop's per-iteration sub-pipeline after SpecializeFuncs.

**PhaseEvalLoop extensions:**
- Interpret: add `mir.opaque_list_create`/`push` evaluation (heap-backed list in the interpreter) and choice values.
- SpecializeFuncs: add replicate/pick expansion — for each `replicate` whose hit list is a constant: stamp the body per record, substituting record constants; for each `pick` with constant choice: inline the tagged alternative (a pick whose choice was never produced is an internal hard error; code after an inlined `return` is dead and folded).
  Worklist alongside existing call specialization (transitive: expansion reveals calls and nested replicates).
  Extend the specialization cache key with the phase offset (§9).
- PhaseEvalLoop's progress/blocked checks must know about unexpanded replicates/picks.

**Later:** mem2reg (vars → SSA, preserving sample positions), CFG-to-dataflow, diagnostics for surprising D-rule programs.

## Milestones (each lands with lit tests; check-silicon green throughout)

- **M0 — ops.** Done (var/read_var/write_var, choice/pick, replicate without threading, list ops; `hir.store` deleted).
- **M1 — PhaseAnalysis.** Done 2026-07-10 (var ops anchored via the generic side-effecting path, choice-ification obligations recorded in the phase map, causality diagnostics, never/bottom phase, floating-frame block-phase default).
  The block-phase default was generalized 2026-07-15 from the original effectful-only fallback: every undemanded floating `uir.expr` frame defaults to its block phase, rising only for the frame-consumed bounds its anchored content imposes (conditions, inits, continue values, effects, out-targeting exits, type operands).
  The frame's control-flow skeleton anchors at that phase while its pure interior distributes per op, so a dead frame resolves exactly like its live twin; only region-free pure ops and constants still float.
- **M2 — sorting stage.** `hir.choice_type`; stages 1–4 of the merged pass (dissolve, call decomposition, hoist, extract) behind a `sort-only` option; pick/replicate verifier updates.
  Lit tests, one per model scenario: sum-over-index, sibling regions with interleaving, distance-2 chaining, nested loops, if/match choice (valued and value-less, else-less), break-value choice, filtering (if-const in for-const), partial activation (continue/break mid-region, incl. multi-exit chain), later-phase return under earlier CF (return inside replicate), hoisting (const write and call-earlier-half under dyn if; hoist-before-extract ordering, e.g. const increment under dyn if inside for-const), all-empty deletion, error cases.
- **M3 — distribution.** Stages 5–6 rewritten on sorted IR: straight-line distribution, signatures, witness; delete the old cross-phase fixup machinery that sorting obsoletes.
  Exit chains dropped out of this milestone: exit normalization (`function-exit-choices.md`, revised 2026-07-19) handles early returns before analysis.
  Tests: normalized early-exit functions (rejoining selections, all-paths-exit bodies), phase −1 exit-freedom, forwarding middle phases, plus the existing split coverage rebased onto the merged pass.
- **M4 — evaluation.** Interpret list/choice support, SpecializeFuncs expansion; full silicon-opt pipeline tests from unified UIR to fully specialized final phase (zero-record passthrough, accumulation, nested expansion, per-record call specialization).
  Reference cells as `(cell, home)` pairs with eval-time home checks; references surviving into residual programs materialize as constants naming later-phase cells.
- **M5 — language surface.** Lexer/parser/AST/codegen: assignment statements, `for`/`while`/`loop`, `var` (or `let mut`), codegen to UIR with vars.
  **Definite-return checking** (guarantees trailing returns; makes zero-activation replicates fall through safely and never-produced choices unreachable).
  End-to-end .si tests; commit the design-doc test-case programs.
- **M6 — docs + demolition.** Rewrite control-flow.md / unified-dialect.md / phase-inference.md to the sampling model; fold `function-exit-choices.md` into `sampling-model.md`.
  Delete the old HIR stack (old SplitPhases, TestPhaseAnalysis, `hir.unified_func`/`unified_call`/`split_func`/`expr`/`yield`).

## Design-doc corrections to fold in at M6

- control-flow.md "phase -1 stops collecting at the return" is wrong: hits are not truncated; dead copies are folded after expansion (D-rule).
- control-flow.md nested-hits section: keep the end state, replace the mechanism with §6's per-boundary rule.
- unified-dialect.md / phase-inference.md: structured CF survives splitting *only when single-phase*; phase-crossing CF is consumed by the sorting stage.
  Transparent-yield futures across phases are replaced by choice-ification; document what remains of yields (same-phase only).
- phase-splits.md: opaque context = the straight-line record; hit lists/choices are additional record fields, not a parallel mechanism; per-return contexts are path-shaped (`function-exit-choices.md`).
