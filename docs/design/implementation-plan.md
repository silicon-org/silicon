# Implementation Plan: Sampling Model

Companion to {{< page-link "/design/sampling-model" >}} and {{< page-link "/design/function-exit-choices" >}}: what the model costs in the existing codebase, and in which order the remaining pieces land.
{{< page-link "/design/phase-ops" >}} owns the architecture of the sorting and splitting stages and supersedes the merged single-pass plan this document originally carried.

Ground truth this plan works from: the `uir` stack (`uir.func`, PhaseAnalysis, exit normalization, sorting, threading, splitting) is silicon-opt-only; `silc` still drives the older HIR-level split; assignment and loops do not exist in the language surface at all; the interpreter cannot evaluate opaque lists.

## Op changes

**New (HIR dialect):**

- `hir.choice_type`: constant type-constructor op mirroring `hir.opaque_type`.
  Choice values stay `!hir.any` at the HIR/UIR level; `hir.choice_type` provides the type operand for rebuilt CF-op choice results and their yields, and `!si.choice` materializes at MIR lowering, exactly like opaque.
- MIR twins for choice ops (construction op + constant attr) so the interpreter can produce them, and MIR twins for var ops (`mir.var`/`read_var`/`write_var`) plus interpreter mutable-cell support: compile-time phases contain vars (replicate bodies communicate through them), so they must be executable long before mem2reg exists.
  Interpreter reference values are `(cell, home)` pairs; the interpreter enforces write@home, read@home-or-later, and errors on deref of not-yet-materialized cells (sampling-model §11).
  HIRToMIR conversion patterns for all new ops (var, choice construction; pick/replicate never lower — they are expanded by SpecializeFuncs; unexpanded ones reaching HIRToMIR are a hard error).

**Changed:**

- `hir.pick` verifier: alternatives may terminate in `uir.return` instead of `hir.yield` (exit paths); pick results simply never materialize on such paths.
  Same for `hir.replicate` bodies.
- `uir.if`/`uir.loop`: rebuilt by the sorting stage with an extra choice result whose type operand is an `hir.choice_type` value; no verifier exemptions needed — the choice result is an ordinary result with an ordinary type operand.

## Pass architecture

Pipeline: codegen → call/type checking → `normalize-exits` → `sort-phases` → `thread-phases` → `split-funcs` → FlattenCF → PhaseEvalLoop (HIRToMIR + Interpret + SpecializeFuncs + FlattenCF, extended) → MIRToCIRCT.

FlattenCF skips replicate/pick regions (their bodies keep structured CF — replicate/pick are single-block; lowering nested `uir.if` to `cf.cond_br` inside them would break that).
Expansion in SpecializeFuncs re-introduces structured CF into flat functions, so FlattenCF joins PhaseEvalLoop's per-iteration sub-pipeline after SpecializeFuncs.

**PhaseEvalLoop extensions:**

- Interpret: add `mir.opaque_list_create`/`push` evaluation (heap-backed list in the interpreter) and choice values.
- SpecializeFuncs: add replicate/pick expansion — for each `replicate` whose hit list is a constant: stamp the body per record, substituting record constants; for each `pick` with constant choice: inline the tagged alternative (a pick whose choice was never produced is an internal hard error; code after an inlined `return` is dead and folded).
  Worklist alongside existing call specialization (transitive: expansion reveals calls and nested replicates).
  Extend the specialization cache key with the phase offset (sampling-model §9).
- The driver keys eager evaluation on `hir.eager_func`: evaluate the head link when it becomes evaluable, specialize the next link forward, and replace the op with a same-name `hir.func` shim once a single link remains.
- Progress and blocked checks must know about unexpanded replicates and picks.

**Later:** mem2reg (vars → SSA, preserving sample positions), CFG-to-dataflow, diagnostics for surprising D-rule programs.

## Milestones

The ops, PhaseAnalysis, exit normalization, phase sorting, threading normalization, and the phase-blind function splitter are in place (M0–M3), each with lit tests.
What remains:

- **M4 — evaluation.** Interpret list/choice support, SpecializeFuncs expansion, the `hir.eager_func` driver; full silicon-opt pipeline tests from unified UIR to fully specialized final phase (zero-record passthrough, accumulation, nested expansion, per-record call specialization).
  Reference cells as `(cell, home)` pairs with eval-time home checks; references surviving into residual programs materialize as constants naming later-phase cells.
  Trap loudly on evaluating `uir.unreachable_value` during replay, and on a caller/callee type disagreement when specializing a per-phase func.
- **M5 — language surface.** Lexer/parser/AST/codegen: assignment statements, `for`/`while`/`loop`, `var` (or `let mut`), codegen to UIR with vars.
  **Definite-return checking** (guarantees trailing returns; makes zero-activation replicates fall through safely and never-produced choices unreachable).
  End-to-end .si tests; commit the design-doc test-case programs.
- **M6 — docs + demolition.** Rewrite control-flow.md, unified-dialect.md, and phase-inference.md to the sampling model; fold `function-exit-choices.md` into `sampling-model.md`.
  Delete the old HIR stack (`SplitPhases`, `TestPhaseAnalysis`, `hir.unified_func`/`unified_call`/`split_func`/`expr`/`yield`) once `silc` runs the `uir` pipeline at parity.
