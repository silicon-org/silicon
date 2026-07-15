# TODO

- cross-ex4.si, cross-ex8.si: `mir.return` type mismatch (`!si.uint<N>` vs `!si.opaque`) when a function calls another dependent-type function.
  The return type in the MIR func doesn't match because the opaque signature types aren't properly resolved after specialization.
- cross-ex5.si: `hir.unify` survives to HIRToMIR with different operands.
  The `const { 4 + 4 }` expression creates a unify between the computed type and the expected type that InferTypes can't resolve.
  We can't symbolically prove equivalence, but we want the canonicalizer to be able to at least resolve constant expressions down to simple constants that will then unify.
- Type unification can't prove structural equivalence of `uint` types with different width operands.
  Need to teach InferTypes/CheckTypes to structurally compare `uint_type` ops by unifying their widths.
- The `in_range` example in `docs/examples/basics/operators.md` fails with `mir.return` type mismatch (`!hir.any` vs `!si.bool`).
  Tagged `silicon-todo` to skip doc testing.
  Root cause: the `&&` short-circuit produces a value typed `!hir.any` that doesn't get coerced to `!si.bool` before the MIR return.
  Unrelated to multi-block signatures (which are now handled by phi tracing in `resolveTypeValue`).
- Document how integer literal type inference works in docs/examples/.
- HIR needs a `hir.constant` op that can materialize any of the `#si.*` constant attributes.
  Use that op to materialize most constants that have a fully-known type and/or value; this would replace constant-like ops such as `hir.int_type`, `hir.unit_type`, etc.
  We'd still need `hir.constant_int` to construct an integer literal with an inferrable type.
- Change syntax from `hir.return %a, %b -> (%a.ty, %b.ty)` to `hir.return %a, %b : %a.ty, %b.ty`
- Rewrite FlattenCF to process regions from outer to inner, iterating over the ops once and inlining structured CF ops as it goes along.
  Once a structured CF op is inlined, its blocks and regions are inserted ahead of that op, such that the linear region/block scan will naturally visit the newly-inlined region.
  For inspiration, look at how the SCF-to-CF lowering in MLIR does things.
- Improve UIR op parsing/printing (e.g. uir.call, maybe others) to not print the implied `!hir.any` type
- Consider migrating `hir::getTypeOf` (Ops.cpp) to op-declared information, e.g. a `getTypeOfResult` method on `TypeOperandsOpInterface`; it currently hardcodes the op-to-type-operand mapping in a TypeSwitch.
- Add an HIR identifier op that just pass through an operand to its result.
  We'll use this op to have a mechanism to capture the location of a named expression like `x`.
  When reporting phase errors, we can stop at `x` and indicate an error right there, instead of pointing the user at a definition like the `x: const int` arg of a function, which is far more confusing.

## Phase Inference Redesign (sampling model)

The semantics were settled in a design interview on 2026-07-08.
**Read `docs/design/sampling-model.md` (the model), `docs/design/function-exit-choices.md` (early exits), `docs/design/implementation-plan.md` (ops, milestones M0-M6), and `docs/design/phase-ops.md` (pass architecture: `uir.phase` contract between sorting and splitting; supersedes the merged-pass plan) before touching anything here.**
The old replicate design in `docs/design/control-flow.md` and the transparent-yield/futures model in `docs/design/phase-inference.md` are superseded where they conflict; docs get rewritten at M6.

- Milestones: M0 ops done → M1 PhaseAnalysis updates done (2026-07-10, incl. design-review fixes) → M2 sorting stage (mostly done 2026-07-10; see below) → M3 phase-op rebuild (immediate next steps below; docs/design/phase-ops.md) → M4 evaluation (interpreter lists/choices/vars, SpecializeFuncs expansion) → M5 language surface (assignment, loops, var, definite-return check) → M6 docs rewrite + old-stack removal.

### Immediate next steps: phase-op rebuild (docs/design/phase-ops.md; sorting first, splitter last)

1. Rebuild sorting case-by-case, one PR per case, porting machinery from lib/UIR/Transforms/SortPhases.cpp: call decomposition into `uir.phase_call` (from callee declaration, no witness) → branch extraction (if/pick) → loop records/replicate → break-value choices → partial activations; hand-written target IR for these cases is in test/UIR/sorted-form.mlir (landed 2026-07-11).
   Landed in the `sort-phases` pass (lib/UIR/Transforms/SortPhases2.cpp): wrap/float/return rules, dissolution, sorted-vocabulary and coverage rejection (case 1), crossings as phase-op results/yields incl. signature-region phases and terminator/return consumers (case 2), call decomposition into `uir.phase_call` (case 3, declaration-derived entries, frame shift = call's analysis phase, context via ordinary crossings, recursion falls out), branch extraction of if/pick choice-ification (case 4, `extractBranches`/`extractFromBranch`), and loop-record extraction (case 5, `extractLoops`/`extractFromLoop`/`scanLoopBody`: hit list + per-activation records with exit-path choices, receiving `hir.replicate` at p+1 with nested picks, nested loops nest hit lists, loop-with-dyn-call composes with decomposition), and break-value choices (case 6, `extractBreakValues`: each break site carries a `hir.choice` of the value's snapshots in place of the later-phase value, the loop is rebuilt with the choice as an extra phase-p result, and a receiving `hir.pick` at p+1 reconstructs the value; runs before record extraction in `extractFromLoop`, so it composes with the hit list/replicate; shared break-value slices and breaks under-supplying a result stay rejected); scaffolding errors remain for early exits, hoist, and backward/distance-2 crossings.
   Case 7 (partial activations, the remaining rejected divergence family in `scanLoopBody`/`rejectUnimplemented`): rejoining-arm content in exit-carrying branches (needs the arm-local sub-choice, function-exit-choices.md §3), nested divergence (exits inside exit-carrying or rejoining arms), and exit-carrying branches with demanded later-phase results. Old-stack porting source: the same-named machinery in SortPhases.cpp. Distinct from case 5's exit-path records, which only cover the flat form (straight-line exit arms, no rejoining content).
   `uir.call` verifier now cross-checks arg/result phases against the callee declaration (lib/UIR/Ops.cpp), so decomposition trusts them; the callee-is-not-a-func case is still only a pass-level error (uir.call, unlike uir.phase_call, does not require its callee to be a uir.func/split_func) — decide whether to tighten the verifier.
   Settled 2026-07-13 (divergence/dead-code interview; work items below):
   - `never` is non-viral: no unreachable-consumer propagation; the analysis is already correct (never contributes no lower bound). Add a `uir.unreachable_value` op (basic/errors tests); the sorter rewrites a real-phase op's use of a never value to it in the consumer's phase; the interpreter traps on executing one.
   - Sorting is demand-blind (partially landed with the zero-use-anchor removal in PhaseAnalysis): dead content now derives real phases and extracts like live content; the backward-use-from-dead-float rejection lifted (test moved to sort-phases.mlir @DeadContentInBranch). Residual: the sorter's float-sink rule and the distribution-time backward-crossing error (SortPhases2.cpp `PhaseSorter::sortRegion`) mostly lose their trigger domain — the internal-crossing bound (below) now derives frames like frame_sinks to a real phase instead of leaving a mis-sinkable float — and the backward error lost its only test coverage; but "float ops are operand-free" is not an invariant (a dead consumer of a `never` region result stays float, and becomes reachable again once `uir.unreachable_value` forwarding lands), so decide then whether to assert or delete the machinery.
   - Float frames landed 2026-07-14 (per the 2026-07-13 evening interview): a floating expr whose anchored channels all carry bottomless values stays float wholesale; interior pure content carries vacuous derived phases no sorting rule reads (`hasLaterContent` skips sentinel-phase subtrees, the `rejectUnimplemented` assert restated as pure-region-free-only); lit tests in sort-phases.mlir (@DeadLoopInDeadExpr etc.) and phase-analysis.mlir @DeadLoopBottomlessChannelsFloats.
     Also fixed `cloneFloatingUses`: only region-interior defs clone (outside defs are visible and used directly), every clone is stamped, and a dead frame in the cloned tree is a loud not-implemented error (rematerializing one would execute its interior; diverging frames would inject nontermination) — tests @ClonedFloatTree and @FloatingFrameRematerialized.
   - Internal-crossing bound (landed 2026-07-15, revising the frame_sinks interview reasoning): the interview justified frame_sinks deriving to 1 via "break loop-locked, loop frame-locked", but that reasoning also holds for the bottomless case that must float — the real differentiator is that frame_sinks has an interior op consuming an operand below its own phase (a 0→1 crossing) while bottomless has all interior content at one phase. The clean general rule landed in `scanFloatingContent`: an internal crossing bounds the frame to the crossing op's phase; frames with no such crossing still float. Revisit if a case wants a crossing to *not* materialize the frame.
   - Settled 2026-07-15: DCE must not change program behavior, so deleting exit-carrying CF (unused `uir.if` holding a `uir.break`) is illegal; likely misdeclared terminator effects — see the structured-CF purity item under Postponed Long-Term Fixes.
2. Then the new territory on the new representation: function-level exit chains on `uir.phase_return` (docs/design/function-exit-choices.md) and lifting the M2 v1 hard errors listed below.
3. Last: minimal phase-blind func splitting — first a structural normalization pre-pass (in or right before the splitter: materialize phase ops for external phases and threading hops, thread all crossings through adjacent phase ops, sink floating ops), then one per-phase func per `uir.phase` op (no synthesis), opaque context from phase results, sig-crossing relocation into the previous phase's body, split_func/multiphase_func emission; diagnostic (not verifier) on non-phase-op content; test/UIR/sorted-form.mlir functions are the input contract.
4. Old SortPhases/SplitPhases2 stay intact and green throughout as porting source and reference; switch silc at parity and delete them plus the sort-only flag in one removal PR (folds into M6).
- Ops landed 2026-07-11: `uir.phase`/`uir.phase_return`/`uir.phase_call` with verifiers and basic/errors tests; phase_call entries are declaration-derived (arg/result phases plus 0), context in/out iff not first/last external phase.
- Settled 2026-07-11 (design review + interview; see phase-ops.md): sorted output allows direct cross-phase use of args/floating ops and skip crossings; phase_returns with values/yields are content (result-only phases get ops, contentless phases none); a normalization pre-pass at split time fills gap phases, threads crossings hop-by-hop, and sinks floating ops, so the splitter's contract is purely structural (phase ops + terminator only) and it never synthesizes funcs; body and signature regions follow one set of rules.
- Splitter questions (step 4, deferred to the very end after sorting is done): execution vehicle for internal phases — trailing, but also leading and mid-range ones, where the declaration-derived context chain already fixes a direct edge between the surrounding external phases that the internal phase's execution must interpose on — today's witness exposes them as caller-visible entries (`groupPhases` trailing group), contradicting declaration-derived `uir.phase_call` entries, and `hir.multiphase_func`'s argIsFirst/results-from-last shape can't absorb them into the last external group as-is; also `uir.split_func`'s remaining purpose (uir.call decomposition dissolves; symbol continuity/SymbolDCE?) and what its signature region holds once signatures contain `uir.phase` ops (`uir.phase` is HasParent uir.func only).
- Early-exit design settled 2026-07-10 (docs/design/function-exit-choices.md): exit chains built at return sites, receiving phases keep multiple returns, picks born at split time, forward-only.
- M2 landed in lib/UIR/Transforms/SortPhases.cpp (sorting stage of split-phases2, sort-only/annotate-phases options): dissolve, call decomposition, hoisting, branch/loop extraction, nested lists, distance-2, break-value choices, return carriers, partial-activation records (per-exit record pushes with nested exit-path choices per sampling-model §6); tests in test/UIR/sort-phases{,-errors}.mlir.
- M2 remaining: all-empty-deletion negative test, and lifting the v1 hard errors on shared break-value slices, break/continue-terminated expr carriers (§4 D-rule break), rejoining-arm content in exit-carrying branches (needs the arm-local sub-choice from function-exit-choices.md §3), nested divergence (exits inside exit-carrying or rejoining arms), and exit-carrying branches with demanded later-phase results.
- Sorting invariant gaps to revisit in M3: uir.expr with unreachable terminator errors in dissolution; return-terminated exprs at the function body's top level stay undissolved until distribution.
- Decide whether "expr bodies end in yield or return" is an ExprOp verifier invariant or a sorting-stage limitation: break/continue/unreachable-terminated exprs are verifier-legal today, and both old and new dissolution reject them with an ICE-style "cannot be dissolved" error.
- sort-phases `extractFromBranch` crashes (`cast<YieldOp>`) on an if arm terminated by `uir.unreachable` (verifier-legal, no exit op on the path); the exit scan only knows return/break/continue — same yield-or-exit classification question as the expr-terminator item above.
- Yield/break arity against the parent op's results is not a verifier invariant (unlike continue and return): PhaseAnalysis silently treats short yields as "does not provide", which is now load-bearing for the multi-provider selection rule (a short-yield if arm makes a result single-provider and dodges the floor); `extractBreakValues` errors at pass level instead — decide on one verifier invariant and assert in passes.
- Decide whether `uir.phase_call` requires a `uir.phase` ancestor (verifier rule like `uir.phase_return`); blocked on the split_func signature question above, since split_func signatures cannot hold `uir.phase` ops; meanwhile the sort-phases pass rejects all sorted vocabulary in its input.
- M3 pass structure superseded (2026-07-11, docs/design/phase-ops.md): sorting emits `uir.phase`/`uir.phase_return`/`uir.phase_call` as an explicit IR contract and is the only consumer of PhaseAnalysis; func splitting is a separate phase-blind pass; `uir.phase_call` is built from callee declarations, so no witnesses at sort time, no topological order, no cycle rejection, and the sort-only flag goes away.
- Floating-expr fallback settled 2026-07-10 (see implementation-plan.md): undemanded effectful floating exprs default to their block phase; a cap-at-block-phase alternative was rejected (breaks dyn-direction sliding of argument sub-exprs). Watch item: removing an expr's only consumer shifts its effects to the block phase — revisit if this bites.
- Never-produced CF results get the bottom phase `never` in PhaseAnalysis (settled 2026-07-10); a language-level never *type* (Rust `!`, `TyKind::Never`-style, coerces to any type) is a separate type-system feature for M5+; The sorting stage (M2) must treat `never` results as unreachable, not as choice-ification obligations.
- SplitPhases2.cpp determinePhaseRange/distribution test `phase == INT16_MIN` instead of `!isRealPhase`, so a `kNeverPhase` op entry would poison the range; the sorter's computePhaseRange does it correctly.
- PhaseAnalysis loose ends: only entry blocks of body/signature are processed in run() while collectTerminators walks all blocks — decide whether pre-FlattenCF multi-block regions are legal and verify or handle; step-3b return-value collection guards `i < values.size()` where it could now assert (FuncOp::verifyRegions enforces return arity as of case 1 of the sorting rebuild).
- InferTypes/CheckCalls/CheckTypes region support (walk into `uir.if`/`uir.expr`/`uir.loop`, optimistic type-tree hoisting) still needed; slot in before/alongside M2.
- Old-stack removal (old `SplitPhases`, `TestPhaseAnalysis`, `hir.unified_func`/`unified_call`/`split_func`/`expr`/`yield`) folds into M6.
- M6 docs: introduce the "frame" concept and the list of ops locked to the frame's phase (terminators always, break/continue phase-equal to target loop, if/loop, call and pinned expr/pin with declared offsets, effectful ops) vs demand-driven floating content (unpinned exprs, pure region-free ops, constants).

## Dialect Review: Missing Error Handling

- **SplitPhases: silent skip when `calleeSplitFunc` is null.**
  Line 383: if a `UnifiedCallOp` references a callee with no `SplitFuncOp`, it is silently skipped.

## Dialect Review: Missing Tests

### Roundtrip tests (`basic.mlir`)

- **HIR**: `FuncTypeOp` and `NextPhaseOp` untested; visibility variants for `UnifiedFuncOp`/`SplitFuncOp`/`MultiphaseFuncOp` untested
- **Base**: limited `OpaqueAttr` roundtrip coverage (no heterogeneous or nested opaque tests)

### Verifier error tests (`errors.mlir`)

- **HIR**: missing tests for `SplitFuncOp` array-size mismatches, `MultiphaseFuncOp` `argIsFirst` size mismatch, `ReturnOp` `values`/`typeOfValues` size mismatch, `SignatureOp` outside valid parent, `NextPhaseOp` outside `hir.func`

### Pass error tests

- **HIRToMIR**: remaining untested `emitBug` paths (negative uint width, excessive uint width, non-constant func_type args/results, non-constant call result type, return op typeOfArgs/typeOfValues mismatch between multiple returns); most are guarded by `shouldLower` and hard to trigger from IR
- **MIRToCIRCT**: empty error file, 8 `emitBug` paths untested
- **PhaseEvalLoop**: no test for "still pending" multiphase_func note, sub-pipeline failure propagation
- **SpecializeFuncs**: no test for void-result evaluated func chaining
- **SplitPhases**: no test for "op uses value from later phase" `emitBug` (defensive check, hard to trigger from well-formed IR)
- **InferTypes**: no error test file

### Pass positive tests

- **HIRToMIR**: remaining untested patterns (`arith.select` requires arith dialect registration in silicon-opt, `hir.opaque_unpack` blocked by shouldLower, `hir.inferrable` requires unresolved type setup)
- **Interpret**: missing tests for `arith.select` (arith dialect not registered in silicon-opt), `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip
- **MIRToCIRCT**: missing tests for multi-instance counter, source materialization (iN→i1)

### Other focused test gaps

- Add `hir.if` → `mir.if` test case to `test/Conversion/hir-to-mir.mlir` -- unless we've already removed the op
- Add positive `test/HIR/check-types.mlir` (only error tests exist today)
- Add codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1` in `test/Codegen/basic.si`
- Add `isModule` test cases to `test/MIR/interpret.mlir`, `test/HIR/specialize-funcs.mlir`, and `test/Transforms/phase-eval-loop.mlir`

## Phase Inference Design Review

Settled by the sampling model (`docs/design/sampling-model.md`); kept here until the M6 docs rewrite lands them:

- Multiple `uir.continue`/`uir.return` with earlier-phase values: floor rule; later-phase values choice-ify. No SSA-identity same-value exemption (dropped 2026-07-13; sampling-model.md §7 revised).
- CF result phase lower bound: yes — a CF op at phase p selecting distinct values cannot deliver at a phase < p (causality); the transparent-yield "future" for later phases is replaced by choice-ification (§7).
- `const { dyn { expr } }` value results: governed by the causality rule — a phase p op can never consume a phase > p value; block result phase is the yielded value's phase, and the asymmetry vs `dyn { const { ... } }` is intentional.

## Language Cleanup

- Remove `const fn` / `dyn fn` support from parser and codegen.
  These don't bring distinct semantics beyond arg/result modifiers and `const { ... }` / `dyn { ... }` blocks.
  See docs/design/phase-inference.md for the rationale.

## HIR Ops

- Comparison ops (`hir.gt`, `hir.lt`, `hir.geq`, `hir.leq`, `hir.eq`, `hir.neq`) should take a second type operand for the type of the values being compared, e.g. `hir.gt %a, %b : %int_ty -> %bool_ty` instead of just `hir.gt %a, %b : %bool_ty`.

## Postponed Long-Term Fixes

- Investigate memory effects and DCE/canonicalize behavior of UIR structured CF (Pure exit terminators make unused exit-carrying `uir.if`/`uir.loop` trivially dead, silently deleting breaks/returns); tackle after the phase analysis/sorting/splitting work is done.

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
- Consider implementing `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix)
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
