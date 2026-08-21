# TODO

## Phase-Op Rebuild

Design: docs/design/phase-ops.md (passes), sampling-model.md (semantics), record-replay.md, function-exit-choices.md.
Resume: drive `hir.eager_func` from PhaseEvalLoop.

1. Drive `hir.eager_func` from PhaseEvalLoop: evaluate the head link, specialize forward, and replace the op with the same-name `hir.func` shim when a single link remains.
2. Switch `silc` from the HIR-level split (lib/HIR/Transforms/SplitPhases.cpp) to `sort-phases`/`thread-phases`/`split-funcs` at parity, then delete the old stack (`SplitPhases`, `TestPhaseAnalysis`, `hir.unified_func`/`unified_call`/`split_func`/`multiphase_func`/`expr`/`yield`).
3. Move the never-compiled-callee diagnostic out of `split-funcs` into the `uir.call` checker; it currently names an op the user never wrote, for the first offending site only.
4. Delete PhaseAnalysis step 3b (function-level multi-return result floor, lib/UIR/Analysis/PhaseAnalysis.cpp) and its multi-return coverage; assert single-exit input instead.
5. `RegionThreader::ensureThreaded` (lib/UIR/Transforms/ThreadPhases.cpp) recurses once per phase, so a wide declared span such as `uir.func @W(%a: -1000000)` blows the stack; nothing bounds a declared span.
6. `RegionThreader::computeRange` encodes the signature P-1 rule a second time behind an `isSignature` flag; it should fall out of the demand set `plan` already walks.
7. `hir.type_of %x` of a function argument sorts to phase p(x)-1, where `%x` is not available, and HIRToMIR papers over it with a dummy type constant; needs a real resolution once per-phase funcs flow through HIRToMIR.
   `check-calls` makes this fatal: coercing an argument turns a block argument into an op result, which `sort-phases` asserts on ("backward crossing survived analysis and extraction") where it tolerated the block argument, so the identity function `fn f(x: int) -> int { x }` crashes under `--check-calls --sort-phases`.
   `--canonicalize` in between folds the redundant `hir.type_of`/`hir.unify` and hides it; decide what runs between checking and sorting, since phase-ops.md names no stage there.
   test/Transforms/check-calls-composed.mlir dodges the shape by declaring its result type as `hir.uint_type %N`.
8. `uir.phase`/`uir.phase_return` tablegen docs claim the function-result-to-phase correspondence is "diagnosed by function splitting"; it is enforced by the thread-phases contract plus the `hir.func` verifier.
9. `CallOp::verifySymbolUses`'s `if (!callee)` branch in lib/UIR/Ops.cpp is dead: `callee` is a non-optional ODS attribute.

### Open questions and hazards

- `check-calls` copies a signature per consumer and shares nothing, so a signature-call DAG expands into a tree: eight nested functions whose signatures call the next twice produce 2008 `uir.call` ops from 90 input lines.
  Decide whether copying stays unbounded.
- `check-calls` coerces body arguments to their declared types but leaves signature-region arguments unconstrained; state why the two consumers of one declaration differ, or make them symmetric.
- `check-calls` silently ignores a `uir.call` outside a module-level `uir.func` (inside a `func.func`, say) and silently accepts already-sorted input holding `uir.phase` ops, where `thread-phases` then blames the wrong op.
  Both want a loud rejection once the checkable-region rule above is settled.
- Record extraction hard-errors on a `hir.pick` that carries a loop exit ("carries a loop exit but is not a branch", `scanBlock` in lib/UIR/Transforms/SortPhases2.cpp), while the same shape on `uir.if` sorts; `extractBranches` already treats pick as a divergence and the sorter itself emits picks.
  Either generalize the scan over divergences or record why picks are excluded.
- `hir.replicate`'s ODS description still sanctions `uir.return` in a receiving body, `hir.pick`'s says function exits never appear in alternatives, and neither op verifies it; state the rule once and enforce it before the M4 expander is written.
- Every function whose arguments start at phase P grows a signature phase P-1 computing their types, so even a plain phase-0 function splits into a pre-span `hir.eager_func` chain with an opaque context round-trip.
  Decide whether that cost is accepted or peepholed away; until item 1 lands nothing splits to a directly callable phase-0 function.
- `opaque_pack(opaque_unpack(x))` → `x` exists only as `OpaquePackOp::canonicalize`, so `createOrFold` in the splitter cannot use it; decide whether it should also be a folder.
- Candidate verifier invariant: all `uir.phase_call` entries of one decomposed call share one offset.
  Nothing relates a `uir.phase_call`'s callee phase to its enclosing `uir.phase`, so an inconsistently shifted decomposition is accepted silently.
- `SplitFuncOp::verifySymbolUses` cannot catch a swapped pair of link entries whose phases have identical arity; closing that needs a phase marker on per-phase funcs, i.e. a second source of truth. Producer obligation for now.
- The positional link list makes an entry's meaning a pure function of `argPhases`/`resultPhases`, so a future in-place declaration edit (M4 specialization dropping a const argument) would silently re-bind every entry. Nothing rewrites declarations today.
- A type both regions compute crosses in two context slots, so a per-phase func declares an argument with the signature's slot and types its body with the body's; CSE collapses the computations, not the slots.
  Decide whether split output is ever structurally type-checked, and whether the duplication goes away upstream. Not splitter logic either way.
- Decide whether "exactly one `uir.phase_return`, as the immediate region terminator" becomes a `uir.phase` verifier invariant.
- `split-funcs` handles only straight-line phases; a phase terminated by `uir.unreachable` is a hard error. Revisit with concrete pipeline experience.
- A branch none of whose arms rejoins has no choice to dispatch on, so its later content falls to the nested-content rejection (@AllArmsDivergeLaterContent in sort-phases-errors.mlir).
- Lift the sorter's hard error on break/continue-terminated `uir.expr` carriers (the D-rule break of sampling-model §4).
- Decide whether "expr bodies end in yield or return" is an `ExprOp` verifier invariant or a sorting-stage limitation; other terminators are verifier-legal today and dissolution rejects them with an ICE.
- Return-terminated exprs at the function body's top level stay undissolved until distribution.
- PhaseAnalysis processes only the entry blocks of body and signature in `run()` while `collectTerminators` walks all blocks; decide whether pre-FlattenCF multi-block regions are legal and verify or handle them.
- `PhaseAnalysis::reaches`/`diverges` are phase-blind: a later-phase breakless loop inside a phase-p arm makes the arm's phase-p yield count as non-provided even though phase p completes. Decide whether divergence should be phase-relative.
- Multi-block (flat-CFG) signature inlining at a call inside a single-block region is structurally impossible; today the shape fails via the post-pass verifier. Dissolves if signatures become single-block structured form.
- Watch: removing an expression's only consumer moves its effects to the block phase, since undemanded floats default there.
- A language-level `never` *type* (Rust `!`, coercing to any type) is an M5+ type-system feature, unrelated to the `never` bottom phase.

## Types and Inference

- cross-ex5.si: `hir.unify` survives to HIRToMIR with different operands, because `const { 4 + 4 }` unifies a computed type against an expected one that InferTypes cannot resolve.
  We cannot prove equivalence symbolically, but the canonicalizer should fold constant expressions down to constants that then unify.
- The `in_range` example in docs/examples/basics/operators.md fails with a `mir.return` type mismatch (`!hir.any` vs `!si.bool`); the `&&` short-circuit result is not coerced to `!si.bool`. Tagged `silicon-todo` to skip doc testing.
- Comparison ops (`hir.gt`, `hir.lt`, `hir.geq`, `hir.leq`, `hir.eq`, `hir.neq`) should take a second type operand for the operand type, e.g. `hir.gt %a, %b : %int_ty -> %bool_ty`.
- Add an HIR identifier op that passes an operand through to its result, so a named expression like `x` carries a location for phase-error diagnostics instead of pointing at the definition.
- Add `hir.constant` materializing any `#si.*` constant attribute, replacing constant-like ops such as `hir.int_type`/`hir.unit_type`; `hir.constant_int` stays for literals with an inferrable type.
- Consider migrating `hir::getTypeOf` (lib/HIR/Ops.cpp) to op-declared information, e.g. a `getTypeOfResult` on `TypeOperandsOpInterface`, instead of a hardcoded TypeSwitch.
- Document how integer literal type inference works in docs/examples/.

## Language Surface

- Remove `const fn` / `dyn fn` from parser and codegen; they add nothing beyond arg/result modifiers and `const`/`dyn` blocks.

## Passes and Lowering

- Rewrite FlattenCF to process regions outer to inner in a single scan, inlining structured CF ops as it goes so newly inlined regions are visited naturally; see MLIR's SCF-to-CF lowering.
- Old HIR SplitPhases silently skips a `UnifiedCallOp` whose callee has no `SplitFuncOp`; dies with the old stack.

## Testing Gaps

- HIR roundtrip: `FuncTypeOp`, `NextPhaseOp`, and visibility variants of `UnifiedFuncOp`/`SplitFuncOp`/`MultiphaseFuncOp`.
- Base roundtrip: heterogeneous and nested `OpaqueAttr`.
- HIR verifier errors: `SplitFuncOp` array-size mismatches, `MultiphaseFuncOp` `argIsFirst` size, `ReturnOp` `values`/`typeOfValues` size, `SignatureOp` outside a valid parent, `NextPhaseOp` outside `hir.func`.
- HIRToMIR: untested `emitBug` paths (negative/excessive uint width, non-constant func_type args/results, non-constant call result type, return type-operand mismatch across returns); most are guarded by `shouldLower`.
- HIRToMIR positive: `arith.select` (needs arith registered in silicon-opt), `hir.opaque_unpack` (blocked by `shouldLower`), `hir.inferrable`.
- MIRToCIRCT: empty error file, 8 untested `emitBug` paths; missing multi-instance counter and iN→i1 source materialization tests.
- Interpret: `arith.select`, `UnrealizedConversionCastOp` forwarding, `cond_br` condition error, module function skip.
- PhaseEvalLoop: "still pending" multiphase_func note, sub-pipeline failure propagation.
- SpecializeFuncs: void-result evaluated func chaining. InferTypes: no error test file at all.
- Old HIR SplitPhases: no test for the "op uses value from later phase" `emitBug`; dies with the old stack.
- Add a positive test/HIR/check-types.mlir (only error tests exist), a `hir.if` → `mir.if` case in test/Conversion/hir-to-mir.mlir if the op still exists, and a codegen test for `const { expr }` producing `hir.expr` with `phaseShift=-1`.
- Add `isModule` cases to test/MIR/interpret.mlir, test/HIR/specialize-funcs.mlir, and test/Transforms/phase-eval-loop.mlir.

## Postponed Long-Term Fixes

- Structured-CF purity: `Pure` exit terminators make an unused exit-carrying `uir.if`/`uir.loop` trivially dead, so DCE silently deletes breaks and returns.
  Load-bearing for M4, since replay is data-only and unused exit-carrying picks must survive to evaluation.
- MIRToCIRCT maps `!si.int` to `i64`; once bitwidth inference exists this should be a diagnostic instead.
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
- Consider `FunctionOpInterface` on `mir.func` and `CallOpInterface` on `mir.call`/`hir.call`.
- Consider consistent naming for Base attributes (currently mixed `Base`/`Si`/no prefix).
