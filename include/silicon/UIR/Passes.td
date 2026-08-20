//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_PASSES_TD
#define SILICON_UIR_PASSES_TD

include "silicon/UIR/Dialect.td"
include "mlir/Pass/PassBase.td"

def FlattenCFPass : Pass<"flatten-cf", "mlir::ModuleOp"> {
  let summary = "Lower structured UIR ops to block-based CF";
  let description = [{
    Converts structured UIR ops into flat block-based IR:

    - **Control flow:** `uir.if`, `uir.loop`, and their terminators
      (`uir.yield`, `uir.break`, `uir.continue`, `uir.return`,
      `uir.unreachable`) are lowered to `cf.br`, `cf.cond_br`, and
      `hir.return`.
    - **Phase grouping:** `uir.expr` is inlined into the parent block,
      and `uir.pin` is replaced by its inputs (identity removal).

    This pass runs after SplitPhases and before PhaseEvalLoop. After
    this pass, no UIR ops remain in the IR (except `uir.split_func`
    witnesses).
  }];
  let dependentDialects = [
    "mlir::cf::ControlFlowDialect",
    "silicon::hir::HIRDialect",
  ];
}

def NormalizeExitsPass : Pass<"normalize-exits", "mlir::ModuleOp"> {
  let summary = "Rewrite early function exits into single-exit value form";
  let description = [{
    Rewrites every function exit op that is not its region's terminator into
    choice/pick value form, leaving exactly one exit op as the region
    terminator. A `uir.func` has two regions that follow one set of rules:
    the body, whose return-like exit is `uir.return`, and the signature,
    whose return-like exit is `uir.signature`. The pass is phase-blind: it
    consults no phase information and runs before phase analysis, so
    analysis and sorting input is always single-exit.

    The encoding keeps every SSA value at a single phase: the exit decision
    and the exit values have different phases in general, so they travel as
    separate values and are never fused into one. Each structured control
    flow op (`uir.if`, `uir.expr`, `uir.loop`) whose regions hold a function
    exit is rebuilt with extra trailing results: one slot per function result
    carrying the exit path's returned value, plus a payload-free choice
    tagging how the region ended (tag 0 = exited, tag 1 = fell through). The
    tag lives at the op's own phase; the value slots are ordinary CF results
    that the standard floor, forwarding, and choice-ification rules deliver
    at their providers' phases. Slots a path does not provide are filled
    with `uir.unreachable_value`, never omitted. On `uir.if` and `uir.expr`
    the slots and tag ride the yield; on `uir.loop` they ride `uir.break`
    (existing breaks gain the tag-1 slots, and exits become breaks), so
    exits under loops become break-carried tags. The value slots' result
    types are one shared `hir.inferrable` per exit slot (one per function
    result for the body; one per argument and result type value for the
    signature), created at the region top; every exit site's own type
    operands unify into them (a `uir.signature` site carries none, so its
    slots are typed by the shared inferrables directly).

    After each rebuilt op, an `hir.pick` on the tag dispatches: the tag-0
    alternative consumes the op's exit-value slots directly (alternatives
    carry no block arguments), and the tag-1 alternative receives the
    remainder of the block. Nested exits produce a linear chain of binary
    picks, so continuations are never duplicated. At the region's top level
    the picks thread the exit values to the single trailing exit op — for
    the body a `uir.return` typed by the shared inferrables (floating, they
    satisfy the type operand rule's one-phase-earlier demand at any phase),
    for the signature a `uir.signature` consuming the delivered slots; dead
    code after exhaustive exits lands in tag-1 alternatives terminated by
    `uir.unreachable`.

    Choice construction and pick dispatch are executable single-phase control
    flow here, not phase-crossing vocabulary: which alternative runs is
    decided by the same-phase tag operand.
  }];
  let dependentDialects = ["silicon::hir::HIRDialect"];
}

def SortPhasesPass : Pass<"sort-phases", "mlir::ModuleOp"> {
  let summary = "Sort uir.func content into per-phase uir.phase ops";
  let description = [{
    Before rewriting anything, the pass takes custody of every function's
    declaration: it creates a `uir.split_func` under the function's original
    symbol name, holding a verbatim clone of the untouched signature region
    and an `impl` reference to the function, which moves aside under a fresh
    uniquified name. The flat signature is the late-binding type interface
    call checking inlines at call sites, and sorting distributes signature
    computations into phase ops irreversibly, so the flat form is preserved
    rather than reconstructed later. No symbol use is rewritten: a `uir.call`
    or `uir.phase_call` naming the function now resolves to the split_func,
    which declares the same argument and result phases. Function splitting
    later fills in the phase list and retires the implementation.

    Runs phase analysis on each `uir.func` and rewrites the internals of its
    body and signature regions into the sorted phase-op form, leaving the
    function op itself untouched: each region holds only `uir.phase` ops (at
    most one per phase, in increasing phase order), floating ops, and the
    region terminator. Ops anchored at a real phase move into that phase's
    `uir.phase` op in program order; floating ops stay at the region top
    level (or sink after operands that moved); function results move onto
    the `uir.phase_return` of their declared phase and the body's
    `uir.return` becomes `uir.unreachable`. Phase ops are only materialized
    for phases with content, where a phase return carrying function results
    counts as content. `uir.expr` and `uir.pin` phase grouping is dissolved
    first.

    Values computed inside a phase and consumed in a later one cross as
    phase-op results: the defining phase's `uir.phase_return` yields the
    value once, and consumers at any later phase — including function-result
    and type operands on later phase returns and the region terminator — use
    the `uir.phase` op's result directly, at any distance. Values already
    visible at the region top level (function arguments and floating ops)
    are consumed directly, without routing through a phase op.

    Each `uir.call` is decomposed into one `uir.phase_call` per externally
    visible callee phase (the callee's declared argument and result phases
    plus phase 0, in the callee's own frame). The call's own analysis phase
    is the single frame shift shared by all entries; consecutive entries
    chain in ascending phase order through an opaque context value that
    crosses phases like any other value. The entries derive from the callee
    declaration alone, so recursion decomposes like any other call.

    A `uir.if` at phase p whose arms hold later-phase content, or whose
    result selects between distinct later-phase values, is extracted into a
    choice-yielding branch plus a receiving `hir.pick`: the if is rebuilt at
    phase p with a per-arm `hir.choice` (tagging the taken arm and carrying
    the arm-local values the moved content samples) as an extra result, and
    the later-phase content and results move into a `hir.pick` at phase p+1
    that dispatches on the choice. The choice crosses phases as an ordinary
    crossing, and nested branches produce nested choices and picks. Only
    genuine choice-ification is lifted this way; a result diverging from its
    op's phase by identity forwarding or by never being produced is left for
    the divergence rejection.

    A `uir.loop` at phase p whose body holds later-phase content is extracted
    into a hit list of per-activation records plus a receiving `hir.replicate`
    at phase p+1: a `hir.opaque_list_create` is placed before the loop, each
    activation packs and pushes one record of the crossing values it samples,
    and the later-phase content moves into the replicate, which replays the
    records. Loop-phase early exits (`break`/`continue` in branch arms) make
    each record end in a nested exit-path choice that a nested `hir.pick` in
    the replicate dispatches on; nested loops produce nested hit lists. The
    list crosses phases as an ordinary crossing. A loop delivering a later-
    phase result first has that break value choice-ified: each break site
    carries a `hir.choice` of the value's snapshots in place of the value, and
    a `hir.pick` at phase p+1 after the loop reconstructs it, composing with
    record extraction.

    This is the only pass that runs or consumes PhaseAnalysis; the phase map
    never crosses the pass boundary.

    The pass currently covers straight-line content, call decomposition,
    single-boundary branch extraction, single-boundary loop-record extraction,
    and break-value choices, and rejects inputs that need the remaining sorting
    machinery with explicit diagnostics: early function exits (function-level
    exit chains), results delivered at a phase different from their op's own
    phase by identity-provided or never-produced control flow (which need
    result forwarding and unreachable-consumer handling), hoisting and content
    more than one phase away from its control flow, and backward uses from
    floating content nested under earlier-phase control flow. The pass's own
    output vocabulary (`uir.phase`, `uir.phase_return`, `uir.phase_call`) is
    likewise rejected in its input, as are ops inside regions the phase
    analysis does not model.
  }];
}

def ThreadPhasesPass : Pass<"thread-phases", "mlir::ModuleOp"> {
  let summary = "Thread sorted uir.func content into fully threaded phase form";
  let description = [{
    A cheap, purely structural normalization run after `sort-phases` and
    before the phase-blind function splitter. It reads phase numbers off the
    `uir.phase` op attributes and the function's declared argument and result
    phases; it never runs phase analysis. Both the body and signature region
    of every `uir.func` follow one set of rules.

    Sorted form leaves crossings free to *jump*: a value yielded by phase P
    may be consumed directly by any later phase, a function argument or a
    floating op may be consumed at any phase, and externally visible phases
    without content have no `uir.phase` op. This pass removes all such
    slack:

    - **Materialize missing phase ops.** After normalization there is exactly
      one `uir.phase` op for every phase in the contiguous range from the
      region's earliest phase to its latest, in increasing order. The body's
      range spans the function's declared argument and result phases (always
      including phase 0); the signature's range spans that same span shifted
      one phase earlier (see below) and may thus reach below the body's. Both
      are widened to cover every existing `uir.phase` op — pre-span phases
      earlier than the declared span and internal phases later than it
      included — and all phases in between. Gap phases and contentless span
      phases become empty `uir.phase` ops holding just a `uir.phase_return`.

    - **Thread crossings hop-by-hop.** Every value a `uir.phase` op at phase Q
      consumes from outside it comes, after normalization, from the
      immediately preceding phase op (Q-1) as one of its results. A value
      produced in phase P and consumed at phase Q > P+1 is yielded and passed
      on by every intermediate phase op; a function argument declared at
      phase P and consumed at a later phase enters the chain at phase P
      (arguments are consumed directly only in their own declared phase).

    - **Sink floats.** A floating op (a region-free pure op at the region top
      level) is cloned into each `uir.phase` region that uses it, before its
      first use; its operands are cloned or threaded along the same rules.
      Floating ops never consume phase-op results — this is verified and
      hard-errored. After normalization no floating op remains at the region
      top level.

    - **Signature interface.** The `uir.signature` terminator's operands are
      the type-level values of the function's arguments and results. A type
      ascribing a phase-P argument or result is phase P-1 content by
      definition: the per-phase function's signature must be constructible
      without evaluating anything, so it can only unpack the incoming context.
      Each operand is therefore delivered as a result of the `uir.phase` op at
      P-1, without exception; floating type values sink into P-1 and thread as
      above, which is what gives the type of a phase -1 argument a phase -2 op
      below the function's declared span. Such phases are ordinary pre-span
      phases: argument-free by construction and eagerly evaluated.

    Every function result already rides the `uir.phase_return` of its declared
    phase in sorted form; the pass asserts this rather than fixing it.

    Output contract: each region holds exactly `uir.phase` ops (one per phase,
    contiguous and increasing) plus the region terminator; every value
    crossing between phases is yielded by the immediately preceding phase op;
    no floating ops remain; nothing else. Inputs that violate the sorted-form
    input expectation — an unexpected op kind at the region top level, a
    floating op consuming a phase-op result, or a value referencing a later or
    non-result phase interior — are rejected with a diagnostic; this compiler
    errors loudly rather than silently mis-threading.
  }];
  let dependentDialects = ["silicon::hir::HIRDialect"];
}

def SplitFuncsPass : Pass<"split-funcs", "mlir::ModuleOp"> {
  let summary = "Split threaded uir.func ops into per-phase hir.func ops";
  let description = [{
    The phase-blind function splitter. It consumes the fully threaded phase
    form produced by `thread-phases` and never runs phase analysis: the only
    phase information it reads is the `uir.phase` op `phase` attribute and each
    function's declared `argPhases`/`resultPhases`.

    The pass is driven by the `uir.split_func` ops phase sorting left in the
    *sorted* state, each pairing the declaration with the `uir.func` its `impl`
    reference names. A split_func without an implementation reference is either
    an uncompiled declaration or already split and is left alone; a `uir.func`
    that no split_func claims is rejected, since nothing took custody of its
    declaration. For every such pair, in dependency-free order, the pass:

    - **Diagnoses** content later than the declared span. The span is the
      integer range over the declared argument and result phases, always
      including phase 0; a `uir.phase` op later than the span maximum executes
      runtime work past the last declared phase, which has no executor, and is
      a user error whose fix is declaring that phase (e.g. a `dyn` result).
      Because calls are already decomposed into `uir.phase_call` entries living
      inside phase ops, this uniformly catches call-implied late phases. The
      pass also rejects any region content that violates the threaded input
      contract (an op that is neither a `uir.phase` nor the terminator, a wrong
      terminator, a phase whose body is not straight-line, or a gap in the
      phases the two regions cover between them).

    - **Merges and splits** the two regions phase by phase into one private
      `hir.func @f.<phase>` per phase present in either region (negative phases
      quote the symbol; the names are cosmetic and uniquified against the
      module). The func's body is the signature region's phase-P content
      followed by the body region's, cloned op for op into the func of the
      phase each op already sits at; nothing moves between phases and nothing
      is recomputed, so a `uir.phase_call` in the signature is ordinary phase
      content needing no special handling. A func takes its phase's
      declared arguments plus a trailing opaque context argument for every phase
      but the first, and returns its phase's declared results plus a trailing
      opaque context result for every phase but the last. Values crossing a
      phase boundary are packed with `hir.opaque_pack` into the producing func's
      context result and unpacked in the consuming func. Each `uir.phase_call`
      becomes an `hir.call` to the callee's concrete per-phase func, resolved
      through the callee split_func's phase list — never by name convention —
      with its context threading like any other crossing.

    - **Reconstructs per-phase signatures** for each per-phase func. A phase
      boundary's opaque context carries what both regions' phase ops yield,
      concatenated signature first and assigned slots positionally. Nothing is
      deduplicated: a type the signature and the body both compute occupies one
      slot each, so a per-phase func declares its arguments with the
      signature's slots while its body reads the body's. Reconstructing that
      two computations are "the same" is not the splitter's job, and cannot be
      done for calls at all. A type ascribing a phase-P argument or result is
      phase P-1 content, so a per-phase signature region holds nothing but
      `hir.opaque_type` and one `hir.opaque_unpack` of the incoming context: it
      never reads the func's own arguments, which is what makes the per-phase
      MIR signature constructible after monomorphization and folding without
      evaluating anything. A `uir.signature` operand that is not a result of the
      phase P-1 signature op violates that contract and is a hard error. The
      function's *flat* signature is not reconstructed: it sits on the
      `uir.split_func`, preserved from before sorting.

    - **Fills the split_func**: its phase list maps each span phase to the func
      implementing it, and the `impl` reference and the implementation function
      are retired. When the function has pre-span compile-time phases, an
      `hir.eager_func` chains them into the earliest span entry and takes that
      phase's slot. Phases that exist only to compute signature types are
      pre-span phases like any other, so a function declaring arguments at its
      earliest span phase always has at least one.

    Because a call resolves through the callee's phase list, the module is
    processed in stages: every phase list is filled before any `uir.phase_call`
    is lowered, and implementations are retired only once all of them are, so
    the result never depends on the order the functions are visited in.
    Recursion and mutual recursion fall out of this.

    Anything the pass cannot handle is a hard error, never a silent skip.
  }];
  let dependentDialects = ["silicon::hir::HIRDialect"];
}

def SplitPhases2Pass : Pass<"split-phases2", "mlir::ModuleOp"> {
  let summary = "Split UIR unified functions into per-phase HIR functions";
  let description = [{
    Runs phase analysis on each `uir.func`, then splits it into per-phase
    `hir.func` ops using the phase map. Produces `uir.split_func` witnesses
    recording how each unified function was split, and `hir.multiphase_func`
    ops for groups of consecutive internal phases.

    The pass processes functions in topological order (callees before callers)
    so that `uir.split_func` witnesses are available when decomposing
    `uir.call` ops in callers.

    After this pass, `uir.func` and `uir.call` ops are gone. Structured CF
    ops (`uir.if`, `uir.loop`) survive in the per-phase `hir.func` bodies
    and are lowered by FlattenCF.

    With the `sort-only` option, the pass stops after the sorting stage:
    each `uir.func` is rewritten in place such that no control flow region
    contains content of a phase different from the region's own phase
    (phase grouping dissolved, calls decomposed, earlier-phase content
    hoisted, later-phase content extracted into replicate/pick receiving
    constructs), without creating any per-phase functions. The
    `annotate-phases` option additionally annotates all ops with the
    maintained phase map, for testing.
  }];
  let options = [
    Option<"sortOnly", "sort-only", "bool", "false",
           "Stop after the sorting stage, leaving rewritten uir.func ops">,
    Option<"annotatePhases", "annotate-phases", "bool", "false",
           "Annotate ops with the maintained phase map (testing)">,
  ];
  let dependentDialects = [
    "silicon::hir::HIRDialect",
    "mlir::cf::ControlFlowDialect",
  ];
}

#endif // SILICON_UIR_PASSES_TD
