//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # Sort Function Content into Per-Phase uir.phase Ops
//
// This pass rewrites the internals of each `uir.func` into the sorted
// phase-op form: every region holds only `uir.phase` ops (at most one per
// phase, in increasing phase order), floating ops, and the region
// terminator. It is the only consumer of PhaseAnalysis; the phase map never
// crosses the pass boundary, and the function op itself is left untouched.
//
// Body and signature regions follow one set of rules; the only dispatch is
// on the terminator kind, not on region identity:
//
// - **Wrap**: every region-top-level op with a real analysis phase p moves,
//   in program order, into the single `uir.phase` op for p; the phase ops
//   are created in ascending phase order immediately before the region
//   terminator. All remaining top-level ops precede the phase ops, so
//   direct uses of them from inside phase regions dominate.
//
// - **Float**: a top-level op without a real phase (the float and never
//   sentinels) stays at the region top level, unmoved: a floating pure op
//   is a phase-compatible provider at any phase, and sorting never assigns
//   it one. Nothing a float references ever moves into a phase region — a
//   real-phase operand would have given the pure op a real derived phase,
//   and never operands are severed by never-forwarding — so keeping its
//   position always preserves dominance. Sinking floats into the phase
//   regions that use them is exclusively the splitter's
//   threading-normalization concern.
//
// - **Return**: the body's `uir.return` dissolves. The value and type
//   operand of result i move onto the `uir.phase_return` of the phase op
//   for the function's declared phase of result i, creating the phase op
//   if it does not exist: a phase return carrying function results is
//   content, so a result-only phase still gets its op, while contentless
//   phases get none. The body terminator becomes `uir.unreachable`. A
//   `uir.signature` terminator stays in place.
//
// - **Cross**: a value computed inside a phase and consumed in a later one
//   crosses as a phase-op result: the defining phase's `uir.phase_return`
//   yields it once, and every use outside the defining phase — later phase
//   content, function-result and type operands riding a later phase
//   return, or the region terminator — is redirected to the `uir.phase`
//   op's result, at any distance. Hop-by-hop threading through
//   intermediate phases is normalization's job, not sorting's.
//
// Before sorting, `uir.expr` bodies are inlined and `uir.pin` outputs are
// replaced by their inputs: the grouping ops exist to communicate phase
// constraints to PhaseAnalysis and only stand in the way once the phase map
// is computed. Then every `uir.call` is decomposed into `uir.phase_call`
// entries, one per externally visible callee phase; every earlier-phase
// branch — a `uir.if` or an `hir.pick` — with later-phase content or a
// choice-ified result is decomposed
// into a choice-yielding branch plus a receiving `hir.pick`; and every
// earlier-phase producing frame — a `uir.loop` or a `hir.replicate` — with
// later-phase content is decomposed into a hit list of per-activation records
// plus a receiving `hir.replicate`:
//
// - **Decompose**: a call is replaced by one `uir.phase_call` per phase in
//   the callee's declaration-derived external phase set (its declared arg
//   and result phases plus phase 0, in the callee's own frame). The call's
//   own analysis phase is the single frame shift shared by all entries, so
//   the entry for callee-frame phase p lands at caller phase shift + p. Each
//   entry carries the call's arguments and results filtered to p, and
//   consecutive entries chain in ascending phase order through an opaque
//   context value. The entries are ordinary ops that the wrap and cross
//   rules then place and thread like any other content; the context value
//   crosses phases as an ordinary crossing. Deriving the entries from the
//   callee declaration alone means no split witness, no callee-before-caller
//   order, and recursion decomposes like any other call.
//
// - **Extract branches**: a branch (`uir.if` or `hir.pick`) at phase p whose
//   arms hold later-phase content, or whose result selects between
//   later-phase values (several arms yield it, regardless of the values'
//   identity), is rebuilt at phase p with a per-arm `hir.choice` as an extra
//   result; the later-phase content and results move into a `hir.pick` at
//   phase p+1 that dispatches on the choice. The choice crosses phases as an
//   ordinary crossing. A pick is a branch like any other — picks are
//   executable single-phase CF, and a receiving pick is the next boundary's
//   producing CF — differing from an if only in its dispatch operand
//   (selector instead of condition), its payload-carrying arm block
//   arguments (ordinary arm-local phase-p values that cross as choice
//   payload), and its lack of result type operands. A later-phase result
//   choice-ifies whenever a reaching arm provides it, whether several arms do
//   (a genuine selection) or a single arm does while the others provably
//   diverge (an unreachable terminator or a breakless loop) — the diverging
//   arms yield `uir.unreachable_value` in every slot and become dead receiving
//   alternatives the choice never selects. A diverging arm's own later-phase
//   content extracts exactly like any other arm's, into its dead receiving
//   alternative: reachability never gates what is sorted, and an execution
//   entering the arm hangs at phase p, so the content's phase never runs.
//   Never-produced results (the whole op
//   diverges) are not lifted; their consumers are already
//   `uir.unreachable_value` from never-forwarding.
//
// - **Extract records**: a producing frame — a `uir.loop` at phase p, or a
//   `hir.replicate` at phase p replaying an earlier extraction's records —
//   whose body holds later-phase content, or whose result is at a later phase,
//   is rebuilt at phase p pushing one record per activation onto a hit list,
//   and the later-phase content moves into a `hir.replicate` at phase p+1 that
//   replays the records. A replicate frame's activations are its replays, one
//   per incoming record, ending at its body's terminal yield. The body
//   decomposes into content segments separated by exit-carrying divergences,
//   and a record carries up to two slots per divergence: a cf choice tagging
//   which arm ran (present when the divergence holds later-phase content or a
//   demanded later result; its payload nests the arm's own record body, so
//   nested divergences deepen it recursively), and a reach link gating the
//   following segment (REACHED with the segment's snapshots and the next
//   divergence's slots, or an empty NOT_REACHED ending the record). Every
//   activation-ending terminator pushes its record in place: passed divergences
//   contribute their rebuilt branch's cf result and a REACHED link, and the
//   exiting divergence a cf built at the site and a NOT_REACHED link. The
//   receiving replicate replays by data alone — per level a cf pick (arm
//   dispatch, recursing into the arm's own levels) and a reach pick (segment
//   gate) — so no content is cloned across record variants and no exit
//   terminators appear in the replay. When a divergence carries demanded
//   later-phase results, the cf pick is valued: each alternative yields the
//   arm's operand for the result, and the pick's results reconstruct the values
//   for the following segments. The list crosses phases as an ordinary
//   crossing. A later-phase loop result (a break value) is not its own delivery
//   concept: before record extraction, each such result is lowered onto a
//   receiving-phase `hir.var`, each break storing its later operand into the
//   variable and dropping it, and a read after the loop feeds the result's
//   uses. The store is ordinary later-phase content on the break's record path,
//   so the record machinery replays it once per activation in the exit
//   divergence's cf pick — value slice, effects, and nested- divergence
//   reconstruction included — and no content is sliced or duplicated.
//
// A `never`-phased value (a control-flow result no path provides) is
// non-viral: before extraction, every real-phase use of one is rewritten to a
// `uir.unreachable_value` at the consuming phase, so the `never` result stays
// unused on its producer and never crosses a boundary.
//
// A single-provider control-flow result (one arm yields it, the others exit or
// never provide) is not a selection but a transparent forwarding of the
// provider's operand: when that operand is defined outside the op it forwards
// directly to the result's uses and the result drops from the op, in either
// direction (the operand may be earlier or later than the op). A floating
// provider defined inside the op forwards the same way — a float is a
// region-free pure tree, materializable at any point, so its tree is cloned in
// front of the op and the result forwards to the clone. Forwarding runs
// before extraction, so branch extraction and break-value delivery only ever
// see genuine selections — a multi-provider result, or a single provider
// defined inside the op whose value varies per path or iteration and so needs
// choice-ification even at a provider count of one.
//
// Branch and record extraction share one fixpoint: every receiving `hir.pick`
// and `hir.replicate` a rule emits is ordinary phase-crossing control flow
// that re-extracts at its next boundary exactly like user-written CF, with no
// per-origin distinction. A `hir.replicate` at phase p is a producing frame
// like a `uir.loop`: content later than p inside its replay body re-records
// onto a fresh hit list, pushed once per replayed record, and moves into a
// receiving replicate at p+1. Per-record values the later content consumes
// ferry hop by hop — a phase-p replay unpacks them from the incoming record
// (block arguments, pick reconstructions) and re-packs them as fields of the
// outgoing one — while iteration-invariant values keep crossing directly.
// Each extraction step moves content exactly one boundary forward, so the
// fixpoint carries it any number of phases; a phase with no content of its
// own then merely transcodes one hit list into the next, which is the general
// rule's output, not a special case. A rule only lifts later-phase content
// that already sits at the top level of the blocks it owns: content still
// buried inside a same- or earlier-phase region op — deposited there by a
// nested frame's extraction mid-sweep — defers to the sweep after the owning
// rule has lifted it, keeping every move in program order.
//
// The pass covers straight-line content, multi-boundary branch and record
// extraction, break values delivered through receiving-phase variables,
// single-provider result forwarding, and hoisting of body content earlier
// than its enclosing frame, in both function regions alike. Inputs that need
// the remaining sorting machinery are rejected with explicit diagnostics
// instead of being sorted incorrectly: later-phase content trapped in a
// branch none of whose arms rejoins (there is no reaching arm to dispatch a
// choice on), which no extraction rule owns.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/Support/Debug.h"
#include <map>

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "sort-phases"

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_SORTPHASESPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

//===----------------------------------------------------------------------===//
// PhaseSorter
//===----------------------------------------------------------------------===//

namespace {
struct BodyPlan;

/// One arm of a cf-carrying divergence: the recursive decomposition of the
/// arm's block. An arm is a block like the loop body, so it decomposes into
/// its own content segments and nested divergences. `region` is null for an
/// absent else region; the divergence rebuild materializes one so the
/// fall-through path yields its cf tag.
struct Arm {
  Region *region = nullptr;
  std::unique_ptr<BodyPlan> plan;
  /// Whether the arm's terminal yield is reachable: the terminal is a yield
  /// and control reaches it (`PhaseAnalysis::reaches`). Only such arms
  /// rebuild their yield to deliver the divergence's cf choice.
  bool reachesYield = false;
  /// The arm's yield operands for the divergence's demanded later-phase
  /// results, in `laterResults` order; a null entry marks a result the arm
  /// does not provide (a short yield, or an arm that never rejoins), which
  /// the receiving cf pick's alternative yields as `uir.unreachable_value` —
  /// its consumers replay only on paths the reach links gate off.
  SmallVector<Value> laterYieldOperands;
};

/// An exit-carrying op in a loop body — or nested in an arm — where
/// activations diverge: some paths end the activation here, others fall
/// through. Each record carries up to two slots for it: a cf choice tagging
/// which arm ran (only when the divergence holds later-phase content or a
/// demanded later result, i.e. `ifOp` is set), and a reach link gating
/// everything after the divergence. A content-free divergence needs no cf
/// and no arm scan; its activation-ending terminators are collected
/// directly, however deeply nested, and each pushes a record whose reach
/// link is the empty NOT_REACHED tag.
struct Divergence {
  /// The branch, when the divergence needs a cf choice; null for a
  /// content-free divergence.
  IfOp ifOp;
  /// One arm per region of `ifOp`, in region order; empty when `ifOp` is
  /// null.
  SmallVector<Arm> arms;
  /// The activation-ending terminators of a content-free divergence.
  SmallVector<Operation *> exitTerminators;
  /// Whether some path through the divergence falls through to the op's next
  /// sibling (`!PhaseAnalysis::diverges`). A divergence with no fall-through
  /// cuts its plan's spine: everything after it is structurally unreachable
  /// and contributes no push site.
  bool fallsThrough = true;
  /// The cf choice the rebuilt `if` delivers as its last result on
  /// fall-through paths; exit paths build their own cf value in place.
  mlir::Value cfValue;
  /// The divergence `if`'s demanded later-phase results (phase > p). A
  /// phase-p branch cannot deliver a later value, so the rebuilt `if` drops
  /// these and the receiving cf pick reconstructs them from the arms' yield
  /// operands. The values held here are throw-away placeholders standing in
  /// for the original results between the `if` rebuild and the cf pick
  /// build; the pick build replaces and erases each one.
  SmallVector<Value, 0> laterResults;

  /// Whether the divergence carries a cf slot in the record.
  bool needsCf() const { return static_cast<bool>(ifOp); }
};

/// One step of the path from a plan root to an activation-ending push site:
/// the divergence the path enters and the arm it takes. The final step's arm
/// index is unused when the exiting divergence is content-free.
struct PathStep {
  size_t divIdx;
  size_t armIdx;
};

/// An activation-ending terminator together with the path of divergence arms
/// leading to it; the record for the activation is built in place right
/// before the terminator.
struct PushSite {
  Operation *terminator;
  SmallVector<PathStep> path;
};

/// The recursive decomposition of a block — a loop body or a divergence arm —
/// into content segments separated by exit-carrying divergences. `segments`
/// holds the later-phase ops between consecutive divergences in program order
/// (one more entry than `divergences`); `segFields` are the crossing values
/// each segment samples, deduplicated against every enclosing level.
/// `terminal` is the block terminator that ends the fully-continued path: a
/// loop body's back-edge continue/break (or unreachable), or an arm's own
/// yield/break/continue.
struct BodyPlan {
  Operation *terminal = nullptr;
  SmallVector<SmallVector<Operation *>> segments;
  SmallVector<llvm::SetVector<Value>> segFields;
  SmallVector<Divergence, 0> divergences;
};

/// Sorts the content of a single `uir.func` into per-phase `uir.phase` ops
/// and maintains the phase map for every op it erases or creates.
struct PhaseSorter {
  PhaseSorter(FuncOp funcOp, PhaseAnalysis &analysis, SymbolTable &symbolTable)
      : funcOp(funcOp), analysis(analysis), symbolTable(symbolTable) {}

  LogicalResult run();

private:
  FuncOp funcOp;
  PhaseAnalysis &analysis;
  SymbolTable &symbolTable;

  LogicalResult rejectSortedVocabulary();
  LogicalResult checkCoverage();
  LogicalResult dissolveExprsAndPins();
  LogicalResult decomposeCalls();
  LogicalResult decomposeCall(CallOp callOp);
  LogicalResult forwardNeverValues();
  LogicalResult extractBranches(bool &changed);
  LogicalResult extractFromBranch(Operation *op, int16_t p, bool &changed);
  LogicalResult extractFrames(bool &changed);
  LogicalResult extractFromFrame(Operation *frameOp, int16_t p, bool &changed);
  FailureOr<LoopOp> deliverBreakValues(LoopOp loopOp, int16_t p);
  LogicalResult forwardSingleProviderResults(bool &changed);
  LogicalResult hoistEarlierContent(bool &changed);
  SmallVector<Value> collectResultProviders(Operation *op, unsigned idx);
  void forwardResults(Operation *op, ArrayRef<Value> forwardTo, int16_t p);
  LogicalResult scanBlock(Block &block, int16_t p, BodyPlan &plan,
                          bool &deferred);
  void collectPlanFields(Operation *frameOp, int16_t p, BodyPlan &plan,
                         ArrayRef<Operation *> allMoved,
                         llvm::SetVector<Value> visible);
  static void collectPlanContent(BodyPlan &plan,
                                 SmallVectorImpl<Operation *> &content);
  LogicalResult rebuildDivergences(BodyPlan &plan, int16_t p);
  LogicalResult rebuildDivergenceIf(Divergence &divergence, int16_t p);
  unsigned countResultProviders(Operation *op, unsigned idx);
  static std::optional<size_t> spinePosition(BodyPlan &plan, Value value);
  void appendTail(BodyPlan &plan, size_t seg, ArrayRef<PathStep> path,
                  OpBuilder &builder, Location loc, int16_t p,
                  SmallVectorImpl<Value> &out);
  static void collectPushSites(BodyPlan &plan, SmallVectorImpl<PathStep> &path,
                               SmallVectorImpl<PushSite> &sites);
  LogicalResult rejectUnimplemented();
  LogicalResult sortRegion(Region &region);

  bool hasLaterContent(Operation *root, int16_t p);
  SmallVector<Operation *> collectLaterContent(Block &block, int16_t p);
  Operation *findExitTerminator(Region &region, int16_t p);
  void collectExitTerminators(Region &region, int16_t p,
                              SmallVectorImpl<Operation *> &exits);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp, int16_t p,
                                               ArrayRef<Operation *> movedOps);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp, int16_t p,
                                               ArrayRef<Operation *> userOps,
                                               ArrayRef<Operation *> movedOps);
  void remapFields(Block &receivingBlock, const llvm::SetVector<Value> &fields);
  void cloneFloatingUses(Operation *regionOp, Block &rootBlock);

  /// Whether a phase is a real phase (not the float/never sentinel).
  static bool isRealPhase(int16_t phase) {
    return phase != PhaseAnalysis::kFloatPhase &&
           phase != PhaseAnalysis::kNeverPhase;
  }

  /// The real phase of an op, or nullopt for the sentinels.
  std::optional<int16_t> realPhaseOf(Operation *op) {
    auto it = analysis.opPhases.find(op);
    assert(it != analysis.opPhases.end() && "op phase not computed");
    if (!isRealPhase(it->second))
      return std::nullopt;
    return it->second;
  }

  /// Record the phase of a created op and all of its results.
  void setPhase(Operation *op, int16_t phase) {
    analysis.opPhases[op] = phase;
    for (auto result : op->getResults())
      analysis.actualPhase[result] = phase;
  }

  /// Drop the phase map entries of an op about to be erased. Op and value
  /// pointers can be reused by later allocations, so stale entries must
  /// never linger.
  void forget(Operation *op) {
    analysis.opPhases.erase(op);
    for (auto result : op->getResults())
      analysis.actualPhase.erase(result);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Input Vocabulary Check
//===----------------------------------------------------------------------===//

/// The sorted-phase structure ops are this pass's output vocabulary: they are
/// only ever produced by sorting itself, and their presence in the input means
/// the pipeline is out of order. Reject them loudly instead of treating them as
/// opaque content, which would silently mis-sort their phase structure (the
/// phase analysis does not model them). `uir.unreachable_value` is not among
/// them: it is legal input marking a control-flow slot no path provides, and
/// analysis models it at the bottom phase `never`.
LogicalResult PhaseSorter::rejectSortedVocabulary() {
  auto result = funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<PhaseOp, PhaseReturnOp, PhaseCallOp>(op))
      return WalkResult::advance();
    op->emitOpError() << "is phase-sorting output vocabulary and cannot "
                         "appear in phase-sorting input";
    return WalkResult::interrupt();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Coverage Check
//===----------------------------------------------------------------------===//

/// Sorting relies on total phase-analysis coverage: an op without a phase
/// entry would be invisible to placement and silently mis-sorted. The
/// analysis stamps everything it models — including dead pure content, which
/// gets the float sentinel — so a missing entry means exactly that the op
/// sits inside a region the analysis does not model. In particular, sorted
/// vocabulary (`uir.phase` interiors) and `hir.replicate` interiors are
/// illegal in the input; `hir.pick` and `hir.choice` are analyzable
/// single-phase CF and pass. Abort loudly before mutating anything.
LogicalResult PhaseSorter::checkCoverage() {
  auto coverage = funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == funcOp || analysis.opPhases.contains(op))
      return WalkResult::advance();
    op->emitOpError() << "is not covered by phase analysis; cannot sort ops "
                         "the analysis does not model";
    return WalkResult::interrupt();
  });
  return failure(coverage.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Dissolve Phase Grouping
//===----------------------------------------------------------------------===//

/// Inline all `uir.expr` bodies and replace all `uir.pin` outputs with their
/// inputs, in both the signature and body regions. Walks post-order so inner
/// exprs dissolve before their parents.
LogicalResult PhaseSorter::dissolveExprsAndPins() {
  // Collect first: splicing while walking invalidates the walk.
  SmallVector<ExprOp> exprs;
  SmallVector<PinOp> pins;
  funcOp->walk([&](Operation *op) {
    if (auto exprOp = dyn_cast<ExprOp>(op))
      exprs.push_back(exprOp);
    else if (auto pinOp = dyn_cast<PinOp>(op))
      pins.push_back(pinOp);
  });

  for (auto exprOp : exprs) {
    auto &block = exprOp.getBody().front();
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp)
      return exprOp.emitOpError()
             << "cannot be dissolved: body is terminated by "
             << block.getTerminator()->getName() << " instead of a yield";
    for (auto [result, value] :
         llvm::zip(exprOp.getResults(), yieldOp.getValues()))
      result.replaceAllUsesWith(value);
    forget(yieldOp);
    yieldOp.erase();
    exprOp->getBlock()->getOperations().splice(exprOp->getIterator(),
                                               block.getOperations());
    forget(exprOp);
    exprOp.erase();
  }

  for (auto pinOp : pins) {
    for (auto [output, input] :
         llvm::zip(pinOp.getOutputs(), pinOp.getInputs()))
      output.replaceAllUsesWith(input);
    forget(pinOp);
    pinOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Decompose Calls
//===----------------------------------------------------------------------===//

/// The callee's declaration-derived externally visible phases: its declared
/// argument and result phases plus phase 0, in the callee's own frame, sorted
/// and deduplicated. This mirrors the `uir.phase_call` verifier, so every
/// entry the decomposition emits validates against the same set. A callee may
/// be a `uir.func` (the common case at sort time) or a `uir.split_func` (an
/// already-split callee); both carry the declared phases.
static LogicalResult calleeExternalPhases(CallOp callOp, Operation *calleeOp,
                                          SmallVectorImpl<int32_t> &externals) {
  ArrayRef<int32_t> argPhases, resultPhases;
  if (auto funcOp = dyn_cast<FuncOp>(calleeOp)) {
    argPhases = funcOp.getArgPhases();
    resultPhases = funcOp.getResultPhases();
  } else if (auto funcOp = dyn_cast<SplitFuncOp>(calleeOp)) {
    argPhases = funcOp.getArgPhases();
    resultPhases = funcOp.getResultPhases();
  } else {
    return callOp.emitOpError()
           << "callee @" << callOp.getCallee()
           << " is neither a 'uir.func' nor a 'uir.split_func'";
  }
  externals.push_back(0);
  externals.append(argPhases.begin(), argPhases.end());
  externals.append(resultPhases.begin(), resultPhases.end());
  llvm::sort(externals);
  externals.erase(std::unique(externals.begin(), externals.end()),
                  externals.end());
  return success();
}

/// Decompose every `uir.call` in the function into `uir.phase_call` entries.
LogicalResult PhaseSorter::decomposeCalls() {
  SmallVector<CallOp> calls;
  funcOp->walk([&](CallOp op) { calls.push_back(op); });
  for (auto callOp : calls)
    if (failed(decomposeCall(callOp)))
      return failure();
  return success();
}

/// Decompose one `uir.call` into per-phase `uir.phase_call` entries.
///
/// The callee's externally visible phases come from its declaration alone —
/// no split witness, no callee-before-caller order, so recursion decomposes
/// like any other call. The call's own analysis phase is the single frame
/// shift shared by all entries: an entry for the callee-frame phase `e` lands
/// at caller phase `shift + e`. Each entry carries the call's arguments and
/// results filtered to `e` (via the call's per-argument and per-result phase
/// attributes, which are the callee's declared phases), and consecutive
/// entries chain through an opaque context value in ascending phase order.
/// The entries are created as ordinary ops at the call site; sorting then
/// wraps each into its phase's `uir.phase` op and threads the context across
/// phases like any other crossing.
LogicalResult PhaseSorter::decomposeCall(CallOp callOp) {
  // The frame shift is the call's own analysis phase. A call is
  // side-effecting, so it is always anchored at a real block phase.
  auto shiftIt = analysis.opPhases.find(callOp);
  assert(shiftIt != analysis.opPhases.end() && "call phase not computed");
  int16_t shift = shiftIt->second;
  assert(isRealPhase(shift) && "side-effecting call must have a real phase");

  auto *calleeOp = symbolTable.lookup(callOp.getCallee());
  assert(calleeOp && "call to undeclared callee survived verification");
  SmallVector<int32_t> externals;
  if (failed(calleeExternalPhases(callOp, calleeOp, externals)))
    return failure();
  assert(!externals.empty() && "external phases always contain phase 0");

  // Partition the call's arguments and results by their callee-frame phase.
  // The lists preserve program order, keeping the entries deterministic.
  std::map<int32_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
  for (auto [arg, typeOfArg, phase] :
       llvm::zip(callOp.getArguments(), callOp.getTypeOfArgs(),
                 callOp.getArgPhases())) {
    phaseArgs[phase].push_back(arg);
    phaseTypeOfArgs[phase].push_back(typeOfArg);
  }
  std::map<int32_t, SmallVector<std::pair<unsigned, Value>>> phaseResults;
  for (auto [index, phase] : llvm::enumerate(callOp.getResultPhases()))
    phaseResults[phase].push_back({index, callOp.getTypeOfResults()[index]});

  // Create one entry per external phase, ascending, chaining opaque context.
  OpBuilder builder(callOp);
  auto loc = callOp.getLoc();
  auto anyTy = hir::AnyType::get(callOp.getContext());
  int32_t maxPhase = externals.back();
  Value prevContext;
  SmallVector<Value> replacements(callOp.getNumResults());
  for (int32_t phase : externals) {
    auto &results = phaseResults[phase];
    SmallVector<Value> typeOfResults;
    SmallVector<Type> resultTypes;
    for (auto &[index, typeOfResult] : results) {
      typeOfResults.push_back(typeOfResult);
      resultTypes.push_back(anyTy);
    }
    Type contextOutType = phase != maxPhase ? anyTy : Type();
    auto entry = PhaseCallOp::create(
        builder, loc, resultTypes, contextOutType, callOp.getCallee(), phase,
        phaseArgs[phase], phaseTypeOfArgs[phase], typeOfResults, prevContext);
    setPhase(entry, shift + static_cast<int16_t>(phase));
    for (auto [i, pair] : llvm::enumerate(results))
      replacements[pair.first] = entry.getResult(i);
    prevContext = entry.getContextOut();
  }

  assert(llvm::all_of(replacements, [](Value v) { return !!v; }) &&
         "every call result has an external phase entry");
  callOp->replaceAllUsesWith(replacements);
  forget(callOp);
  callOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Forward Never Values
//===----------------------------------------------------------------------===//
//
// # Replace Real-Phase Uses of Never Values with `uir.unreachable_value`
//
// A `never`-phased value is the result of a control-flow op that no path ever
// provides (a loop without breaks, an `if` whose providing paths all exit).
// The value never materializes at runtime, so every consumer is unreachable.
// `never` is deliberately non-viral: analysis assigns it as a bottom that
// satisfies any demand and never propagates an unreachable-ness to consumers,
// so a real-phase op consuming a `never` value keeps its own real phase.
//
// Sorting must not carry a `never` value across a phase boundary: the value
// stays on its producer, but the wrap and crossing rules would otherwise see a
// use whose value phase (`never`) sits below every real phase, and a
// backward-crossing or a spurious yield would result. This step severs every
// use ahead of any placement: for each operand carrying a `never` value, it
// materializes a fresh `uir.unreachable_value` right before the consumer, at
// the phase where sorting will attribute that use, and rewrites the operand to
// it. The `never` result is left on its producer, entirely unused. The
// invariant established: no op operand carries a `never` value, so wrapping
// and crossing never encounter one, and a floating consumer keeps no
// reference into the phase region its producer wraps into (floats are never
// placed, so such a reference could not dominate).
//
// The use-attribution phase mirrors `sortRegion`: every op consumes its
// operands at its own phase — the float sentinel for a floating (dead pure)
// consumer, whose stand-in then floats alongside it — except `uir.return`,
// which distributes its value and type operands onto the `uir.phase_return`
// of each result's declared phase, so a return operand is consumed at that
// result's declared phase, not the return op's own (always 0).

/// Replace every use of a `never`-phased value with a fresh
/// `uir.unreachable_value` at the consuming phase. See the section header.
LogicalResult PhaseSorter::forwardNeverValues() {
  auto resultPhases = funcOp.getResultPhases();

  // The phase at which sorting attributes operand `operand` of op `op`: the
  // op's own phase, except a `uir.return` fans each value/type operand onto
  // the phase_return of that result's declared phase.
  auto consumerPhase = [&](Operation *op, OpOperand &operand) -> int16_t {
    if (auto returnOp = dyn_cast<ReturnOp>(op)) {
      unsigned numValues = returnOp.getValues().size();
      unsigned index = operand.getOperandNumber();
      unsigned slot = index < numValues ? index : index - numValues;
      return static_cast<int16_t>(resultPhases[slot]);
    }
    return analysis.opPhases.at(op);
  };

  // Collect first: materializing ops while walking invalidates the walk.
  SmallVector<Operation *> ops;
  funcOp->walk([&](Operation *op) {
    if (op != funcOp)
      ops.push_back(op);
  });

  auto anyTy = hir::AnyType::get(funcOp.getContext());
  for (auto *op : ops) {
    // One unreachable value per consuming phase suffices for all of this op's
    // never operands that land there. Keyed by std::map: the float sentinel
    // is not a representable DenseMap key.
    std::map<int16_t, Value> perPhase;
    for (auto &operand : op->getOpOperands()) {
      auto it = analysis.actualPhase.find(operand.get());
      assert(it != analysis.actualPhase.end() &&
             "op operand has no computed phase");
      if (it->second != PhaseAnalysis::kNeverPhase)
        continue;
      int16_t phase = consumerPhase(op, operand);
      auto &slot = perPhase[phase];
      if (!slot) {
        OpBuilder builder(op);
        auto uv =
            UnreachableValueOp::create(builder, operand.get().getLoc(), anyTy);
        setPhase(uv, phase);
        slot = uv.getResult();
      }
      operand.set(slot);
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Branch Extraction
//===----------------------------------------------------------------------===//
//
// # Extract Later-Phase Content Out of Earlier-Phase Branches
//
// A branch — a `uir.if` or an `hir.pick` — at phase p whose arms hold
// later-phase content, or whose result selects between later-phase values,
// cannot wrap wholesale into
// phase p: the later-phase content must run in a later phase, and a later
// value cannot be produced at phase p (causality). Branch extraction rewrites
// such a branch into two cooperating ops that the ordinary wrap and cross
// rules then place:
//
// - The branch is rebuilt at phase p, dispatching on the same operand (an
//   if's condition, a pick's selector). Its kept results (those already at
//   phase p) stay, and it gains one extra phase-p result per arm: an
//   `hir.choice` tagging the taken arm and carrying, as its payload, the
//   arm-local values the moved content samples — including a pick
//   alternative's own block arguments, which are ordinary arm-local phase-p
//   values. Later-phase results are removed from the branch.
// - One `hir.pick` is created right after the branch, at phase p+1,
//   dispatching on the choice. Each alternative holds that arm's moved
//   later-phase content, reconstructing the payload from block arguments,
//   and yields the arm's contribution to the removed later-phase results.
//
// The two branch kinds differ only structurally: an if's arms are its then
// and else regions (an absent else is materialized so the not-taken tag has
// a constructor) and its continuation terminator is `uir.yield` with paired
// value/type operands, while a pick's arms are its alternatives (always
// present, with payload block args) and its continuation terminator is a
// plain `hir.yield`; the if carries one type operand per result, the pick
// none. Everything else — result partition, crossing collection, choice
// construction, and the receiving pick — is shared.
//
// The choice is an ordinary phase-p value; the cross rule yields it out of
// phase p and the pick consumes it in phase p+1 like any other crossing. The
// distinction between an obligation to choice-ify and the neighbouring
// divergent-result cases is local: a later-phase result choice-ifies whenever a
// reaching arm provides it — the values' identity is irrelevant, and whether
// several arms provide it (a genuine selection) or a single reaching arm does
// while every other arm provably diverges (an unreachable terminator or a
// breakless loop). External single-provider results were already forwarded
// away, so the remaining single-provider case is an interior value that varies
// per path and must choice-ify. The diverging arms yield
// `uir.unreachable_value` in every slot and become dead receiving alternatives
// the choice never selects; their own later-phase content moves into those
// alternatives all the same — reachability never gates what is sorted, and the
// content can never run (entering the arm hangs the branch's phase, so later
// phases never start), so the alternative is harmless dead code. A
// never-produced result (the whole op diverges) is not extracted; its
// consumers were already rewritten to `uir.unreachable_value` by
// never-forwarding, so it is simply left unused.

/// Whether any op transitively inside `root` sits at a real phase later than
/// p. A sentinel-phase op is a floating region-free pure value (a constant or
/// arithmetic on constants); it has no regions and contributes no later
/// content, so the walk skips it.
bool PhaseSorter::hasLaterContent(Operation *root, int16_t p) {
  auto result = root->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    auto ph = realPhaseOf(op);
    if (!ph)
      return WalkResult::skip();
    if (*ph > p)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// The top-level ops of a block at a real phase later than p, in program
/// order.
SmallVector<Operation *> PhaseSorter::collectLaterContent(Block &block,
                                                          int16_t p) {
  SmallVector<Operation *> ops;
  for (auto &op : block)
    if (auto ph = realPhaseOf(&op); ph && *ph > p)
      ops.push_back(&op);
  return ops;
}

/// The first terminator at phase <= p that exits the region — a break/continue
/// targeting a loop outside the region — or null if every path yields. Branch
/// extraction bails on arms with such exits: they belong to the enclosing
/// loop's records, not to branch extraction.
Operation *PhaseSorter::findExitTerminator(Region &region, int16_t p) {
  Operation *found = nullptr;
  std::function<void(Region &, bool)> visit = [&](Region &r, bool loopScoped) {
    for (auto &block : r) {
      for (auto &op : block) {
        if (found)
          return;
        bool isExit = !loopScoped && isa<BreakOp, ContinueOp>(op);
        if (isExit)
          if (auto ph = realPhaseOf(&op); ph && *ph <= p) {
            found = &op;
            return;
          }
        for (auto &inner : op.getRegions())
          visit(inner, loopScoped || isa<LoopOp>(op));
      }
    }
  };
  visit(region, isa<LoopOp>(region.getParentOp()));
  return found;
}

/// Collect all terminators at phase <= p inside the region that exit out of
/// the region instead of yielding, in walk order. A break/continue targeting a
/// loop outside the region exits it; break/continue inside a nested loop
/// (including the queried region itself when it belongs to a loop) target that
/// loop and do not exit.
void PhaseSorter::collectExitTerminators(Region &region, int16_t p,
                                         SmallVectorImpl<Operation *> &exits) {
  std::function<void(Region &, bool)> visit = [&](Region &r, bool loopScoped) {
    for (auto &block : r) {
      for (auto &op : block) {
        bool isExit = !loopScoped && isa<BreakOp, ContinueOp>(op);
        if (isExit)
          if (auto ph = realPhaseOf(&op); ph && *ph <= p)
            exits.push_back(&op);
        for (auto &inner : op.getRegions())
          visit(inner, loopScoped || isa<LoopOp>(op));
      }
    }
  };
  visit(region, isa<LoopOp>(region.getParentOp()));
}

/// Collect the real-phase values defined inside `regionOp` that the
/// `movedOps` (and their nested ops) use and that are not themselves among
/// the moved ops: these cross the phase boundary as choice payload fields.
/// Values defined outside the region are used directly (they dominate the
/// receiving pick); floating values defined inside are cloned instead of
/// crossing.
llvm::SetVector<Value>
PhaseSorter::collectCrossingValues(Operation *regionOp, int16_t p,
                                   ArrayRef<Operation *> movedOps) {
  return collectCrossingValues(regionOp, p, movedOps, movedOps);
}

/// Collect the values defined inside `regionOp` that the `userOps` use and
/// that are not themselves produced by any of the `movedOps`: these cross the
/// phase boundary and become record fields / choice payloads. `movedOps` lists
/// everything that leaves the region; with partial activations it is more than
/// the `userOps` whose uses are being collected here (a segment's uses are
/// collected while all segments and exit paths count as moved).
llvm::SetVector<Value>
PhaseSorter::collectCrossingValues(Operation *regionOp, int16_t p,
                                   ArrayRef<Operation *> userOps,
                                   ArrayRef<Operation *> movedOps) {
  auto isMoved = [&](Operation *op) {
    return llvm::any_of(movedOps, [&](Operation *m) {
      return m == op || m->isProperAncestor(op);
    });
  };

  llvm::SetVector<Value> fields;
  for (auto *moved : userOps) {
    moved->walk([&](Operation *user) {
      for (auto value : user->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          auto *owner = blockArg.getOwner()->getParentOp();
          if (owner == regionOp || regionOp->isProperAncestor(owner))
            if (!isMoved(owner))
              fields.insert(value);
          continue;
        }
        auto *defOp = value.getDefiningOp();
        if (!regionOp->isProperAncestor(defOp) || isMoved(defOp))
          continue;
        // Floating values are cloned into the receiving block instead of
        // crossing through a field.
        if (!isRealPhase(analysis.opPhases.at(defOp)))
          continue;
        // A record field is a phase-p snapshot crossing to phase p+1; a value
        // at a later phase never crosses as a field. Such a value is produced
        // and consumed entirely within the receiving phase — here, a
        // divergence result the arm-pick reconstructs — so it is left for that
        // reconstruction rather than fielded.
        if (auto it = analysis.actualPhase.find(value);
            it != analysis.actualPhase.end() && isRealPhase(it->second) &&
            it->second > p)
          continue;
        fields.insert(value);
      }
    });
  }
  return fields;
}

/// After moving content into a receiving block whose leading arguments
/// correspond 1:1 to `fields`, replace field uses inside the block (and any
/// nested receiving blocks) with the block arguments.
void PhaseSorter::remapFields(Block &receivingBlock,
                              const llvm::SetVector<Value> &fields) {
  auto *receivingRegion = receivingBlock.getParent();
  for (auto [field, arg] : llvm::zip(fields, receivingBlock.getArguments())) {
    Value fieldValue = field;
    analysis.actualPhase[arg] = analysis.actualPhase.at(fieldValue);
    fieldValue.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return receivingRegion->isAncestor(use.getOwner()->getParentRegion());
    });
  }
}

// NOLINTBEGIN(misc-no-recursion)

/// Clone the defining tree of a floating pure value at the builder's
/// insertion point. Only defs inside `regionOp` are cloned; values defined
/// outside remain visible from the receiving block and are used directly.
/// Every clone is recorded in `clonedOps` so the caller can stamp it.
static Value cloneFloatingTree(Operation *regionOp, Value value,
                               OpBuilder &builder, IRMapping &mapping,
                               SmallVectorImpl<Operation *> &clonedOps) {
  auto *defOp = value.getDefiningOp();
  if (!defOp || !regionOp->isProperAncestor(defOp))
    return value;
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;
  for (auto operand : defOp->getOperands())
    cloneFloatingTree(regionOp, operand, builder, mapping, clonedOps);
  clonedOps.push_back(builder.clone(*defOp, mapping));
  return mapping.lookup(value);
}

// NOLINTEND(misc-no-recursion)

/// Clone floating trees that moved content still references from inside the
/// (former) region of `regionOp`. A floating value defined inside an arm is
/// not visible from the receiving pick once the arm wraps into its own phase,
/// so it is duplicated in front of its user; any other remaining use of a
/// region-interior value is a causality violation the analysis rejects.
///
/// A floating value is a region-free pure tree — constants and arithmetic on
/// them: every frame now ends with a real phase, so no dead region-bearing
/// def survives here to be rematerialized.
void PhaseSorter::cloneFloatingUses(Operation *regionOp, Block &rootBlock) {
  rootBlock.walk([&](Operation *user) {
    for (auto &use : user->getOpOperands()) {
      auto *defOp = use.get().getDefiningOp();
      if (!defOp || !regionOp->isProperAncestor(defOp))
        continue;
      assert(!isRealPhase(analysis.opPhases.at(defOp)) &&
             hir::isEffectivelyPure(defOp) && defOp->getNumRegions() == 0 &&
             "moved op uses non-floating, non-crossing value from region");
      OpBuilder builder(user);
      IRMapping mapping;
      SmallVector<Operation *> clonedOps;
      auto clone =
          cloneFloatingTree(regionOp, use.get(), builder, mapping, clonedOps);
      for (auto *clonedOp : clonedOps)
        clonedOp->walk([&](Operation *cloned) {
          setPhase(cloned, PhaseAnalysis::kFloatPhase);
        });
      use.set(clone);
    }
  });
}

/// The arm regions of a branch-shaped op: a `uir.if`'s then and else regions
/// (the else possibly empty), or an `hir.pick`'s alternatives (always
/// non-empty, one block each). Extraction and forwarding treat the arms of
/// both branch kinds uniformly through this helper.
static SmallVector<Region *> branchArms(Operation *op) {
  if (auto ifOp = dyn_cast<IfOp>(op))
    return {&ifOp.getThenRegion(), &ifOp.getElseRegion()};
  SmallVector<Region *> arms;
  for (auto &region : cast<hir::PickOp>(op).getAlternatives())
    arms.push_back(&region);
  return arms;
}

/// The values an arm's continuation terminator provides for the parent
/// branch's results: the paired values of a `uir.yield` (if arms) or the
/// plain operands of an `hir.yield` (pick alternatives). Nullopt for any
/// other terminator — an exit (return/break/continue) or a dead path
/// (`uir.unreachable`) provides nothing.
static std::optional<ValueRange> armProvidedValues(Operation *terminator) {
  if (auto yieldOp = dyn_cast<YieldOp>(terminator))
    return ValueRange(yieldOp.getValues());
  if (auto yieldOp = dyn_cast<hir::YieldOp>(terminator))
    return ValueRange(yieldOp.getOperands());
  return std::nullopt;
}

/// The number of arms providing result `idx` of a branch-shaped op (`uir.if`
/// or `hir.pick`): the arms whose continuation yield is structurally reachable.
/// An arm terminated by an exit or a dead `uir.unreachable` is not a carrier
/// and provides nothing; provider-ness is otherwise a control-flow property, so
/// a reachable arm provides its slot whatever it yields — a
/// `uir.unreachable_value` poison included. A later-phase result is a
/// choice-ification obligation whenever a reaching arm provides it, whether
/// more than one arm does (a genuine selection) or a single reaching arm does
/// while every other arm provably diverges; both reconstruct in the receiving
/// pick. Zero providers is a never result — the whole op diverges — and is not
/// choice-ified.
unsigned PhaseSorter::countResultProviders(Operation *op, unsigned idx) {
  unsigned count = 0;
  for (auto *region : branchArms(op)) {
    if (region->empty())
      continue;
    auto *term = region->front().getTerminator();
    if (armProvidedValues(term) && analysis.reaches(term, op))
      ++count;
  }
  return count;
}

/// Extract later-phase content out of every phase-p branch (`uir.if` or
/// `hir.pick`) whose interior or results demand it, innermost first — in one
/// walk across both op kinds — so a nested branch becomes ordinary
/// later-phase content of its enclosing branch regardless of kind. Sets
/// `changed` when it rewrites any branch, so the caller's joint extraction
/// fixpoint can re-run it.
///
/// A branch is extracted only when every one of its results is deliverable at
/// its own phase after extraction: results already at phase p stay, and
/// results at a later phase are removed and reconstructed by the pick whenever
/// a reaching arm provides them. External single-provider results were
/// forwarded away before extraction, so a surviving later-phase result
/// choice-ifies whether several arms provide it (a genuine selection) or a
/// single reaching arm provides it while the others provably diverge (a single
/// interior provider). A never-produced result is not extracted — the whole op
/// diverges, its consumers were already rewritten to `uir.unreachable_value`,
/// so it is simply left unused.
///
/// Extraction applies to its own output: a receiving `hir.pick` is an ordinary
/// divergence, so if its reconstructed result still lives later than its own
/// phase p+1 — the branch crossed more than one phase — the same rule
/// choice-ifies it again at the p+1-to-p+2 boundary. This single sweep handles
/// one boundary of every chain; the caller re-runs it (alongside loop-record
/// extraction) until nothing changes, so a result crosses any number of phases.
/// Each step moves a result's delivery exactly one phase forward and strictly
/// reduces its phase distance, bounding the fixpoint. A receiving pick a
/// *sibling* rule emitted — a record replay's cf or reach pick inside a
/// `hir.replicate`, stamped by the sorter as it was built — is
/// indistinguishable from a user pick here; all are picked up by the same walk
/// with no provenance test.
LogicalResult PhaseSorter::extractBranches(bool &changed) {
  SmallVector<Operation *> branchOps;
  funcOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, hir::PickOp>(op) && realPhaseOf(op))
      branchOps.push_back(op);
  });

  for (auto *op : branchOps) {
    int16_t p = *realPhaseOf(op);

    // Classify the results. Extraction is viable only if none diverge except
    // by choice-ification.
    bool extractable = true;
    bool hasChoiceResult = false;
    for (auto result : op->getResults()) {
      int16_t vph = analysis.actualPhase.at(result);
      if (vph == p)
        continue; // Kept at the op's phase.
      if (isRealPhase(vph) && vph > p) {
        // A later-phase result choice-ifies whenever a reaching arm provides
        // it. The whole op is non-diverging (a real later phase implies some
        // arm rejoins), so the provider count is at least one: greater than
        // one for a genuine selection, exactly one when a single reaching arm
        // delivers the result while every other arm provably diverges. Both
        // reconstruct in the receiving pick, so both extract; the diverging
        // arms become dead receiving alternatives the choice never selects.
        assert(countResultProviders(op, result.getResultNumber()) >= 1 &&
               "a later-phase result must have a reaching provider");
        hasChoiceResult = true;
        continue;
      }
      // A never result (zero reaching providers; the whole op diverges) is
      // left unused after never-forwarding, its consumers already
      // `uir.unreachable_value`, so the op is not extracted. No other phase
      // reaches here: single-provider float and earlier-phase results were
      // forwarded away (after hoisting, for anchored interior providers), and
      // a multi-provider result is floored at the op's phase.
      assert(vph == PhaseAnalysis::kNeverPhase &&
             "mismatched branch result survived forwarding and hoisting");
      extractable = false;
      break;
    }
    if (!extractable)
      continue;
    if (!hasChoiceResult && !hasLaterContent(op, p))
      continue; // Nothing to extract; wraps wholesale.

    if (failed(extractFromBranch(op, p, changed)))
      return failure();
  }
  return success();
}

/// Rewrite one phase-p branch (`uir.if` or `hir.pick`) into a choice-yielding
/// branch plus a receiving `hir.pick` at phase p+1. See the section header
/// for the shape and the invariant. Sets `changed` when it rewrites the branch;
/// the bail cases (an arm exiting to an enclosing loop, or no reaching arm at
/// all) leave the branch untouched and do not set it, so the caller's fixpoint
/// terminates.
LogicalResult PhaseSorter::extractFromBranch(Operation *op, int16_t p,
                                             bool &changed) {
  auto ifOp = dyn_cast<IfOp>(op);
  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());

  // Whether an arm's continuation yield is reachable — a control-flow property:
  // the terminator is a yield/hir.yield and control provably reaches it. A
  // non-reaching arm provably diverges at phase p (a `uir.unreachable`
  // terminator, or a breakless loop before its yield), so its choice tag is
  // never produced and its receiving alternative is dead. An absent else region
  // falls through, hence reaches.
  auto armReaches = [&](Region *arm) {
    if (arm->empty())
      return true;
    auto *term = arm->front().getTerminator();
    return armProvidedValues(term).has_value() && analysis.reaches(term, op);
  };

  // An arm that exits to an enclosing loop belongs to that loop's records, so
  // it leaves the branch for that machinery. A branch with no reaching arm
  // never rejoins, so there is no choice to dispatch on; it is left for the
  // nested-content rejection. Otherwise the branch extracts: reachability
  // never gates what is sorted, so a diverging arm's later-phase content
  // extracts exactly like any other arm's — its receiving alternative is
  // simply dead at runtime, because the arm's tag is never delivered and an
  // execution entering the arm hangs at phase p, so the later phase never
  // runs at all. Harmless dead code, no miscompile — and divergence cannot
  // change how the code around it compiles.
  //
  // Later-phase content buried inside a same- or earlier-phase region op at an
  // arm's top level (a nested frame's record content, or a nested branch that
  // itself deferred) is not this branch's to lift: the owning rule lifts it to
  // the arm's top level first — a frame deposits its receiving replicate right
  // after itself — and moving the arm's other later ops around it now would
  // tear the content's program order across sweeps and break dominance. Defer
  // the whole branch (without setting `changed`) until every later op in the
  // arm is a top-level, liftable one; content no rule ever lifts is rejected
  // once the fixpoint settles.
  bool anyReaches = false;
  for (auto *arm : branchArms(op)) {
    if (findExitTerminator(*arm, p))
      return success();
    if (!arm->empty())
      for (auto &blockOp : arm->front().without_terminator()) {
        if (auto ph = realPhaseOf(&blockOp); ph && *ph > p)
          continue;
        if (blockOp.getNumRegions() != 0 && hasLaterContent(&blockOp, p))
          return success();
      }
    if (armReaches(arm))
      anyReaches = true;
  }
  if (!anyReaches)
    return success();

  // Past every bail: the branch is rewritten below.
  changed = true;

  // Materialize an else region for an else-less if so the not-taken tag has a
  // constructor; pick alternatives always exist.
  if (ifOp && ifOp.getElseRegion().empty()) {
    auto &elseBlock = ifOp.getElseRegion().emplaceBlock();
    auto builder = OpBuilder::atBlockEnd(&elseBlock);
    auto yield = YieldOp::create(builder, loc, ValueRange{}, ValueRange{});
    setPhase(yield, p);
  }
  SmallVector<Region *> arms = branchArms(op);

  // Partition the results into kept (at phase p) and later ones. Later
  // results are reconstructed by the receiving pick.
  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : op->getResults()) {
    if (analysis.actualPhase.at(result) > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }

  // Per arm: the moved later-phase content, the crossing values it samples,
  // and the operands the old yield provided for the later results.
  struct ArmInfo {
    SmallVector<Operation *> moved;
    llvm::SetVector<Value> fields;
    SmallVector<Value> laterYieldOperands;
  };
  SmallVector<ArmInfo> armInfos;
  for (auto *arm : arms) {
    ArmInfo info;
    info.moved = collectLaterContent(arm->front(), p);
    info.fields = collectCrossingValues(op, p, info.moved);

    if (!armReaches(arm)) {
      // A non-reaching arm provably diverges; it provides nothing, so its
      // receiving alternative yields `uir.unreachable_value` for every later
      // result. Its own later-phase content still moves into that alternative
      // like any other arm's, and the crossings it samples ride the payload
      // fields — all dead at runtime, since the arm's tag is never delivered.
      info.laterYieldOperands.assign(laterIdx.size(), Value());
      armInfos.push_back(std::move(info));
      continue;
    }

    auto providedValues = *armProvidedValues(arm->front().getTerminator());
    // A later result provided by more than one arm is a selection the receiving
    // pick reconstructs: each providing arm contributes its provider operand.
    // An arm that does not provide the slot — a non-materializing operand, e.g.
    // `uir.unreachable_value` in a slot of an N-ary `hir.pick`, which cannot
    // arise for a binary `if` since a non-provider there leaves a single
    // provider — contributes a null operand, and the receiving alternative
    // yields a fresh `uir.unreachable_value` for it.
    for (auto idx : laterIdx) {
      Value operand = providedValues[idx];
      info.laterYieldOperands.push_back(
          analysis.materializes(operand) ? operand : Value());
    }

    // The later yield operands cross the boundary too: collect their crossing
    // needs as if the yield were moved content.
    for (auto value : info.laterYieldOperands) {
      if (!value)
        continue;
      auto *defOp = value.getDefiningOp();
      if (defOp && op->isProperAncestor(defOp) &&
          !llvm::any_of(info.moved,
                        [&](Operation *m) {
                          return m == defOp || m->isProperAncestor(defOp);
                        }) &&
          isRealPhase(analysis.opPhases.at(defOp)))
        info.fields.insert(value);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto *owner = blockArg.getOwner()->getParentOp();
        if (owner == op || op->isProperAncestor(owner))
          info.fields.insert(value);
      }
    }
    armInfos.push_back(std::move(info));
  }

  // Construct the per-arm choice and rewrite each continuation terminator to
  // carry the kept values plus the choice. The choice type exists only for
  // the if kind: an if pairs every yielded value (and result) with a type
  // operand, while a pick and its `hir.yield` carry none.
  OpBuilder builder(op);
  Value choiceType;
  if (ifOp) {
    auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
    setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
    choiceType = choiceTypeOp.getResult();
  }

  for (auto [tag, arm] : llvm::enumerate(arms)) {
    auto *terminator = arm->front().getTerminator();
    OpBuilder termBuilder(terminator);

    // A non-reaching arm's choice tag is never produced, so build no real
    // choice for it. A `uir.unreachable` terminator has no yield to rebuild; a
    // yield-terminated diverging arm (a breakless loop before it) keeps exact
    // result arity, so rebuild its yield with `uir.unreachable_value` in every
    // kept slot and the choice slot.
    if (!armReaches(arm)) {
      auto poison = [&]() {
        auto uv = UnreachableValueOp::create(termBuilder, loc, anyTy);
        setPhase(uv, p);
        return uv.getResult();
      };
      if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
        SmallVector<Value> values, typeOfValues;
        for (auto idx : keptIdx) {
          values.push_back(poison());
          typeOfValues.push_back(ifOp.getResultTypes()[idx]);
        }
        values.push_back(poison());
        typeOfValues.push_back(choiceType);
        auto newYield = YieldOp::create(termBuilder, loc, values, typeOfValues);
        setPhase(newYield, p);
        forget(yieldOp);
        yieldOp.erase();
      } else if (auto hirYieldOp = dyn_cast<hir::YieldOp>(terminator)) {
        SmallVector<Value> values(keptIdx.size() + 1);
        for (auto &value : values)
          value = poison();
        auto newYield = hir::YieldOp::create(termBuilder, loc, values);
        setPhase(newYield, p);
        forget(hirYieldOp);
        hirYieldOp.erase();
      }
      continue;
    }

    auto choice = hir::ChoiceOp::create(termBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        armInfos[tag].fields.getArrayRef());
    setPhase(choice, p);

    if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
      SmallVector<Value> values, typeOfValues;
      for (auto idx : keptIdx) {
        values.push_back(yieldOp.getValues()[idx]);
        typeOfValues.push_back(yieldOp.getTypeOfValues()[idx]);
      }
      values.push_back(choice.getResult());
      typeOfValues.push_back(choiceType);
      auto newYield = YieldOp::create(termBuilder, loc, values, typeOfValues);
      setPhase(newYield, p);
    } else {
      auto hirYieldOp = cast<hir::YieldOp>(terminator);
      SmallVector<Value> values;
      for (auto idx : keptIdx)
        values.push_back(hirYieldOp.getOperands()[idx]);
      values.push_back(choice.getResult());
      auto newYield = hir::YieldOp::create(termBuilder, loc, values);
      setPhase(newYield, p);
    }
    forget(terminator);
    terminator->erase();
  }

  // Rebuild the branch with the kept results plus the choice result,
  // dispatching on the same operand; the if additionally carries its kept
  // result type operands plus the choice type.
  SmallVector<Type> newResultTypes(keptIdx.size() + 1, anyTy);
  Operation *newOp;
  if (ifOp) {
    SmallVector<Value> newTypeOperands;
    for (auto idx : keptIdx)
      newTypeOperands.push_back(ifOp.getResultTypes()[idx]);
    newTypeOperands.push_back(choiceType);
    auto newIf = IfOp::create(builder, loc, newResultTypes, ifOp.getCondition(),
                              newTypeOperands);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    newOp = newIf;
  } else {
    auto pickOp = cast<hir::PickOp>(op);
    auto newPick = hir::PickOp::create(builder, loc, newResultTypes,
                                       pickOp.getChoice(), arms.size());
    for (auto [i, arm] : llvm::enumerate(arms))
      newPick.getAlternatives()[i].takeBody(*arm);
    newOp = newPick;
  }
  setPhase(newOp, p);
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newOp->getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newOp->getResult(i));
  }
  Value choiceResult = newOp->getResult(keptIdx.size());
  analysis.actualPhase[choiceResult] = p;

  // Create the receiving pick after the branch, one alternative per arm,
  // reconstructing the later results.
  builder.setInsertionPointAfter(newOp);
  SmallVector<Type> pickResultTypes(laterIdx.size(), anyTy);
  auto receivingPick = hir::PickOp::create(builder, loc, pickResultTypes,
                                           choiceResult, arms.size());
  analysis.opPhases[receivingPick] = p + 1;
  for (auto [i, idx] : llvm::enumerate(laterIdx)) {
    analysis.actualPhase[receivingPick.getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(receivingPick.getResult(i));
  }

  for (auto [tag, alt] : llvm::enumerate(receivingPick.getAlternatives())) {
    auto &info = armInfos[tag];
    auto &altBlock = alt.emplaceBlock();
    for (unsigned i = 0; i < info.fields.size(); ++i)
      altBlock.addArgument(anyTy, loc);
    for (auto *moved : info.moved)
      moved->moveBefore(&altBlock, altBlock.end());
    OpBuilder altBuilder(&altBlock, altBlock.end());
    // A null operand is a slot this arm does not provide: yield a fresh
    // `uir.unreachable_value` at the receiving phase, whose consumers this
    // alternative's choice tag gates off.
    SmallVector<Value> altYieldValues;
    for (auto value : info.laterYieldOperands) {
      if (value) {
        altYieldValues.push_back(value);
        continue;
      }
      auto unreachable = UnreachableValueOp::create(altBuilder, loc, anyTy);
      setPhase(unreachable, static_cast<int16_t>(p + 1));
      altYieldValues.push_back(unreachable.getResult());
    }
    auto altYield = hir::YieldOp::create(altBuilder, loc, altYieldValues);
    analysis.opPhases[altYield] = p + 1;
    remapFields(altBlock, info.fields);
    cloneFloatingUses(newOp, altBlock);
  }

  forget(op);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Record Extraction
//===----------------------------------------------------------------------===//
//
// # Extract Later-Phase Content Out of Earlier-Phase Producing Frames
//
// A producing frame at phase p — a `uir.loop`, or a `hir.replicate` replaying
// an earlier extraction's records — whose body holds later-phase content
// cannot wrap wholesale into phase p: the later-phase content runs once per
// activation of the frame, so the phase-p execution must tell phase p+1 *how
// often* it is instantiated, and with which per-iteration snapshots. Record
// extraction rewrites such a frame into two cooperating constructs the wrap
// and cross rules then place:
//
// - A hit list (`hir.opaque_list_create`) is created at phase p, right before
//   the frame. Every activation packs a record of the crossing values it
//   samples and pushes it onto the list; the record is built in place at the
//   terminator that ends the activation. The list is an ordinary phase-p value
//   that crosses to phase p+1 like any other crossing.
// - One `hir.replicate` at phase p+1 replays the records, holding the frame's
//   later-phase content. It reconstructs the per-iteration snapshots from the
//   record fields.
//
// A replicate frame is the degenerate case of the same rule: its activations
// are its replays, one per incoming record, each ending at the body's terminal
// `hir.yield`, and it has no exits, so every record is a single segment. Its
// later-phase content samples the replay's per-record values — the replicate's
// unpacked block arguments and its picks' reconstructions — so a chain of
// extractions ferries per-iteration values hop by hop, each phase unpacking
// the previous list and re-packing what the next phase consumes, while
// path-dependent values ride along inside the re-recorded choices.
//
// Loop bodies with loop-phase early exits (`break`/`continue` inside branch
// arms) diverge in how much later-phase content each activation runs. The body
// decomposes into content segments separated by exit-carrying divergence
// points, and each record carries up to two slots per divergence:
//
// - The cf choice tags which arm ran and carries the arm's own record body —
//   the arm's first-segment samples plus, recursively, the arm's own cf and
//   reach slots — so arm content replays exactly once in the receiving cf
//   pick's alternative for its arm, whether the activation later exited or
//   fell through. It exists only for divergences with later-phase content or
//   a demanded later result, and is built at every arm-ending point: at an
//   activation-ending terminator inside the arm for exit paths, and at the
//   arm's yield for the fall-through, where it becomes the rebuilt branch's
//   last result. Building it inside the arm is what lets arm-interior
//   samples — out of scope at the activation-end push — still cross.
//
// - The reach link gates everything after the divergence: an empty
//   NOT_REACHED tag ends the record at an exit, and a REACHED tag carries the
//   following segment's samples plus the next divergence's slots, so the
//   record nests along program order. It is built at the push site.
//
// Every activation-ending terminator pushes exactly one record in place: the
// shared prefix segment's samples, then per divergence either its rebuilt
// branch's cf result and a REACHED link (the path fell past it) or a cf built
// bottom-up at the site and a NOT_REACHED link (the path ends here). A value
// is recorded exactly on the paths that compute it, and the receiving replay
// — a cf pick and a reach pick per level, data-only, with no exit
// terminators — reads each field exactly on the paths that replay its
// consumers, guaranteed by SSA dominance in the source: any consumer of a
// value sits on paths that computed it, and the replay preserves the path
// structure.
//
// The decomposition is recursive and uniform: every arm of a divergence is a
// block like the loop body and is scanned into its own sub-plan, so a nested
// exit-carrying branch inside an arm is another divergence that deepens the
// cf payload and the receiving picks by one level.
//
// When a divergence's demanded results are at a later phase, the rebuilt
// branch cannot deliver them (a phase-p branch produces no later value), so
// they are dropped and reconstructed in the receiving phase: the cf pick is
// valued, each alternative yielding the arm's operand for the result, and the
// pick's results stand in for the values in the following segments. An arm
// that does not provide a result (short yield, or an arm that never rejoins)
// yields `uir.unreachable_value` — its consumers replay only on paths the
// reach links gate off.
//
// A demanded later result an arm provides is reconstructed at whichever level
// defines it, threaded up to the receiving cf pick through the same reach
// picks. A value produced on the arm's spine after a nested exit rides the
// reach pick gating its segment (and every reach pick above): each becomes
// valued, its NOT_REACHED alternative yielding `uir.unreachable_value` and its
// REACHED alternative forwarding the value from the level below. A value that
// is itself a nested divergence's demanded later result surfaces as that
// divergence's own valued cf pick result, at the level gating the divergence,
// and threads up identically — the reconstruction is one recursion of the same
// valued-carrier mechanism, deepening by a level per nesting.
//
// One extraction moves content one boundary forward. Later-phase loop results
// are lowered onto receiving-phase variables by break-value delivery first, so
// their stores become ordinary later-phase content, and content more than one
// phase later than its frame simply remains later-phase content of the
// receiving replicate, which the joint fixpoint re-extracts at the next
// boundary until every phase holds only its own content.

// NOLINTBEGIN(misc-no-recursion)

/// Decompose a block — a loop body or a divergence arm — into content
/// segments separated by exit-carrying divergences. `plan.segments` holds the
/// later-phase ops between consecutive divergences in program order (one more
/// entry than `plan.divergences`); `plan.terminal` is the block terminator
/// that ends the fully-continued path.
///
/// Every arm of a cf-carrying divergence is a block like any other, so it is
/// scanned recursively: a nested exit-carrying branch inside it becomes a
/// nested divergence of the arm's own plan. A divergence with no later-phase
/// content and no demanded later result needs no cf and no arm scan: its
/// activation-ending terminators are collected directly, no matter how deeply
/// nested; only cf-carrying divergences must be branches and recurse.
///
/// Later-phase content nested inside non-exit CF at phase <= p is not this
/// frame's to record: branch extraction owns lifting it to the frame's levels,
/// and a nested frame's extraction may deposit fresh later content (its
/// receiving replicate) inside such CF after this sweep's branch extraction
/// already ran. The scan then sets `deferred` and the frame waits for the next
/// sweep of the joint fixpoint to lift the deposit; content no rule ever lifts
/// is rejected after the fixpoint settles.
LogicalResult PhaseSorter::scanBlock(Block &block, int16_t p, BodyPlan &plan,
                                     bool &deferred) {
  plan.terminal = block.getTerminator();
  plan.segments.emplace_back();
  for (auto &blockOp : block) {
    if (&blockOp == plan.terminal)
      break;
    if (auto ph = realPhaseOf(&blockOp); ph && *ph > p) {
      plan.segments.back().push_back(&blockOp);
      continue;
    }
    if (blockOp.getNumRegions() == 0)
      continue;

    SmallVector<Operation *> exitTerms;
    for (auto &region : blockOp.getRegions())
      collectExitTerminators(region, p, exitTerms);
    if (exitTerms.empty()) {
      // Later-phase content inside non-exit CF belongs to branch extraction;
      // defer this frame until a later sweep has lifted it to this level.
      if (hasLaterContent(&blockOp, p))
        deferred = true;
      continue;
    }

    // A demanded later-phase result makes the divergence cf-carrying even
    // without content: the receiving cf pick reconstructs the result from the
    // arms' yield operands, which requires knowing which arm ran.
    bool laterResults = llvm::any_of(blockOp.getResults(), [&](Value result) {
      auto it = analysis.actualPhase.find(result);
      return it != analysis.actualPhase.end() && isRealPhase(it->second) &&
             it->second > p;
    });

    Divergence divergence;
    divergence.fallsThrough = !analysis.diverges(&blockOp);
    if (!hasLaterContent(&blockOp, p) && !laterResults) {
      // Content-free divergence: no cf slot; every activation-ending
      // terminator pushes a record whose reach link is NOT_REACHED, no matter
      // how deeply the terminator is nested.
      divergence.exitTerminators = std::move(exitTerms);
    } else {
      // Later-phase content (or a demanded later result) must be attributed
      // to the arms that run (or provide) it, which requires a branch whose
      // arms name the paths.
      auto ifOp = dyn_cast<IfOp>(&blockOp);
      if (!ifOp)
        return blockOp.emitOpError()
               << "carries a loop exit but is not a branch; cannot build "
                  "partial activation records";
      divergence.ifOp = ifOp;

      for (auto &region : ifOp->getRegions()) {
        Arm arm;
        if (!region.empty()) {
          arm.region = &region;
          auto &armBlock = region.front();
          arm.plan = std::make_unique<BodyPlan>();
          if (failed(scanBlock(armBlock, p, *arm.plan, deferred)))
            return failure();
          auto *armTerm = armBlock.getTerminator();
          arm.reachesYield =
              isa<YieldOp>(armTerm) && analysis.reaches(armTerm, ifOp);
        }
        divergence.arms.push_back(std::move(arm));
      }
    }

    plan.divergences.push_back(std::move(divergence));
    plan.segments.emplace_back();
  }
  return success();
}

/// Collect every op the plan moves into the receiving replicate — segment
/// content and arm sub-plan content, recursively. These count as moved for
/// the crossing analysis, so a value one segment samples that another
/// produces does not spuriously become a record field.
void PhaseSorter::collectPlanContent(BodyPlan &plan,
                                     SmallVectorImpl<Operation *> &content) {
  for (auto &segment : plan.segments)
    content.append(segment.begin(), segment.end());
  for (auto &divergence : plan.divergences)
    for (auto &arm : divergence.arms)
      if (arm.plan)
        collectPlanContent(*arm.plan, content);
}

/// Collect the crossing values of every segment and arm sub-plan of `plan`,
/// deduplicating each level's fields against the values already visible from
/// its enclosing levels. `visible` is the set of values the enclosing levels
/// already bind; the spine extends it segment by segment, and each arm
/// receives a copy so its own extension stays local to its subtree (sibling
/// arms and the following segments never see an arm's interior fields).
void PhaseSorter::collectPlanFields(Operation *frameOp, int16_t p,
                                    BodyPlan &plan,
                                    ArrayRef<Operation *> allMoved,
                                    llvm::SetVector<Value> visible) {
  plan.segFields.assign(plan.segments.size(), {});
  for (auto [k, segment] : llvm::enumerate(plan.segments)) {
    plan.segFields[k] = collectCrossingValues(frameOp, p, segment, allMoved);
    plan.segFields[k].set_subtract(visible);
    visible.set_union(plan.segFields[k]);
    // The arms of this level's divergence replay nested under the levels up
    // to and including this one; scan each arm sub-plan against a copy of the
    // current visible set.
    if (k < plan.divergences.size())
      for (auto &arm : plan.divergences[k].arms)
        if (arm.plan)
          collectPlanFields(frameOp, p, *arm.plan, allMoved, visible);
  }
}

/// The level index of `plan` at which `value` becomes available during replay,
/// or `nullopt` if `value` is defined inside the plan but at no replayable
/// level.
///
/// Availability is a level position, not just a spine segment: a value born in
/// a segment surfaces at that segment's block, and a nested divergence's
/// demanded later result surfaces — through that divergence's own valued cf
/// pick — as a value in the same block that gates the divergence, i.e. at the
/// divergence's index. Both sit at the block for level k and thread up to the
/// consuming level through the same reach picks (see `buildLevel`). A null
/// operand (an arm that provides nothing) and a value defined outside the
/// plan's block subtree both dominate the plan root, so they sit at index 0.
std::optional<size_t> PhaseSorter::spinePosition(BodyPlan &plan, Value value) {
  if (!value)
    return 0;
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return 0; // A block argument (crossing field / loop init) dominates root.
  auto *planRegion = plan.terminal->getBlock()->getParent();
  if (!planRegion->isAncestor(defOp->getParentRegion()))
    return 0; // Defined outside the plan; dominates the plan root.
  for (auto [k, segment] : llvm::enumerate(plan.segments))
    for (auto *op : segment)
      if (op == defOp || op->isProperAncestor(defOp))
        return k;
  // A demanded later result of a nested divergence is reconstructed by that
  // divergence's cf pick, which lives in the block that gates the divergence
  // (its index). The placeholder standing in for the result until the cf pick
  // is built identifies the divergence.
  for (auto [k, divergence] : llvm::enumerate(plan.divergences))
    if (llvm::is_contained(divergence.laterResults, value))
      return k;
  return std::nullopt;
}

/// Rebuild every cf-carrying divergence `if` in `plan` so its fall-through
/// paths deliver the cf choice as the branch's last result, recursing into
/// arm sub-plans first so a nested divergence's `if` is rebuilt — and its cf
/// result available for the enclosing arm's payload — before the enclosing
/// yield is rewritten.
LogicalResult PhaseSorter::rebuildDivergences(BodyPlan &plan, int16_t p) {
  for (auto &divergence : plan.divergences) {
    for (auto &arm : divergence.arms)
      if (arm.plan)
        if (failed(rebuildDivergences(*arm.plan, p)))
          return failure();
    if (divergence.needsCf())
      if (failed(rebuildDivergenceIf(divergence, p)))
        return failure();
  }
  return success();
}

// NOLINTEND(misc-no-recursion)

// NOLINTBEGIN(misc-no-recursion)

/// Append the record-body values of `plan` from segment `seg` onward, as seen
/// from the point described by `path`: the segment's sample values, then per
/// divergence its cf value (the rebuilt branch's result for passed
/// divergences, or a cf built here from the taken arm's own record body for
/// the divergence the path exits through) and its reach link (a REACHED
/// choice wrapping the recursive tail for passed divergences, or an empty
/// NOT_REACHED ending the record at the exit). An empty path is the
/// fully-continued case: every divergence is passed and every link is
/// REACHED. Every referenced value dominates the build point: segment fields
/// and cf results of passed divergences are defined before the exit, and the
/// taken arm's fields are defined on the path to it.
void PhaseSorter::appendTail(BodyPlan &plan, size_t seg,
                             ArrayRef<PathStep> path, OpBuilder &builder,
                             Location loc, int16_t p,
                             SmallVectorImpl<Value> &out) {
  auto anyTy = hir::AnyType::get(builder.getContext());
  out.append(plan.segFields[seg].begin(), plan.segFields[seg].end());
  if (seg == plan.divergences.size())
    return;
  auto &divergence = plan.divergences[seg];
  assert((path.empty() || path.front().divIdx >= seg) &&
         "push-site path steps behind the tail being built");
  bool exitsHere = !path.empty() && path.front().divIdx == seg;
  if (!exitsHere) {
    if (divergence.needsCf()) {
      assert(divergence.cfValue &&
             "a passed cf-carrying divergence must have a fall-through path "
             "delivering its cf result");
      out.push_back(divergence.cfValue);
    }
    SmallVector<Value> payload;
    appendTail(plan, seg + 1, path, builder, loc, p, payload);
    auto reached = hir::ChoiceOp::create(builder, loc, anyTy, 1, payload);
    setPhase(reached, p);
    out.push_back(reached.getResult());
    return;
  }
  if (divergence.needsCf()) {
    auto &arm = divergence.arms[path.front().armIdx];
    SmallVector<Value> payload;
    appendTail(*arm.plan, 0, path.drop_front(), builder, loc, p, payload);
    auto cf = hir::ChoiceOp::create(builder, loc, anyTy,
                                    static_cast<uint32_t>(path.front().armIdx),
                                    payload);
    setPhase(cf, p);
    out.push_back(cf.getResult());
  }
  auto notReached = hir::ChoiceOp::create(builder, loc, anyTy, 0, ValueRange{});
  setPhase(notReached, p);
  out.push_back(notReached.getResult());
}

/// Collect every activation-ending push site of `plan`, depth-first in
/// program order: each content-free divergence's exit terminators, each arm's
/// sites recursively, and the plan's own terminal when it ends an activation.
/// A break/continue always does (the loop body's back edge, or an
/// exit-terminated arm whose interior divergences were all passed), and so
/// does a terminal `hir.yield`, which only ever appears as a replicate frame's
/// replay-body terminator and completes one replay; a `uir.yield` arm terminal
/// falls through to the enclosing level, and a `uir.unreachable` never
/// completes an activation.
///
/// A divergence with no fall-through (`Divergence::fallsThrough`) makes the
/// entire remainder of its plan's spine structurally unreachable: no path
/// leaves it toward the next sibling. Its own arms (or exit terminators) still
/// hold the real exits, so they are collected, but every later divergence and
/// the plan terminal are dead and must contribute no push site — a record path
/// walking through it as a passed divergence would describe a path that cannot
/// execute (and, for a cf-carrying divergence, would find no cf result for
/// `appendTail`). So the scan stops right after such a divergence's sites,
/// independent of whether the divergence carries a cf slot.
void PhaseSorter::collectPushSites(BodyPlan &plan,
                                   SmallVectorImpl<PathStep> &path,
                                   SmallVectorImpl<PushSite> &sites) {
  for (auto [k, divergence] : llvm::enumerate(plan.divergences)) {
    if (!divergence.needsCf()) {
      for (auto *term : divergence.exitTerminators) {
        PushSite site{term, {path.begin(), path.end()}};
        site.path.push_back({k, 0});
        sites.push_back(std::move(site));
      }
    } else {
      for (auto [j, arm] : llvm::enumerate(divergence.arms)) {
        if (!arm.plan)
          continue;
        path.push_back({k, j});
        collectPushSites(*arm.plan, path, sites);
        path.pop_back();
      }
    }
    // A divergence with no fall-through cuts the spine: nothing after it is
    // structurally reachable.
    if (!divergence.fallsThrough)
      return;
  }
  if (isa<BreakOp, ContinueOp, hir::YieldOp>(plan.terminal))
    sites.push_back({plan.terminal, {path.begin(), path.end()}});
}

/// Rebuild a cf-carrying divergence `if` so its fall-through paths deliver
/// the cf choice as the branch's last result.
///
/// Each arm whose yield is reachable rebuilds it: the cf tag is the arm's
/// region index, and the payload is the arm's record body on the
/// fully-continued path — first-segment samples, then nested cf values and
/// REACHED links, built right before the yield where all arm-interior samples
/// are in scope. An arm whose yield is unreachable provides nothing and its
/// yield is emptied; an exit-terminated arm builds its cf at its own push
/// sites instead. An absent else region is materialized to yield its empty
/// cf tag, so the fall-through path always delivers the choice.
///
/// A demanded later-phase result of the `if` cannot be delivered by a phase-p
/// branch. Such results are dropped from the rebuilt `if`; each arm's yield
/// operand for a later result is recorded (null when the arm does not provide
/// it) so the receiving cf pick reconstructs the values. The original results
/// are replaced by throw-away placeholders that the pick build later rewrites
/// to its own results and erases.
LogicalResult PhaseSorter::rebuildDivergenceIf(Divergence &divergence,
                                               int16_t p) {
  IfOp ifOp = divergence.ifOp;
  auto loc = ifOp.getLoc();
  auto anyTy = hir::AnyType::get(ifOp.getContext());

  // Partition the results: those at phase p stay on the rebuilt `if`; those
  // at a later phase are reconstructed by the receiving cf pick.
  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : ifOp.getResults()) {
    int16_t vph = analysis.actualPhase.at(result);
    if (isRealPhase(vph) && vph > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }

  // The cf choice is delivered as the branch's last result by the arms that
  // fall through. A divergence none of whose arms falls through is never a
  // passed divergence of any record (every path through it ends the
  // activation), so its branch keeps no cf result — the exit sites build
  // their cf values locally — and the rebuilt branch honors the op's rule
  // that an `if` whose regions all exit early produces no results.
  bool anyFallthrough = llvm::any_of(divergence.arms, [](const Arm &arm) {
    return !arm.region || arm.reachesYield;
  });

  OpBuilder builder(ifOp);
  auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
  setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
  Value choiceType = choiceTypeOp.getResult();

  for (auto [tag, arm] : llvm::enumerate(divergence.arms)) {
    if (!arm.plan)
      continue;
    auto &armBlock = arm.region->front();
    auto yieldOp = dyn_cast<YieldOp>(armBlock.getTerminator());
    if (!yieldOp) {
      // Exit-terminated arm: it never rejoins, provides no results, and
      // builds its cf at its own push sites.
      arm.laterYieldOperands.assign(laterIdx.size(), Value());
      continue;
    }
    OpBuilder armBuilder(yieldOp);
    if (!arm.reachesYield) {
      // The arm's yield is unreachable (every path exits before it), so it
      // provides nothing. Rebuild it at full arity with `uir.unreachable_value`
      // in every slot the rebuilt branch carries — the kept results, plus the
      // cf choice when some arm falls through — keeping the result mapping
      // positional.
      arm.laterYieldOperands.assign(laterIdx.size(), Value());
      SmallVector<Value> deadValues, deadTypes;
      auto addDead = [&](Value typeOperand) {
        auto uv = UnreachableValueOp::create(armBuilder, loc, anyTy);
        setPhase(uv, p);
        deadValues.push_back(uv.getResult());
        deadTypes.push_back(typeOperand);
      };
      for (auto idx : keptIdx)
        addDead(ifOp.getResultTypes()[idx]);
      if (anyFallthrough)
        addDead(choiceType);
      auto newYield = YieldOp::create(armBuilder, loc, deadValues, deadTypes);
      setPhase(newYield, p);
      if (arm.plan->terminal == yieldOp.getOperation())
        arm.plan->terminal = newYield;
      forget(yieldOp);
      yieldOp.erase();
      continue;
    }
    // A slot the arm does not provide (a non-materializing operand) becomes a
    // null operand: the receiving cf pick yields a fresh unreachable_value for
    // it, and it needs no spine position.
    for (auto idx : laterIdx) {
      Value operand = yieldOp.getValues()[idx];
      arm.laterYieldOperands.push_back(
          analysis.materializes(operand) ? operand : Value());
    }
    SmallVector<Value> payload;
    appendTail(*arm.plan, 0, {}, armBuilder, loc, p, payload);
    auto choice = hir::ChoiceOp::create(armBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag), payload);
    setPhase(choice, p);
    SmallVector<Value> yieldValues, yieldTypes;
    for (auto idx : keptIdx) {
      // The cf choice lands in the rebuilt branch's last result slot; the kept
      // results precede it. Full-arity yields guarantee an operand per kept
      // slot — a slot the arm does not provide is a `uir.unreachable_value`,
      // which rides the kept-result slot exactly like a real value.
      assert(idx < yieldOp.getValues().size() &&
             "yield under-provides the branch's kept results");
      yieldValues.push_back(yieldOp.getValues()[idx]);
      yieldTypes.push_back(yieldOp.getTypeOfValues()[idx]);
    }
    yieldValues.push_back(choice.getResult());
    yieldTypes.push_back(choiceType);
    auto newYield = YieldOp::create(armBuilder, loc, yieldValues, yieldTypes);
    setPhase(newYield, p);
    if (arm.plan->terminal == yieldOp.getOperation())
      arm.plan->terminal = newYield;
    forget(yieldOp);
    yieldOp.erase();
  }

  // Rebuild the `if` with the kept results plus, when a fall-through path
  // exists, one extra result for the cf choice. Every if result is an
  // `!hir.any`; the `resultTypes` operand carries the per-result type
  // operands.
  SmallVector<Type> resultTypes(keptIdx.size() + (anyFallthrough ? 1 : 0),
                                anyTy);
  SmallVector<Value> typeOperands;
  for (auto idx : keptIdx)
    typeOperands.push_back(ifOp.getResultTypes()[idx]);
  if (anyFallthrough)
    typeOperands.push_back(choiceType);
  auto newIf = IfOp::create(builder, loc, resultTypes, ifOp.getCondition(),
                            typeOperands);
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  if (!ifOp.getElseRegion().empty())
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  setPhase(newIf, p);
  for (auto [tag, arm] : llvm::enumerate(divergence.arms)) {
    if (arm.region) {
      arm.region = &newIf->getRegion(tag);
      continue;
    }
    // Materialize the absent else region: an empty fall-through arm that
    // yields its empty cf tag. An if without an else has no results, so the
    // choice is the only yielded value.
    auto &elseBlock = newIf.getElseRegion().emplaceBlock();
    OpBuilder elseBuilder(&elseBlock, elseBlock.end());
    auto choice = hir::ChoiceOp::create(
        elseBuilder, loc, anyTy, static_cast<uint32_t>(tag), ValueRange{});
    setPhase(choice, p);
    auto yield =
        YieldOp::create(elseBuilder, loc, ValueRange{choice.getResult()},
                        ValueRange{choiceType});
    setPhase(yield, p);
    arm.region = &newIf.getElseRegion();
    arm.laterYieldOperands.assign(laterIdx.size(), Value());
  }
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newIf.getResult(i)] =
        analysis.actualPhase.at(ifOp.getResult(idx));
    ifOp.getResult(idx).replaceAllUsesWith(newIf.getResult(i));
  }
  // The dropped later results have no phase-p producer to point at yet; stand
  // them up with placeholders the cf pick build replaces and erases. The
  // placeholder value keeps the original result's phase: the cf pick that
  // replaces it inherits the phase, so a result wider than one boundary is
  // delivered by a pick whose result is later than its own phase and
  // re-extracts at the next boundary like any other wide receiving pick.
  for (auto idx : laterIdx) {
    auto placeholder =
        hir::OpaquePackOp::create(builder, loc, anyTy, ValueRange{});
    setPhase(placeholder, static_cast<int16_t>(p + 1));
    analysis.actualPhase[placeholder.getResult()] =
        analysis.actualPhase.at(ifOp.getResult(idx));
    ifOp.getResult(idx).replaceAllUsesWith(placeholder.getResult());
    divergence.laterResults.push_back(placeholder.getResult());
  }
  if (anyFallthrough) {
    Value cf = newIf.getResult(newIf.getNumResults() - 1);
    analysis.actualPhase[cf] = p;
    divergence.cfValue = cf;
  }
  divergence.ifOp = newIf;
  forget(ifOp);
  ifOp.erase();
  return success();
}

// NOLINTEND(misc-no-recursion)

/// Invoke `fn` on every `uir.break` targeting `loopOp`, i.e. every break in its
/// body not nested inside an inner loop (whose breaks target that inner loop).
static void forEachLoopBreak(LoopOp loopOp,
                             llvm::function_ref<void(BreakOp)> fn) {
  std::function<void(Block &)> visit = [&](Block &block) {
    for (auto &blockOp : block) {
      if (isa<LoopOp>(&blockOp))
        continue;
      if (auto breakOp = dyn_cast<BreakOp>(&blockOp)) {
        fn(breakOp);
        continue;
      }
      for (auto &region : blockOp.getRegions())
        for (auto &inner : region)
          visit(inner);
    }
  };
  visit(loopOp.getBody().front());
}

//===----------------------------------------------------------------------===//
// Break-Value Delivery
//===----------------------------------------------------------------------===//
//
// # Deliver Later-Phase Loop Results Through Receiving-Phase Variables
//
// A `uir.loop` at phase p whose break carries a later-phase value cannot
// deliver that value at phase p: the value materializes once the loop has run
// to completion, in a later phase (causality, sampling-model.md §7). A later
// break value is not its own delivery concept — it is a demanded later result
// of the loop, delivered by the record/replicate machinery like any other
// later-phase content. This step lowers each later loop result onto a
// receiving-phase variable so the record machinery does the rest:
//
// - One `hir.var` is created for each later result slot, at the result's phase.
// - Right before each break, one `hir.write_var` per later slot stores the
//   break's later operand into that slot's variable, at the result's phase.
//   The break drops its later operands and the loop drops its later result
//   slots, so the phase-p loop and its breaks hold only phase-p values.
// - After the loop, one `hir.read_var` per later slot reads the final value and
//   replaces the loop result's uses.
//
// The write is ordinary later-phase content on the break's record path: it
// forces the exiting divergence to carry a cf choice, replays exactly once per
// activation in that arm's receiving `hir.pick` alternative (built by record
// extraction below), and its value operand's own computation replays there too
// — including a value shared between several break sites (recorded once in the
// shared segment) or a value produced by a nested divergence (reconstructed by
// that divergence's own valued cf pick). Because the record machinery replays
// each path once and the write dominates its read, the variable observes the
// value the breaking activation produced. No content is sliced or duplicated,
// so the former shared-break-value-slice rejections dissolve.

/// Lower the later-phase results of a phase-p `uir.loop` onto receiving-phase
/// variables written on each break and read after the loop. See the section
/// header. Returns the rebuilt loop; the loop is returned unchanged when it has
/// no later-phase result. The receiving reads are placed immediately after the
/// rebuilt loop, so record extraction's replicate (inserted right after the
/// loop) slots in ahead of them and all activations run before the read.
FailureOr<LoopOp> PhaseSorter::deliverBreakValues(LoopOp loopOp, int16_t p) {
  Operation *op = loopOp;

  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : op->getResults()) {
    int16_t phase = analysis.actualPhase.at(result);
    if (isRealPhase(phase) && phase > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }
  if (laterIdx.empty())
    return loopOp;

  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());

  // One receiving-phase variable per later result slot, at the result's phase.
  // The slot's type operand doubles as the variable's value type and the type
  // operand of every write and the read.
  OpBuilder builder(op);
  SmallVector<Value> vars;
  SmallVector<Value> varTypes;
  SmallVector<int16_t> varPhases;
  for (auto idx : laterIdx) {
    int16_t resultPhase = analysis.actualPhase.at(op->getResult(idx));
    Value varType = loopOp.getResultTypes()[idx];
    auto varOp = hir::VarOp::create(builder, loc, anyTy, varType);
    setPhase(varOp, resultPhase);
    vars.push_back(varOp.getResult());
    varTypes.push_back(varType);
    varPhases.push_back(resultPhase);
  }

  // Rewrite each break: store its later operands into the variables and drop
  // them from the break, keeping only the phase-p results. Break arity matches
  // the loop's results (verifier-enforced), so every slot indexes in range.
  SmallVector<BreakOp> breaks;
  forEachLoopBreak(loopOp, [&](BreakOp breakOp) { breaks.push_back(breakOp); });
  for (auto breakOp : breaks) {
    assert(breakOp.getValues().size() == op->getNumResults() &&
           "break under-provides the loop's results");
    OpBuilder breakBuilder(breakOp);
    for (auto [i, idx] : llvm::enumerate(laterIdx)) {
      auto write = hir::WriteVarOp::create(
          breakBuilder, loc, vars[i], breakOp.getValues()[idx], varTypes[i]);
      setPhase(write, varPhases[i]);
    }
    SmallVector<Value> values, typeOfValues;
    for (auto idx : keptIdx) {
      values.push_back(breakOp.getValues()[idx]);
      typeOfValues.push_back(breakOp.getTypeOfValues()[idx]);
    }
    auto newBreak = BreakOp::create(breakBuilder, loc, values, typeOfValues);
    setPhase(newBreak, p);
    forget(breakOp);
    breakOp.erase();
  }

  // Rebuild the loop with the kept results only.
  SmallVector<Type> newResultTypes(keptIdx.size(), anyTy);
  SmallVector<Value> newTypeOperands;
  for (auto idx : keptIdx)
    newTypeOperands.push_back(loopOp.getResultTypes()[idx]);
  auto newLoop = LoopOp::create(builder, loc, newResultTypes, loopOp.getInits(),
                                loopOp.getInitTypes(), newTypeOperands);
  newLoop.getBody().takeBody(loopOp.getBody());
  setPhase(newLoop, p);
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newLoop.getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newLoop.getResult(i));
  }

  // Read each variable right after the loop and forward the loop result's uses.
  builder.setInsertionPointAfter(newLoop);
  for (auto [i, idx] : llvm::enumerate(laterIdx)) {
    auto read =
        hir::ReadVarOp::create(builder, loc, anyTy, vars[i], varTypes[i]);
    setPhase(read, varPhases[i]);
    op->getResult(idx).replaceAllUsesWith(read.getResult());
  }

  forget(op);
  op->erase();
  return newLoop;
}

/// Extract the later-phase content of a phase-p producing frame — a `uir.loop`
/// or a `hir.replicate` — into a hit list and a receiving `hir.replicate` at
/// phase p+1. See the section header for the shape and invariants. Every
/// activation pushes exactly one record, built in place at the terminator that
/// ends it: the record chains a `continued` link per divergence the activation
/// fell past and ends in the taken exit path's choice, whose payload is empty
/// when that path ran no later-phase content. Sets `changed` when it rewrites
/// anything; a deferred scan (later content still nested inside non-exit CF
/// awaiting branch extraction) leaves the frame untouched for the next sweep.
LogicalResult PhaseSorter::extractFromFrame(Operation *frameOp, int16_t p,
                                            bool &changed) {
  auto loc = frameOp->getLoc();
  auto anyTy = hir::AnyType::get(frameOp->getContext());

  // Later-phase loop results are delivered through receiving-phase variables:
  // each break stores its later operand into a variable (ordinary later-phase
  // content on the break's record path) and a read after the loop feeds the
  // result's uses. Lower these first, so the writes are in place as ordinary
  // record content the analysis below picks up; the loop is rebuilt in the
  // process. A replicate frame has no results and no breaks, so there is
  // nothing to deliver.
  if (auto loopOp = dyn_cast<LoopOp>(frameOp)) {
    auto rebuilt = deliverBreakValues(loopOp, p);
    if (failed(rebuilt))
      return failure();
    if (*rebuilt != loopOp)
      changed = true;
    frameOp = *rebuilt;
  }
  Operation *op = frameOp;

  // A loop with a later result always gains write content from break-value
  // delivery, but a kept-only loop reached here for its record content may have
  // none left; nothing then remains to record-extract.
  if (!hasLaterContent(op, p))
    return success();

  // Both frame kinds hold one body region. A loop body ends in the back-edge
  // continue, a break, or unreachable; a replicate replay body ends in the
  // empty `hir.yield` that completes one replay.
  auto &bodyBlock = op->getRegion(0).front();
  auto *bodyTerm = bodyBlock.getTerminator();
  if (!isa<BreakOp, ContinueOp, UnreachableOp, hir::YieldOp>(bodyTerm))
    return bodyTerm->emitOpError()
           << "unsupported frame body terminator for record extraction";

  // Decompose the frame body into the recursive plan of content segments and
  // exit-carrying divergences, then collect the crossing values every segment
  // and arm sub-plan samples. Everything the plan moves counts as moved for
  // the crossing analysis. A deferred scan found later content still nested
  // inside non-exit CF; leave the frame for the next sweep, after branch
  // extraction has lifted it to this frame's levels.
  BodyPlan plan;
  bool deferred = false;
  if (failed(scanBlock(bodyBlock, p, plan, deferred)))
    return failure();
  if (deferred)
    return success();
  changed = true;

  SmallVector<Operation *> allMoved;
  collectPlanContent(plan, allMoved);
  collectPlanFields(op, p, plan, allMoved, {});

  // With the fields known, rebuild every cf-carrying divergence `if` so its
  // fall-through paths deliver the cf choice as the branch's last result;
  // exit paths build their cf at their own push sites below.
  if (failed(rebuildDivergences(plan, p)))
    return failure();

  // Hit list before the op.
  OpBuilder builder(op);
  auto list = hir::OpaqueListCreateOp::create(builder, loc, anyTy);
  setPhase(list, p);

  // # Build the record pushes
  //
  // Every terminator that ends an activation — each activation-ending
  // break/continue, however deeply nested, and the loop's own back edge —
  // pushes exactly one record, built in place from the statically-known path
  // to that terminator: the root segment's samples, then per divergence its
  // cf slot and reach link (see `appendTail`). Everything a path records
  // dominates its terminator, so the record is built bottom-up right before
  // the push.
  SmallVector<PushSite> sites;
  {
    SmallVector<PathStep> path;
    collectPushSites(plan, path, sites);
  }
  for (auto &site : sites) {
    OpBuilder siteBuilder(site.terminator);
    SmallVector<Value> recordFields;
    appendTail(plan, 0, site.path, siteBuilder, loc, p, recordFields);
    auto record =
        hir::OpaquePackOp::create(siteBuilder, loc, anyTy, recordFields);
    setPhase(record, p);
    auto push = hir::OpaqueListPushOp::create(
        siteBuilder, loc, list.getResult(), record.getResult());
    setPhase(push, p);
  }

  // # Build the receiving replicate
  //
  // The replicate mirrors the source control-flow skeleton with data-only
  // dispatch: per level, the segment's content replays first, then the
  // divergence's cf pick dispatches the arm that ran (recursing into the
  // arm's own levels), then the reach pick either stops (NOT_REACHED) or
  // binds the next segment's samples and continues one level deeper in its
  // REACHED alternative. No exit terminators appear; a record's tags fully
  // determine which content replays, and every piece of content replays in
  // exactly one place.
  builder.setInsertionPointAfter(op);
  auto replicate = hir::ReplicateOp::create(builder, loc, list.getResult());
  analysis.opPhases[replicate] = p + 1;

  SmallVector<std::pair<Block *, const llvm::SetVector<Value> *>> remapBlocks;
  std::function<void(BodyPlan &, size_t, Block *, ArrayRef<Value>)> buildLevel =
      [&](BodyPlan &pl, size_t seg, Block *block, ArrayRef<Value> levelYield) {
        // The values this level yields, as a mutable copy: building this
        // level's cf pick reconstructs a nested divergence's later result and
        // erases the placeholder that stood in for it, so the yield entry must
        // be re-pointed to the fresh cf pick result below.
        SmallVector<Value> yield(levelYield.begin(), levelYield.end());

        // Block args mirror the record tail this level consumes: the
        // segment's samples, then the divergence's cf choice and reach link.
        for (unsigned i = 0; i < pl.segFields[seg].size(); ++i)
          block->addArgument(anyTy, loc);
        remapBlocks.push_back({block, &pl.segFields[seg]});
        for (auto *moved : pl.segments[seg])
          moved->moveBefore(block, block->end());

        // Partition the demanded later values by level position: those defined
        // in a deeper level are not in scope at this block's terminal, so they
        // ride this level's reach pick as its results (threaded up from the
        // level that defines them); the rest are delivered here. A value at
        // this level (or dominating the root, index 0) is available directly.
        SmallVector<bool> deeperMask(yield.size(), false);
        SmallVector<Value> deeperValues;
        // The phase of each deeper value, captured now: a deeper value may be
        // a nested divergence's placeholder that the recursive level build
        // erases, after which only the reach pick result carrying its phase
        // remains.
        SmallVector<int16_t> deeperPhases;
        for (auto [i, value] : llvm::enumerate(yield)) {
          auto pos = spinePosition(pl, value);
          assert(pos && "off-spine later operand should have been rejected");
          if (*pos > seg) {
            deeperMask[i] = true;
            deeperValues.push_back(value);
            deeperPhases.push_back(analysis.actualPhase.at(value));
          }
        }

        // The reach pick's results, one per deeper value, in `deeperValues`
        // order; empty when no deeper value threads through this level.
        SmallVector<Value> reachResults;

        if (seg < pl.divergences.size()) {
          auto &divergence = pl.divergences[seg];
          if (divergence.needsCf()) {
            auto cfArg = block->addArgument(anyTy, loc);
            analysis.actualPhase[cfArg] = p;
            OpBuilder cfBuilder(block, block->end());
            SmallVector<Type> pickResultTypes(divergence.laterResults.size(),
                                              anyTy);
            auto cfPick = hir::PickOp::create(cfBuilder, loc, pickResultTypes,
                                              cfArg, divergence.arms.size());
            analysis.opPhases[cfPick] = p + 1;
            for (auto [tag, arm] : llvm::enumerate(divergence.arms)) {
              auto &altBlock = cfPick.getAlternatives()[tag].emplaceBlock();
              if (arm.plan) {
                buildLevel(*arm.plan, 0, &altBlock, arm.laterYieldOperands);
              } else {
                // Materialized fall-through arm: nothing to replay.
                OpBuilder altBuilder(&altBlock, altBlock.end());
                SmallVector<Value> altYieldValues;
                for (unsigned i = 0; i < divergence.laterResults.size(); ++i) {
                  auto unreachable =
                      UnreachableValueOp::create(altBuilder, loc, anyTy);
                  setPhase(unreachable, static_cast<int16_t>(p + 1));
                  altYieldValues.push_back(unreachable.getResult());
                }
                auto altYield =
                    hir::YieldOp::create(altBuilder, loc, altYieldValues);
                analysis.opPhases[altYield] = p + 1;
              }
            }
            // Replace each dropped later result's placeholder with the cf
            // pick's reconstructed result and drop the placeholder op. A value
            // this level itself yields is the reconstructed result of this same
            // divergence (a nested divergence whose result the enclosing arm
            // provides), so re-point the yield copy to the fresh cf pick result
            // before the placeholder is erased.
            // The pick result inherits the placeholder's phase (the original
            // result's), so a value wider than one boundary re-extracts.
            for (auto [j, placeholder] :
                 llvm::enumerate(divergence.laterResults)) {
              Operation *placeholderOp = placeholder.getDefiningOp();
              analysis.actualPhase[cfPick.getResult(j)] =
                  analysis.actualPhase.at(placeholder);
              for (auto &y : yield)
                if (y == placeholder)
                  y = cfPick.getResult(j);
              placeholder.replaceAllUsesWith(cfPick.getResult(j));
              forget(placeholderOp);
              placeholderOp->erase();
            }
          }
          auto linkArg = block->addArgument(anyTy, loc);
          analysis.actualPhase[linkArg] = p;
          OpBuilder linkBuilder(block, block->end());
          // A demanded later value defined beyond this divergence rides the
          // reach pick: NOT_REACHED cannot provide it (the activation exited
          // here) and yields `uir.unreachable_value`; REACHED forwards it from
          // the level below. The pick's results reconstruct the values at this
          // level, exactly as the valued cf pick reconstructs arm results.
          SmallVector<Type> reachResultTypes(deeperValues.size(), anyTy);
          auto reachPick = hir::PickOp::create(linkBuilder, loc,
                                               reachResultTypes, linkArg, 2);
          analysis.opPhases[reachPick] = p + 1;
          auto &nrBlock = reachPick.getAlternatives()[0].emplaceBlock();
          OpBuilder nrBuilder(&nrBlock, nrBlock.end());
          SmallVector<Value> nrYieldValues;
          for (unsigned i = 0; i < deeperValues.size(); ++i) {
            auto unreachable =
                UnreachableValueOp::create(nrBuilder, loc, anyTy);
            setPhase(unreachable, static_cast<int16_t>(p + 1));
            nrYieldValues.push_back(unreachable.getResult());
          }
          auto nrYield = hir::YieldOp::create(nrBuilder, loc, nrYieldValues);
          analysis.opPhases[nrYield] = p + 1;
          auto &rBlock = reachPick.getAlternatives()[1].emplaceBlock();
          buildLevel(pl, seg + 1, &rBlock, deeperValues);
          // Each reach pick result stands in for its threaded value and keeps
          // that value's phase, so a wide value stays wide at every level and
          // re-extracts at the next boundary.
          for (auto [result, phase] :
               llvm::zip(reachPick.getResults(), deeperPhases)) {
            analysis.actualPhase[result] = phase;
            reachResults.push_back(result);
          }
        } else {
          assert(deeperValues.empty() &&
                 "deeper later value with no divergence to gate it");
        }

        // Terminal yield of this level, carrying the cf alternative's later
        // result values (empty elsewhere): a deeper value comes from the reach
        // pick's result, a null value is a result the arm does not provide and
        // is yielded as `uir.unreachable_value` (its consumers replay only on
        // paths the reach links gate off), and any other value is in scope
        // here and yielded directly.
        OpBuilder endBuilder(block, block->end());
        SmallVector<Value> yieldValues;
        size_t deeperIdx = 0;
        for (auto [i, value] : llvm::enumerate(yield)) {
          if (deeperMask[i]) {
            yieldValues.push_back(reachResults[deeperIdx++]);
            continue;
          }
          if (!value) {
            auto unreachable =
                UnreachableValueOp::create(endBuilder, loc, anyTy);
            setPhase(unreachable, static_cast<int16_t>(p + 1));
            yieldValues.push_back(unreachable.getResult());
            continue;
          }
          yieldValues.push_back(value);
        }
        auto levelYieldOp = hir::YieldOp::create(endBuilder, loc, yieldValues);
        analysis.opPhases[levelYieldOp] = p + 1;
      };
  buildLevel(plan, 0, &replicate.getBody().emplaceBlock(), {});

  // Remap fields outermost-first so that deduplicated inner uses resolve to
  // the enclosing level's arguments, then clone floating trees once over the
  // whole structure.
  for (auto [block, fields] : remapBlocks)
    remapFields(*block, *fields);
  cloneFloatingUses(op, replicate.getBody().front());
  return success();
}

/// Extract later-phase content out of every phase-p producing frame — a
/// `uir.loop` or a `hir.replicate` — whose interior or results demand it,
/// innermost first so a nested frame becomes ordinary later-phase content of
/// its enclosing frame's records. A later-phase loop result is an extraction
/// obligation on its own: break-value delivery turns it into a receiving-phase
/// variable store, which is later-phase content the record extraction then
/// picks up. A replicate is the producing frame of its own phase: its replay
/// runs once per record, so later-phase content inside it re-records onto a
/// fresh hit list — sampling the replicate's unpacked block arguments and pick
/// reconstructions like any other per-iteration value — and each sweep moves
/// the content exactly one boundary forward, the caller's joint fixpoint
/// driving multi-boundary chains. Sets `changed` when it rewrites any frame.
LogicalResult PhaseSorter::extractFrames(bool &changed) {
  SmallVector<Operation *> frames;
  funcOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<LoopOp, hir::ReplicateOp>(op) && realPhaseOf(op))
      frames.push_back(op);
  });

  for (auto *frameOp : frames) {
    int16_t p = *realPhaseOf(frameOp);
    bool hasLaterResult = llvm::any_of(frameOp->getResults(), [&](Value r) {
      int16_t vph = analysis.actualPhase.at(r);
      return isRealPhase(vph) && vph > p;
    });
    if (!hasLaterResult && !hasLaterContent(frameOp, p))
      continue;
    if (failed(extractFromFrame(frameOp, p, changed)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Forward Single-Provider Control-Flow Results
//===----------------------------------------------------------------------===//
//
// # Forward Single-Provider Control-Flow Results to Their Provider
//
// A control-flow result (`uir.if`, `hir.pick`, `uir.loop`) whose value is
// delivered by
// exactly one terminator is not a selection: it *is* that terminator's operand.
// When the operand is defined outside the op it dominates the op, hence every
// use of the result, so the result forwards: every use is replaced by the
// operand and the result — with its type operand and the terminators' operands
// for the slot — is dropped from the op.
//
// This restores the invariant the wrap and cross rules rely on: every op
// delivers each of its results at its own phase. A single-provider result may
// otherwise sit at the operand's phase, earlier or later than the op (the op
// imposes no floor on a non-selection), which those rules cannot express — the
// same forwarding covers both directions.
//
// The externality of the provider, not its count, is the boundary between
// forwarding and selection. A single-provider result whose operand is defined
// outside the op forwards directly. A single interior provider at the float
// phase forwards too: a float is a region-free pure tree, materializable at
// any point, so its tree is cloned in front of the op — where it dominates
// every use of the result — and the result forwards to the clone, exactly as
// hoisting clones the floating operands of the content it moves. Everything
// else is a genuine selection left to branch extraction and break-value
// delivery, which run *after* this step. Running forwarding first means those
// steps only ever see selections: a multi-provider result (delivered
// regardless of distance), or a single anchored provider defined *inside* the
// op — later-phase arm content or an iteration-snapshotted break value, where
// the interior value genuinely varies per path or iteration and so cannot
// forward even at a provider count of one. A multi-provider result is skipped
// here without inspecting its operands; its phase is floored at the op's, so
// only single-provider results ever sit at the float phase. Never results
// were already severed by never-forwarding. A single anchored provider
// defined inside the op at an earlier phase (backward interior content) is
// hoisted out by the joint fixpoint with `hoistEarlierContent`, after which
// it is external and forwards here.

/// The terminator operands delivering result `idx` of a control-flow op from
/// its reachable carriers: the yield operands of a branch's reachable,
/// non-exiting arms (`uir.if` or `hir.pick`), or the break operands of a
/// reachable break of a `loop`. Provider-ness is a control-flow property
/// (`reaches`), so a reachable carrier is a provider whatever it delivers — a
/// `uir.unreachable_value` poison operand included — and a single entry marks a
/// single-provider result.
SmallVector<Value> PhaseSorter::collectResultProviders(Operation *op,
                                                       unsigned idx) {
  SmallVector<Value> providers;
  if (isa<IfOp, hir::PickOp>(op)) {
    for (auto *region : branchArms(op)) {
      if (region->empty())
        continue;
      auto *term = region->front().getTerminator();
      if (auto values = armProvidedValues(term))
        if (analysis.reaches(term, op))
          providers.push_back((*values)[idx]);
    }
  } else {
    forEachLoopBreak(cast<LoopOp>(op), [&](BreakOp breakOp) {
      if (analysis.reaches(breakOp, op))
        providers.push_back(breakOp.getValues()[idx]);
    });
  }
  return providers;
}

/// Whether `value` is defined outside `op` and thus dominates it: a result of
/// an op that is neither `op` nor nested in it, or a block argument whose owner
/// is neither `op` nor nested in it.
static bool definedOutside(Value value, Operation *op) {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *owner = blockArg.getOwner()->getParentOp();
    return owner != op && !op->isProperAncestor(owner);
  }
  Operation *defOp = value.getDefiningOp();
  return defOp && defOp != op && !op->isProperAncestor(defOp);
}

/// Forward the results of `op` whose entry in `forwardTo` is non-null: replace
/// every use of the result with its provider operand, then rebuild `op` (an
/// `if`, `pick`, or `loop`) without those results, dropping the same slots
/// from every result-carrying terminator and preserving the kept results'
/// phases.
void PhaseSorter::forwardResults(Operation *op, ArrayRef<Value> forwardTo,
                                 int16_t p) {
  for (auto [idx, result] : llvm::enumerate(op->getResults()))
    if (forwardTo[idx])
      result.replaceAllUsesWith(forwardTo[idx]);

  SmallVector<unsigned> keptIdx;
  for (unsigned idx = 0, e = op->getNumResults(); idx < e; ++idx)
    if (!forwardTo[idx])
      keptIdx.push_back(idx);

  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());
  SmallVector<Type> keptTypes(keptIdx.size(), anyTy);

  // Rewrite one result-carrying terminator (`uir.yield` or `uir.break`, with
  // paired value/type operands) to keep only the kept slots, preserving its
  // phase.
  auto rewriteTerminator = [&](auto termOp) {
    SmallVector<Value> values, typeOfValues;
    for (auto idx : keptIdx) {
      values.push_back(termOp.getValues()[idx]);
      typeOfValues.push_back(termOp.getTypeOfValues()[idx]);
    }
    int16_t phase = analysis.opPhases.at(termOp);
    OpBuilder builder(termOp);
    auto newTerm = std::decay_t<decltype(termOp)>::create(
        builder, termOp.getLoc(), values, typeOfValues);
    setPhase(newTerm, phase);
    forget(termOp);
    termOp.erase();
  };

  // Same for a pick alternative's `hir.yield`, which carries plain operands
  // and no type operands.
  auto rewriteHirYield = [&](hir::YieldOp yieldOp) {
    SmallVector<Value> values;
    for (auto idx : keptIdx)
      values.push_back(yieldOp.getOperands()[idx]);
    int16_t phase = analysis.opPhases.at(yieldOp);
    OpBuilder builder(yieldOp);
    auto newTerm = hir::YieldOp::create(builder, yieldOp.getLoc(), values);
    setPhase(newTerm, phase);
    forget(yieldOp);
    yieldOp.erase();
  };

  // Rebuild the op with the kept result types and type operands, moving the
  // regions across and re-pointing the kept results.
  auto finalize = [&](Operation *newOp) {
    analysis.opPhases[newOp] = p;
    for (auto [i, idx] : llvm::enumerate(keptIdx)) {
      analysis.actualPhase[newOp->getResult(i)] =
          analysis.actualPhase.at(op->getResult(idx));
      op->getResult(idx).replaceAllUsesWith(newOp->getResult(i));
    }
    forget(op);
    op->erase();
  };

  OpBuilder builder(op);
  if (isa<IfOp, hir::PickOp>(op)) {
    for (auto *region : branchArms(op)) {
      if (region->empty())
        continue;
      auto *terminator = region->front().getTerminator();
      if (auto yieldOp = dyn_cast<YieldOp>(terminator))
        rewriteTerminator(yieldOp);
      else if (auto yieldOp = dyn_cast<hir::YieldOp>(terminator))
        rewriteHirYield(yieldOp);
    }
    if (auto ifOp = dyn_cast<IfOp>(op)) {
      SmallVector<Value> keptTypeOperands;
      for (auto idx : keptIdx)
        keptTypeOperands.push_back(ifOp.getResultTypes()[idx]);
      auto newIf = IfOp::create(builder, loc, keptTypes, ifOp.getCondition(),
                                keptTypeOperands);
      newIf.getThenRegion().takeBody(ifOp.getThenRegion());
      newIf.getElseRegion().takeBody(ifOp.getElseRegion());
      finalize(newIf);
    } else {
      auto pickOp = cast<hir::PickOp>(op);
      auto newPick =
          hir::PickOp::create(builder, loc, keptTypes, pickOp.getChoice(),
                              pickOp.getAlternatives().size());
      for (auto [i, region] : llvm::enumerate(pickOp.getAlternatives()))
        newPick.getAlternatives()[i].takeBody(region);
      finalize(newPick);
    }
  } else {
    auto loopOp = cast<LoopOp>(op);
    // Collect the breaks before rewriting: rewriting erases each in place,
    // which would invalidate a live walk of the body.
    SmallVector<BreakOp> breaks;
    forEachLoopBreak(loopOp,
                     [&](BreakOp breakOp) { breaks.push_back(breakOp); });
    for (auto breakOp : breaks)
      rewriteTerminator(breakOp);
    SmallVector<Value> keptTypeOperands;
    for (auto idx : keptIdx)
      keptTypeOperands.push_back(loopOp.getResultTypes()[idx]);
    auto newLoop = LoopOp::create(builder, loc, keptTypes, loopOp.getInits(),
                                  loopOp.getInitTypes(), keptTypeOperands);
    newLoop.getBody().takeBody(loopOp.getBody());
    finalize(newLoop);
  }
}

/// Forward every single-provider control-flow result whose provider operand is
/// defined outside its op or is a floating pure tree (cloned in front of the
/// op). Sets `changed` when it forwards any result. See the section header for
/// the rule and invariant.
LogicalResult PhaseSorter::forwardSingleProviderResults(bool &changed) {
  // Innermost first, so a forwarded inner result is already an ordinary
  // external operand of its enclosing op when that op is considered (forwarding
  // chains).
  SmallVector<Operation *> ops;
  funcOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, hir::PickOp, LoopOp>(op) && realPhaseOf(op))
      ops.push_back(op);
  });

  for (auto *op : ops) {
    int16_t p = analysis.opPhases.at(op);
    SmallVector<Value> forwardTo(op->getNumResults());
    bool anyForward = false;
    for (auto [idx, result] : llvm::enumerate(op->getResults())) {
      int16_t vph = analysis.actualPhase.at(result);
      if (vph == p || vph == PhaseAnalysis::kNeverPhase)
        continue;
      // Only a single-provider result is a transparent forwarding; a
      // multi-provider selection is left for branch extraction or break-value
      // delivery, and its provider operands are irrelevant here.
      auto providers = collectResultProviders(op, idx);
      if (providers.size() != 1)
        continue;
      Value provider = providers.front();
      if (definedOutside(provider, op)) {
        forwardTo[idx] = provider;
        anyForward = true;
        continue;
      }
      // An interior provider at the float phase is a region-free pure tree,
      // materializable at any point: clone it in front of the op, where the
      // clone dominates every use of the result, and forward to the clone.
      // Any other interior provider is anchored later-phase arm content or an
      // iteration-snapshotted break value; extraction delivers it.
      if (analysis.actualPhase.at(provider) == PhaseAnalysis::kFloatPhase) {
        auto *defOp = provider.getDefiningOp();
        assert(defOp && !isRealPhase(analysis.opPhases.at(defOp)) &&
               hir::isEffectivelyPure(defOp) && defOp->getNumRegions() == 0 &&
               "float-phase provider must be a region-free pure tree");
        OpBuilder builder(op);
        IRMapping mapping;
        SmallVector<Operation *> clonedOps;
        auto clone =
            cloneFloatingTree(op, provider, builder, mapping, clonedOps);
        for (auto *clonedOp : clonedOps)
          clonedOp->walk([&](Operation *cloned) {
            setPhase(cloned, PhaseAnalysis::kFloatPhase);
          });
        forwardTo[idx] = clone;
        anyForward = true;
      }
    }
    if (anyForward) {
      forwardResults(op, forwardTo, p);
      changed = true;
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Hoist Earlier-Phase Content Out of Later-Phase Frames
//===----------------------------------------------------------------------===//
//
// # Hoist Content Earlier Than Its Enclosing Frame
//
// An op whose derived phase is earlier than the phase of a region-bearing frame
// enclosing it does not belong inside that frame: a later-phase frame's branch
// and iteration decisions are made in a later phase and so cannot gate
// earlier-phase execution, which therefore runs unconditionally, exactly once.
// Hoisting moves every such op out to immediately before the frame, in program
// order; its results then dominate the frame and flow back into the interior as
// ordinary SSA uses.
//
// Frames nest, so an op moves out past every enclosing frame that is later than
// it, landing before the outermost such frame — the op then sits at its own
// phase, or is later-phase content of an earlier frame, an extraction residual
// left to the rejection below. A hoisted op that still references a floating
// value defined inside that frame (a constant or region-free pure tree, which
// does not cross as a phase result) has that value's tree cloned in front of
// it; every other interior reference would be a causality violation the
// analysis rejects, and a co-hoisted earlier-phase definition precedes its user
// in program order, so it has already left the frame when the user is cloned.
//
// Hoisting composes with single-provider forwarding in both directions, so the
// two run together to a fixpoint before extraction: forwarding turns an
// interior provider into an external operand and exposes the earlier-phase
// content that consumes it (an earlier value feeding a divergent branch), while
// hoisting an interior provider out of its op turns it into an external
// provider whose result then forwards (a loop whose sole break carries an
// earlier-phase computation). Running the fixpoint before extraction also keeps
// earlier-phase ops from being pulled into a replicate body and replayed once
// per activation.

/// Hoist every op earlier than an enclosing frame out to before that frame.
/// Sets `changed` when it moves any op. See the section header.
LogicalResult PhaseSorter::hoistEarlierContent(bool &changed) {
  // Collect every op strictly earlier than its immediate enclosing frame, in
  // program order, then move each before the outermost enclosing frame that is
  // later than it. Program order preserves the relative order of co-hoisted
  // siblings, so their mutual SSA dependencies stay dominated; a single move to
  // the outermost frame avoids re-cloning a floating value once per level.
  SmallVector<Operation *> hoist;
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (op == funcOp)
      return;
    auto *parent = op->getParentOp();
    if (parent == funcOp)
      return;
    auto opPhase = realPhaseOf(op);
    if (!opPhase)
      return;
    // A real-phase op's parent is a region-bearing frame, which the analysis
    // always stamps with a real phase.
    int16_t parentPhase = analysis.opPhases.at(parent);
    assert(isRealPhase(parentPhase) && "frame parent is never floating");
    if (*opPhase < parentPhase)
      hoist.push_back(op);
  });
  if (hoist.empty())
    return success();
  changed = true;

  for (auto *op : hoist) {
    int16_t opPhase = analysis.opPhases.at(op);
    // The outermost enclosing frame that is later than the op: walk up while
    // each next-outer frame is still strictly later.
    Operation *frame = op->getParentOp();
    while (auto *outer = frame->getParentOp()) {
      if (outer == funcOp || analysis.opPhases.at(outer) <= opPhase)
        break;
      frame = outer;
    }
    op->moveBefore(frame);

    // A floating value the op still references from inside the frame is no
    // longer visible now that the op sits outside; clone its tree in front of
    // the op. A co-hoisted earlier-phase definition already left the frame.
    OpBuilder builder(op);
    op->walk([&](Operation *user) {
      for (auto &use : user->getOpOperands()) {
        auto *defOp = use.get().getDefiningOp();
        if (!defOp || !frame->isProperAncestor(defOp))
          continue;
        assert(!isRealPhase(analysis.opPhases.at(defOp)) &&
               hir::isEffectivelyPure(defOp) && defOp->getNumRegions() == 0 &&
               "hoisted op uses non-floating value from later-phase frame");
        builder.setInsertionPoint(op);
        IRMapping mapping;
        SmallVector<Operation *> clonedOps;
        auto clone =
            cloneFloatingTree(frame, use.get(), builder, mapping, clonedOps);
        for (auto *clonedOp : clonedOps)
          clonedOp->walk([&](Operation *cloned) {
            setPhase(cloned, PhaseAnalysis::kFloatPhase);
          });
        use.set(clone);
      }
    });
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Reject Unimplemented Sorting Machinery
//===----------------------------------------------------------------------===//

/// Reject inputs that need sorting machinery beyond wrapping, call
/// decomposition, and forward crossings, with explicit diagnostics instead of
/// sorting them incorrectly:
///
/// - Anchored content whose phase differs from its parent frame's phase.
///   Hoisting already lifted every op earlier than its frame out, and the
///   joint extraction fixpoint carries later-phase content across any number of
///   boundaries, in both function regions alike — a diverging arm's content
///   included — so a residual is later-phase content the fixpoint settled
///   without lifting: content trapped in a branch none of whose arms rejoins,
///   which no extraction rule owns (there is no reaching arm to dispatch a
///   choice on).
///
/// Cross-phase result delivery, by contrast, is fully covered and asserted
/// rather than rejected: single-provider results were forwarded (externally
/// defined providers directly, floating interior providers via clones,
/// earlier-phase interior providers after hoisting), branch selection
/// obligations were lifted into a choice plus a receiving pick by branch
/// extraction, later-phase loop results were lowered onto receiving-phase
/// variables by break-value delivery, and never-produced results stay unused
/// on their producer (their consumers are already `uir.unreachable_value`).
///
/// Together these establish that every op and every value inside anchored
/// content is at that content's phase, which is what makes the wrap and
/// crossing rules below exact.
LogicalResult PhaseSorter::rejectUnimplemented() {
  auto result = funcOp->walk([&](Operation *op) {
    if (op == funcOp)
      return WalkResult::advance();

    auto phaseIt = analysis.opPhases.find(op);
    assert(phaseIt != analysis.opPhases.end() && "op phase not computed");
    int16_t opPhase = phaseIt->second;

    // Every op delivers each result at its own phase, or holds an unused
    // never result; forwarding, extraction, and break-value delivery
    // established this above.
    if (isRealPhase(opPhase)) {
      for (auto value : op->getResults()) {
        auto valueIt = analysis.actualPhase.find(value);
        assert(valueIt != analysis.actualPhase.end() &&
               "value phase not computed");
        assert((valueIt->second == opPhase ||
                valueIt->second == PhaseAnalysis::kNeverPhase) &&
               "cross-phase result survived forwarding and extraction");
        (void)valueIt;
      }
    }

    // Reject content nested inside a differently-phased frame that neither
    // hoisting nor extraction lifted. Every frame ends with a real phase, so a
    // parent — always a region-bearing frame — is never floating. Hoisting and
    // extraction run on both function regions, so a residual survives here
    // only where no rule owns it: later-phase content trapped in a branch
    // none of whose arms rejoins.
    if (auto *parent = op->getParentOp(); parent != funcOp) {
      auto parentIt = analysis.opPhases.find(parent);
      assert(parentIt != analysis.opPhases.end() && "op phase not computed");
      assert(isRealPhase(parentIt->second) &&
             "a region-bearing parent is never floating");
      if (isRealPhase(opPhase) && opPhase != parentIt->second) {
        if (opPhase < parentIt->second)
          op->emitOpError()
              << "at phase " << opPhase << " is nested inside later phase "
              << parentIt->second
              << " content; hoisting it out is not implemented in phase "
                 "sorting yet";
        else
          op->emitOpError()
              << "at phase " << opPhase << " is nested inside earlier phase "
              << parentIt->second
              << " content; extracting this later-phase content is not "
                 "implemented in phase sorting yet";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

//===----------------------------------------------------------------------===//
// Sort One Region
//===----------------------------------------------------------------------===//

/// Sort the top level of one function region into per-phase `uir.phase` ops.
/// See the file header for the wrap, float, and return rules this applies.
LogicalResult PhaseSorter::sortRegion(Region &region) {
  // The phase analysis only models entry blocks, so the coverage check has
  // already rejected any region with a second block (its ops are unstamped).
  assert(region.hasOneBlock() &&
         "coverage check admits only single-block regions");
  auto &block = region.front();
  auto *terminator = block.getTerminator();
  auto returnOp = dyn_cast<ReturnOp>(terminator);

  // Decide placement for every top-level op, in program order: real-phase
  // ops go to their phase; sentinel-phase ops stay at the region top level,
  // unmoved, where they precede the phase ops and remain visible to every
  // phase. A float is never assigned a phase, and it never references a
  // placed value — a real-phase operand would have anchored the pure op, and
  // never operands were severed — so keeping its position preserves
  // dominance. The map is ordered so phase ops are later created in
  // ascending phase order.
  std::map<int16_t, SmallVector<Operation *>> phaseContent;
  DenseMap<Operation *, int16_t> placement;
  for (auto &op : block.without_terminator()) {
    if (auto target = realPhaseOf(&op)) {
      phaseContent[*target].push_back(&op);
      placement[&op] = *target;
    }
  }

  // Partition the function results by their declared phases. Each result's
  // value and type operand ride on the phase return of that phase; a phase
  // return carrying results is content, so its phase op is materialized
  // even without other content.
  struct PhaseResults {
    SmallVector<Value> values;
    SmallVector<Value> typeOfValues;
  };
  std::map<int16_t, PhaseResults> phaseResults;
  auto resultPhases = funcOp.getResultPhases();
  if (returnOp) {
    assert(returnOp.getValues().size() == resultPhases.size() &&
           "return arity verified against the function declaration");
    for (auto [value, typeOfValue, phase] : llvm::zip(
             returnOp.getValues(), returnOp.getTypeOfValues(), resultPhases)) {
      auto &results = phaseResults[static_cast<int16_t>(phase)];
      results.values.push_back(value);
      results.typeOfValues.push_back(typeOfValue);
      phaseContent.try_emplace(static_cast<int16_t>(phase));
    }
  }

  // Collect the yields of every phase: a value computed inside a phase and
  // consumed in a later one crosses as a phase-op result, yielded once by
  // the defining phase's `uir.phase_return`. To find the yields, attribute
  // every use of a placed op's result to the phase where it lives after
  // sorting: uses inside placed content belong to that content's phase, the
  // dissolving `uir.return`'s operands belong to the declared phase of
  // their function result, and uses by the region terminator sit after all
  // phase ops. With nested phase mismatches rejected up front and never uses
  // severed, no floating op uses a placed value and no use sits at a phase
  // before its definition, so every attributed use is a same-phase use or a
  // forward crossing. Insertion follows program order of the defining ops,
  // keeping the yield lists deterministic.
  std::map<int16_t, llvm::SetVector<Value>> phaseYields;
  for (auto &op : block.without_terminator()) {
    auto placementIt = placement.find(&op);
    if (placementIt == placement.end())
      continue; // Top-level values are visible everywhere; nothing crosses.
    int16_t defPhase = placementIt->second;
    for (auto value : op.getResults()) {
      for (auto &use : value.getUses()) {
        auto *ancestor = block.findAncestorOpInBlock(*use.getOwner());
        assert(ancestor && "use outside the function region");
        // The phase where this use lives after sorting; nullopt for the
        // region terminator, which follows all phase ops.
        std::optional<int16_t> usePhase;
        if (ancestor != terminator) {
          auto it = placement.find(ancestor);
          assert(it != placement.end() &&
                 "floating op cannot use a placed value");
          usePhase = it->second;
        } else if (returnOp) {
          auto numValues = returnOp.getValues().size();
          auto index = use.getOperandNumber();
          usePhase = static_cast<int16_t>(
              resultPhases[index < numValues ? index : index - numValues]);
        }
        if (usePhase == defPhase)
          continue;
        assert((!usePhase || *usePhase > defPhase) &&
               "backward crossing survived analysis and extraction");
        phaseYields[defPhase].insert(value);
      }
    }
  }
  // Every yielding phase has content (at least the value's defining op), so
  // the creation loop below attaches every yield list to a phase op.
  assert(llvm::all_of(phaseYields,
                      [&](const auto &entry) {
                        return phaseContent.count(entry.first);
                      }) &&
         "yielding phase without content");

  // Create the phase ops in ascending phase order before the terminator and
  // move each phase's content into them, preserving program order. Each
  // phase op gets one result per yielded value, provided by its
  // `uir.phase_return`. Remaining top-level ops all precede the phase ops,
  // so direct uses of them from inside the phase regions dominate.
  OpBuilder builder(terminator);
  SmallVector<std::pair<PhaseOp, llvm::SetVector<Value> *>> createdPhaseOps;
  for (auto &[phase, content] : phaseContent) {
    auto loc =
        content.empty() ? terminator->getLoc() : content.front()->getLoc();
    auto &yields = phaseYields[phase];
    SmallVector<Type> resultTypes;
    for (auto value : yields)
      resultTypes.push_back(value.getType());
    auto phaseOp = PhaseOp::create(builder, loc, resultTypes, phase);
    setPhase(phaseOp, phase);
    createdPhaseOps.push_back({phaseOp, &yields});
    auto &phaseBlock = phaseOp.getBody().emplaceBlock();
    for (auto *op : content)
      op->moveBefore(&phaseBlock, phaseBlock.end());

    auto returnBuilder = OpBuilder::atBlockEnd(&phaseBlock);
    auto resultsIt = phaseResults.find(phase);
    auto phaseReturnOp =
        resultsIt != phaseResults.end()
            ? PhaseReturnOp::create(
                  returnBuilder, terminator->getLoc(), resultsIt->second.values,
                  resultsIt->second.typeOfValues, yields.getArrayRef())
            : PhaseReturnOp::create(returnBuilder, loc, ValueRange{},
                                    ValueRange{}, yields.getArrayRef());
    setPhase(phaseReturnOp, phase);
  }

  // With all phase ops in place, redirect every use of a yielded value from
  // outside its phase region to the corresponding phase-op result. This
  // rewires consumers inside later phases, the function-result and type
  // operands captured on later phase returns, and the region terminator;
  // uses inside the defining phase (including its own phase return) keep
  // the value directly.
  for (auto &createdPhaseOp : createdPhaseOps) {
    Operation *phaseOp = createdPhaseOp.first;
    for (auto [index, yielded] :
         llvm::enumerate(createdPhaseOp.second->getArrayRef())) {
      Value value = yielded;
      Value replacement = createdPhaseOp.first.getResult(index);
      value.replaceUsesWithIf(replacement, [&](OpOperand &use) {
        return !phaseOp->isProperAncestor(use.getOwner());
      });
    }
  }

  // The function results now ride on the phase returns; the body terminator
  // becomes unreachable.
  if (returnOp) {
    auto unreachableOp = UnreachableOp::create(builder, returnOp.getLoc());
    setPhase(unreachableOp, 0);
    forget(returnOp);
    returnOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult PhaseSorter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Sorting phases in @" << funcOp.getSymName()
                          << "\n");

  // Normalized input is single-exit in both regions: the body's sole
  // `uir.return` is its terminator and the signature's sole `uir.signature`
  // is its terminator, so no exit op is ever nested in control flow. Every
  // stage below relies on this — dissolution, extraction, and the
  // exit-terminator helpers all treat only `break`/`continue` as exits — so
  // assert it up front, before any transformation runs.
#ifndef NDEBUG
  auto *bodyTerminator = funcOp.getBody().front().getTerminator();
  funcOp.getBody().walk([&](ReturnOp op) {
    assert(op == bodyTerminator &&
           "expected normalized single-exit input: the sole uir.return is the "
           "body terminator (run normalize-exits first)");
  });
  auto *sigTerminator = funcOp.getSignature().front().getTerminator();
  funcOp.getSignature().walk([&](SignatureOp op) {
    assert(op == sigTerminator &&
           "expected normalized single-exit input: the sole uir.signature is "
           "the signature terminator (run normalize-exits first)");
  });
#endif

  if (failed(rejectSortedVocabulary()))
    return failure();
  if (failed(checkCoverage()))
    return failure();
  if (failed(dissolveExprsAndPins()))
    return failure();
  if (failed(decomposeCalls()))
    return failure();
  if (failed(forwardNeverValues()))
    return failure();
  // Single-provider forwarding and hoisting expose each other, so run them to a
  // fixpoint before extraction: forwarding an interior provider away lets an
  // earlier-phase consumer hoist, and hoisting an interior provider out lets
  // its op's result forward.
  while (true) {
    bool changed = false;
    if (failed(forwardSingleProviderResults(changed)))
      return failure();
    if (failed(hoistEarlierContent(changed)))
      return failure();
    if (!changed)
      break;
  }
  // Branch and record extraction share one fixpoint. Each rule emits receiving
  // `hir.pick`s and `hir.replicate`s that may themselves hold or deliver
  // later-phase content, and every such op re-extracts at its next boundary
  // exactly like user-written CF, regardless of which rule emitted it: a wide
  // receiving pick choice-ifies again, and a replicate still holding content
  // later than its own phase re-records it onto a fresh hit list. A single
  // sweep of each rule moves every result's delivery and every frame's later
  // content one phase forward, strictly reducing its phase distance, so
  // alternating the sweeps until neither changes terminates. A frame whose
  // later content still sits inside same-phase non-exit CF (deposited there by
  // a nested frame's extraction mid-sweep) defers without setting `changed`;
  // the next sweep's branch extraction lifts the deposit, and anything no rule
  // ever lifts is rejected below once the fixpoint settles.
  while (true) {
    bool changed = false;
    if (failed(extractBranches(changed)))
      return failure();
    if (failed(extractFrames(changed)))
      return failure();
    if (!changed)
      break;
  }
  if (failed(rejectUnimplemented()))
    return failure();
  if (failed(sortRegion(funcOp.getSignature())))
    return failure();
  if (failed(sortRegion(funcOp.getBody())))
    return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct SortPhasesPass : public uir::impl::SortPhasesPassBase<SortPhasesPass> {
  void runOnOperation() override {
    // The symbol table resolves call decomposition's callee lookups. Sorting
    // only rewrites function interiors, never adds or removes symbols, so one
    // table stays valid across all functions.
    SymbolTable symbolTable(getOperation());
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      PhaseAnalysis analysis(funcOp);
      if (failed(analysis.run()))
        return signalPassFailure();
      if (failed(PhaseSorter(funcOp, analysis, symbolTable).run()))
        return signalPassFailure();
    }
  }
};
} // namespace
