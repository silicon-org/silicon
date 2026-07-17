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
//   sentinels) stays at the region top level, unless it uses a value
//   defined by an op that moved into a phase region — then it sinks into
//   the latest such phase region, after its operands; operands from
//   earlier phases cross like any other use. Such ops are dead (any live
//   consumer would have anchored them through demand), so their placement
//   is only a dominance concern.
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
// `uir.if` with later-phase content or a choice-ified result is decomposed
// into a choice-yielding branch plus a receiving `hir.pick`; and every
// earlier-phase `uir.loop` with later-phase content is decomposed into a hit
// list of per-activation records plus a receiving `hir.replicate`:
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
// - **Extract branches**: a `uir.if` at phase p whose arms hold later-phase
//   content, or whose result selects between later-phase values (both arms
//   yield it, regardless of the values' identity), is rebuilt at phase p with
//   a per-arm `hir.choice` as an extra result; the later-phase content and
//   results move into a `hir.pick` at phase p+1 that dispatches on the
//   choice. The choice crosses phases as an ordinary crossing. Only
//   multi-provider results are lifted this way; single-provider divergent
//   results stay rejected below, and never-produced results are handled by
//   never-forwarding (their consumers are already `uir.unreachable_value`).
//
// - **Extract loops**: a `uir.loop` at phase p whose body holds later-phase
//   content, or whose result is at a later phase, is rebuilt at phase p
//   pushing one record per activation onto a hit list, and the later-phase
//   content moves into a `hir.replicate` at phase p+1 that replays the
//   records. Loop-phase early exits (`break`/`continue` in branch arms) make
//   the record end in a nested exit-path choice that a nested `hir.pick` in
//   the replicate dispatches on. Every arm of such a branch is itself a block
//   that decomposes recursively — an exit-carrying arm, or a rejoining arm that
//   hides a nested exit, deepens the record's chain and the replicate's picks
//   by one more level. When rejoining arms hold later-phase content, the
//   rebuilt branch yields an arm-choice tagging which rejoining arm ran and
//   carrying its snapshots; the choice rides the `continued` link and a nested
//   `hir.pick` in the replicate replays that arm's content before the following
//   segment. When such a rejoining arm also yields a demanded later-phase
//   result, that arm-pick is made valued: each alternative yields the arm's
//   operand for the result (produced by the arm's replayed content), and the
//   pick's result reconstructs the value for the following segment. The list
//   crosses phases as an ordinary crossing. A loop delivering a later-phase
//   result first has that break value choice-ified: each break site carries a
//   `hir.choice` of the value's snapshots in place of the value, and a
//   `hir.pick` at phase p+1 after the loop reconstructs it (composing with
//   record extraction).
//
// A `never`-phased value (a control-flow result no path provides) is
// non-viral: before extraction, every real-phase use of one is rewritten to a
// `uir.unreachable_value` at the consuming phase, so the `never` result stays
// unused on its producer and never crosses a boundary.
//
// The pass covers straight-line content, single-boundary branch extraction,
// single-boundary loop-record extraction, and break-value choices. Inputs that
// need the remaining sorting machinery are rejected with explicit diagnostics
// instead of being sorted incorrectly: early function exits (function-level
// exit chains), results delivered at a phase different from their op's own
// phase by single-provider control flow (which needs result forwarding),
// hoisting and content more than one phase away from its control flow, and
// backward uses from floating content nested under earlier-phase control flow.
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

/// One exit path of a divergence: a break/continue terminator that ends an
/// activation, together with the recursive decomposition of the arm block that
/// leads to it. A straight-line arm decomposes into a single content segment
/// and no nested divergences (the flat form); an arm that itself carries a
/// nested exit-carrying branch decomposes recursively, so the exit path's
/// record and the receiving pick nest one level deeper per nested divergence.
struct ExitArm {
  Operation *terminator;
  std::unique_ptr<BodyPlan> plan;
};

/// One rejoining arm of a divergence: an arm that falls back out (its terminal
/// yields) instead of ending the activation. Such an arm may hold later-phase
/// content that runs only on the fall-through path; that content is
/// choice-ified so that arm-interior samples still cross. `content` is the
/// fall-through content (empty for a rejoining arm that runs none), and
/// `fields` are the values it samples. An arm that both rejoins and hides a
/// nested exit appears here (for its fall-through content) and in `exits` (for
/// its exit records): its exit structure and its fall-through content are
/// disjoint.
struct RejoinArm {
  Region *region;
  SmallVector<Operation *> content;
  llvm::SetVector<Value> fields;
  /// The arm's yield operands for the divergence's demanded later-phase
  /// results, in `laterResults` order. Their defining ops are part of `content`
  /// (they are the later-phase content this arm runs), so the receiving
  /// arm-pick reconstructs each result by yielding these values.
  SmallVector<Value> laterYieldOperands;
};

/// An exit-carrying op in a loop body — or in a nested exit arm — with
/// later-phase content: activations diverge here in how much of that content
/// they instantiate, and every exit path contributes one chain link to the
/// activation's record. Divergences without any later-phase content carry one
/// exit path per exit terminator with a trivial (empty, divergence-free) plan.
///
/// `exits` holds the arms that end the activation (their terminal exits, or
/// they hide a nested exit); `rejoins` holds the arms that fall back out. An
/// arm can appear in both: it hides a nested exit (an `exits` entry that builds
/// the exit records) yet still falls through with content (a `rejoins` entry).
/// When any rejoining arm holds later-phase content the rebuilt `if` yields an
/// `hir.choice` (the arm-choice) tagging which rejoining arm ran and capturing
/// its samples; the choice rides the divergence's `continued` chain link, and a
/// nested `hir.pick` in the receiving `continued` alternative replays that
/// arm's content before the following segment. `armChoice` stays null when no
/// rejoining arm runs later-phase content.
struct Divergence {
  IfOp ifOp;
  SmallVector<ExitArm> exits;
  SmallVector<RejoinArm> rejoins;
  mlir::Value armChoice;
  /// The divergence `if`'s demanded later-phase results (phase > p). A phase-p
  /// branch cannot deliver a later value, so the rebuilt `if` drops these and
  /// the receiving arm-pick reconstructs them from the rejoining arms' yield
  /// operands. The values held here are throw-away placeholders standing in for
  /// the original results between the `if` rebuild and the arm-pick build; the
  /// arm-pick replaces and erases each one.
  SmallVector<Value, 0> laterResults;
};

/// The recursive decomposition of a block — a loop body or an exit arm — into
/// content segments separated by exit-carrying divergences. `segments` holds
/// the later-phase ops between consecutive divergences in program order (one
/// more entry than `divergences`); `segFields` are the crossing values each
/// segment samples, deduplicated against every enclosing level. `terminal` is
/// the block terminator that ends the fully-continued path: a loop body's
/// back-edge continue/break (or unreachable), or an exit arm's own
/// break/continue.
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
  LogicalResult extractBranches();
  LogicalResult extractFromBranch(IfOp ifOp, int16_t p);
  LogicalResult extractLoops();
  LogicalResult extractFromLoop(LoopOp loopOp, int16_t p);
  FailureOr<LoopOp> extractBreakValues(LoopOp loopOp, int16_t p);
  LogicalResult scanBlock(Operation *loopOp, Block &block, int16_t p,
                          BodyPlan &plan);
  bool blockAllExits(Block &block, int16_t p);
  void collectPlanFields(Operation *loopOp, int16_t p, BodyPlan &plan,
                         ArrayRef<Operation *> allMoved,
                         llvm::SetVector<Value> visible);
  static void collectPlanContent(BodyPlan &plan,
                                 SmallVectorImpl<Operation *> &content);
  void rebuildRejoins(BodyPlan &plan, int16_t p);
  Value rebuildRejoiningIf(Divergence &divergence, int16_t p);
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

/// The sorted-phase ops are this pass's output vocabulary: they are only
/// ever produced by sorting itself, and their presence in the input means
/// the pipeline is out of order. Reject them loudly instead of treating
/// them as opaque content, which would silently mis-sort their phase
/// structure (the phase analysis does not model them).
LogicalResult PhaseSorter::rejectSortedVocabulary() {
  auto result = funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<PhaseOp, PhaseReturnOp, PhaseCallOp, UnreachableValueOp>(op))
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
/// vocabulary (`uir.phase` interiors) and receiving constructs (`hir.pick`,
/// `hir.replicate`) are illegal in the input. Abort loudly before mutating
/// anything.
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
/// exprs dissolve before their parents. Return-terminated exprs (later-phase
/// early exits) are left intact; the early-exit rejection below reports them.
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
    if (!yieldOp) {
      if (isa<ReturnOp>(block.getTerminator()))
        continue;
      return exprOp.emitOpError()
             << "cannot be dissolved: body is terminated by "
             << block.getTerminator()->getName() << " instead of a yield";
    }
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
// backward-crossing or a spurious yield would result. This step severs those
// uses ahead of any placement: for every operand of a real-phase op whose value
// is `never`, it materializes a fresh `uir.unreachable_value` right before the
// consumer, at the phase where sorting will attribute that use, and rewrites
// the operand to it. The `never` result is left on its producer, now unused by
// real-phase content (a dead float consumer, if any, keeps it — see the sorter
// float rules). The invariant established: no real-phase op operand carries a
// `never` value, so wrapping and crossing never encounter one.
//
// The use-attribution phase mirrors `sortRegion`: every op consumes its
// operands at its own phase, except `uir.return`, which distributes its value
// and type operands onto the `uir.phase_return` of each result's declared
// phase — so a return operand is consumed at that result's declared phase, not
// the return op's own (always 0). Float and never consumers are left alone;
// they are dead and never placed.

/// Replace every real-phase use of a `never`-phased value with a fresh
/// `uir.unreachable_value` at the consuming phase. See the section header.
LogicalResult PhaseSorter::forwardNeverValues() {
  auto resultPhases = funcOp.getResultPhases();

  // The phase at which sorting attributes operand `operand` of real-phase op
  // `op`: the op's own phase, except a `uir.return` fans each value/type
  // operand onto the phase_return of that result's declared phase.
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
    if (op != funcOp && realPhaseOf(op))
      ops.push_back(op);
  });

  auto anyTy = hir::AnyType::get(funcOp.getContext());
  for (auto *op : ops) {
    // One unreachable value per consuming phase suffices for all of this op's
    // never operands that land there.
    DenseMap<int16_t, Value> perPhase;
    for (auto &operand : op->getOpOperands()) {
      auto it = analysis.actualPhase.find(operand.get());
      assert(it != analysis.actualPhase.end() &&
             "real-phase op operand has no computed phase");
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
// A `uir.if` at phase p whose arms hold later-phase content, or whose result
// selects between later-phase values, cannot wrap wholesale into
// phase p: the later-phase content must run in a later phase, and a later
// value cannot be produced at phase p (causality). Branch extraction rewrites
// such an if into two cooperating ops that the ordinary wrap and cross rules
// then place:
//
// - The if is rebuilt at phase p. Its kept results (those already at phase p)
//   stay, and it gains one extra phase-p result per arm: an `hir.choice`
//   tagging the taken arm and carrying, as its payload, the arm-local values
//   the moved content samples. Later-phase results are removed from the if.
// - One `hir.pick` is created right after the if, at phase p+1, dispatching
//   on the choice. Each alternative holds that arm's moved later-phase
//   content, reconstructing the payload from block arguments, and yields the
//   arm's contribution to the removed later-phase results.
//
// The choice is an ordinary phase-p value; the cross rule yields it out of
// phase p and the pick consumes it in phase p+1 like any other crossing. The
// distinction between an obligation to choice-ify and the neighbouring
// divergent-result cases is local: a later-phase result is choice-ified only
// when more than one arm yield provides it — the values' identity is
// irrelevant. A single-provider result (the other arm exits) is not extracted;
// the residual if still delivers it divergently, so the rejection below reports
// it. A never-produced result is not extracted either, but its consumers were
// already rewritten to `uir.unreachable_value` by never-forwarding, so it is
// simply left unused. Extraction thus lifts exactly the choice-ification case.

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

/// The first terminator at phase <= p that exits the region (return, or
/// break/continue targeting a loop outside the region), or null if every
/// path yields. Branch extraction bails on arms with such exits: they belong
/// to function-level exit chains or the enclosing loop's records, not to
/// branch extraction.
Operation *PhaseSorter::findExitTerminator(Region &region, int16_t p) {
  Operation *found = nullptr;
  std::function<void(Region &, bool)> visit = [&](Region &r, bool loopScoped) {
    for (auto &block : r) {
      for (auto &op : block) {
        if (found)
          return;
        bool isExit =
            isa<ReturnOp>(op) || (!loopScoped && isa<BreakOp, ContinueOp>(op));
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
/// the region instead of yielding, in walk order. Break/continue inside loops
/// (including the queried region itself when it belongs to a loop) target
/// those loops and do not exit; returns exit the function from any depth.
void PhaseSorter::collectExitTerminators(Region &region, int16_t p,
                                         SmallVectorImpl<Operation *> &exits) {
  std::function<void(Region &, bool)> visit = [&](Region &r, bool loopScoped) {
    for (auto &block : r) {
      for (auto &op : block) {
        bool isExit =
            isa<ReturnOp>(op) || (!loopScoped && isa<BreakOp, ContinueOp>(op));
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

/// The number of arm yields providing result `idx` of `ifOp`. An arm
/// terminated by an exit (return/break/continue) rather than a yield does not
/// provide the result. A result provided by more than one yield is a
/// selection and a choice-ification obligation when it is later-phase; a
/// single-provider result (the other arm exits) is not choice-ified but left
/// for the early-exit rejection.
static unsigned countResultProviders(IfOp ifOp, unsigned idx) {
  unsigned count = 0;
  for (auto *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    if (region->empty())
      continue;
    auto yieldOp = dyn_cast<YieldOp>(region->front().getTerminator());
    if (yieldOp && idx < yieldOp.getValues().size())
      ++count;
  }
  return count;
}

/// Extract later-phase content out of every phase-p `uir.if` whose interior
/// or results demand it, innermost first so a nested if becomes ordinary
/// later-phase content of its enclosing branch.
///
/// An if is extracted only when every one of its results is deliverable at
/// its own phase after extraction: results already at phase p stay, and
/// results at a later phase are removed and reconstructed by the pick — but
/// only if they choice-ify (provided by more than one arm yield). A result
/// diverging from the op's phase in any other way (a single-provider result
/// whose other arm exits, or a never-produced result whose short-yielding arms
/// would break the kept-result rebuild) is not extracted; the residual if is
/// left for the divergence rejection, except a never result, whose consumers
/// never-forwarding already rewrote, so it is simply left unused.
LogicalResult PhaseSorter::extractBranches() {
  SmallVector<IfOp> ifOps;
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](IfOp op) {
    if (realPhaseOf(op))
      ifOps.push_back(op);
  });

  for (auto ifOp : ifOps) {
    int16_t p = *realPhaseOf(ifOp);

    // Classify the results. Extraction is viable only if none diverge except
    // by choice-ification.
    bool extractable = true;
    bool hasChoiceResult = false;
    for (auto result : ifOp.getResults()) {
      int16_t vph = analysis.actualPhase.at(result);
      if (vph == p)
        continue; // Kept at the op's phase.
      if (isRealPhase(vph) && vph > p &&
          countResultProviders(ifOp, result.getResultNumber()) > 1) {
        hasChoiceResult = true;
        continue; // Choice-ification obligation.
      }
      // Single-provider divergence, or a never result (unused after
      // never-forwarding but still un-extractable: its short-yielding arms
      // carry no value for the kept-result rebuild).
      extractable = false;
      break;
    }
    if (!extractable)
      continue;
    if (!hasChoiceResult && !hasLaterContent(ifOp, p))
      continue; // Nothing to extract; wraps wholesale.

    if (failed(extractFromBranch(ifOp, p)))
      return failure();
  }
  return success();
}

/// Rewrite one phase-p `uir.if` into a choice-yielding if plus a receiving
/// `hir.pick` at phase p+1. See the section header for the shape and the
/// invariant.
LogicalResult PhaseSorter::extractFromBranch(IfOp ifOp, int16_t p) {
  Operation *op = ifOp;
  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());

  // Arms that exit instead of yielding belong to the enclosing loop's records
  // (break/continue) or to function-level exit chains (return); neither is
  // branch extraction's concern. Leave the if for the corresponding rejection.
  for (auto *arm : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    if (arm->empty())
      continue;
    if (findExitTerminator(*arm, p))
      return success();
  }

  // Materialize an else region for an else-less if so the not-taken tag has a
  // constructor.
  if (ifOp.getElseRegion().empty()) {
    auto &elseBlock = ifOp.getElseRegion().emplaceBlock();
    auto builder = OpBuilder::atBlockEnd(&elseBlock);
    auto yield = YieldOp::create(builder, loc, ValueRange{}, ValueRange{});
    setPhase(yield, p);
  }
  SmallVector<Region *> arms = {&ifOp.getThenRegion(), &ifOp.getElseRegion()};

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

    auto yieldOp = cast<YieldOp>(arm->front().getTerminator());
    for (auto idx : laterIdx)
      if (idx < yieldOp.getValues().size())
        info.laterYieldOperands.push_back(yieldOp.getValues()[idx]);

    // The later yield operands cross the boundary too: collect their crossing
    // needs as if the yield were moved content.
    for (auto value : info.laterYieldOperands) {
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

  // Construct the per-arm choice and rewrite each yield to carry the kept
  // values plus the choice.
  OpBuilder builder(op);
  auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
  setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
  Value choiceType = choiceTypeOp.getResult();

  for (auto [tag, arm] : llvm::enumerate(arms)) {
    auto yieldOp = cast<YieldOp>(arm->front().getTerminator());
    OpBuilder termBuilder(yieldOp);
    auto choice = hir::ChoiceOp::create(termBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        armInfos[tag].fields.getArrayRef());
    setPhase(choice, p);

    SmallVector<Value> values, typeOfValues;
    for (auto idx : keptIdx) {
      values.push_back(yieldOp.getValues()[idx]);
      typeOfValues.push_back(yieldOp.getTypeOfValues()[idx]);
    }
    values.push_back(choice.getResult());
    typeOfValues.push_back(choiceType);
    auto newYield = YieldOp::create(termBuilder, loc, values, typeOfValues);
    setPhase(newYield, p);
    forget(yieldOp);
    yieldOp->erase();
  }

  // Rebuild the if with the kept results plus the choice result.
  SmallVector<Type> newResultTypes(keptIdx.size() + 1, anyTy);
  SmallVector<Value> newTypeOperands;
  for (auto idx : keptIdx)
    newTypeOperands.push_back(ifOp.getResultTypes()[idx]);
  newTypeOperands.push_back(choiceType);
  auto newIf = IfOp::create(builder, loc, newResultTypes, ifOp.getCondition(),
                            newTypeOperands);
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  setPhase(newIf, p);
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newIf.getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newIf.getResult(i));
  }
  Value choiceResult = newIf.getResult(keptIdx.size());
  analysis.actualPhase[choiceResult] = p;

  // Create the receiving pick after the if, one alternative per arm,
  // reconstructing the later results.
  builder.setInsertionPointAfter(newIf);
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
    auto altYield =
        hir::YieldOp::create(altBuilder, loc, info.laterYieldOperands);
    analysis.opPhases[altYield] = p + 1;
    remapFields(altBlock, info.fields);
    cloneFloatingUses(newIf, altBlock);
  }

  forget(op);
  op->erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Loop-Record Extraction
//===----------------------------------------------------------------------===//
//
// # Extract Later-Phase Content Out of Earlier-Phase Loops
//
// A `uir.loop` at phase p whose body holds later-phase content cannot wrap
// wholesale into phase p: the later-phase content runs once per activation of
// the loop, so the phase-p execution must tell phase p+1 *how often* it is
// instantiated, and with which per-iteration snapshots. Loop-record extraction
// rewrites such a loop into two cooperating constructs the wrap and cross
// rules then place:
//
// - A hit list (`hir.opaque_list_create`) is created at phase p, right before
//   the loop. Every activation packs a record of the crossing values it
//   samples and pushes it onto the list; the record is built in place at the
//   terminator that ends the activation. The list is an ordinary phase-p value
//   that crosses to phase p+1 like any other crossing.
// - One `hir.replicate` at phase p+1 replays the records, holding the loop's
//   later-phase content. It reconstructs the per-iteration snapshots from the
//   record fields.
//
// Loop bodies with loop-phase early exits (`break`/`continue` inside branch
// arms) diverge in how much later-phase content each activation runs. The body
// decomposes into content segments separated by exit-carrying divergence
// points, and each record ends in a nested exit-path choice: each exit
// terminator packs and pushes its own record in place — the shared prefix
// segment, then one `continued` link per divergence it fell past, ending in
// its own exit tag. The receiving replicate mirrors this shape with nested
// picks (`function-exit-choices.md` §3, sampling-model.md §6).
//
// The decomposition is recursive: every arm of a divergence is a block like the
// loop body. An exit-carrying arm, or a rejoining arm that hides a nested exit,
// is itself scanned into a sub-plan, so a nested exit-carrying branch inside it
// is another divergence. Its exit tag's payload is not a flat sample list but
// the arm's own record body (the arm's first-segment samples plus a nested
// chain over the arm's own divergences), and the receiving exit alternative is
// the arm sub-plan's own level structure, one pick deeper. The flat form is the
// base case — an exit arm with a single content segment and no nested
// divergence.
//
// A divergence's rejoining arms (the ones that fall back out) may hold
// later-phase content. Reaching the `continued` link means exactly one
// rejoining arm ran, so no `not_ran` tag is needed: the arm-choice tags which
// one, one tag per rejoining arm. The content is choice-ified rather than
// folded into the following segment, so that samples defined arm-interior — not
// in scope at the activation-end record push — still cross. The rebuilt `if`
// yields the arm-choice; it rides the `continued` link as one field, and a
// nested pick in the receiving `continued` alternative replays the taken arm's
// content before the following segment, in program order. A rejoining arm that
// both hides a nested exit and falls through with content keeps its exit
// structure (in the exit records) disjoint from its fall-through content (in
// the arm-choice); the fall-through content is required to sit entirely after
// the arm's exit divergences so the two never share an op.
//
// When a rejoining arm also yields a demanded later-phase result of the `if`,
// the rebuilt `if` cannot deliver that value (a phase-p branch produces no
// later value), so the result is dropped and reconstructed in the receiving
// phase: the arm-pick that replays the arm's content is made valued, each
// alternative yielding the arm's operand for the result, and the pick's result
// stands in for the value in the following segment. This reduces the result to
// a sample carried by the arm's own replayed content — no separate machinery.
// It applies only when the rejoining content produces the result; a value from
// outside the arm needs plain single-provider forwarding, unimplemented.
//
// Only single-boundary loop-record extraction is handled here (later-phase
// loop results are choice-ified by break-value extraction first): returns
// under loops (function-level exit chains), a rejoining arm with content shared
// across an exit and its fall-through or with a nested rejoining divergence in
// its fall-through, single-provider results a rejoining arm's content does not
// produce, and content more than one phase later (which needs the replicate
// re-extracted at the next boundary) are rejected below and by
// `rejectUnimplemented`.

// NOLINTBEGIN(misc-no-recursion)

/// Whether every path through `block` ends the activation with a phase-<=p exit
/// (a `break`/`continue`/`return`) rather than reaching the block's yield
/// terminal. A block all-exits when it holds a top-level exit or an `if` whose
/// both arms all-exit; nested loops are not descended into (their exits target
/// the inner loop). Such a block's yield is unreachable, so an arm that
/// all-exits contributes no fall-through even though its terminal is a yield.
bool PhaseSorter::blockAllExits(Block &block, int16_t p) {
  for (auto &op : block) {
    if (auto ph = realPhaseOf(&op);
        ph && *ph <= p && isa<BreakOp, ContinueOp, ReturnOp>(op))
      return true;
    if (auto ifOp = dyn_cast<IfOp>(&op))
      if (!ifOp.getThenRegion().empty() && !ifOp.getElseRegion().empty() &&
          blockAllExits(ifOp.getThenRegion().front(), p) &&
          blockAllExits(ifOp.getElseRegion().front(), p))
        return true;
  }
  return false;
}

/// Decompose a block — a loop body or an exit arm — into content segments
/// separated by exit-carrying divergences. `plan.segments` holds the
/// later-phase ops between consecutive divergences in program order (one more
/// entry than `plan.divergences`); `plan.terminal` is the block terminator that
/// ends the fully-continued path.
///
/// An exit-carrying arm is a block like any other, so it is scanned
/// recursively: a nested exit-carrying branch inside it becomes a nested
/// divergence whose exit paths build their own nested chain links. A divergence
/// with no later-phase content anywhere carries one exit path per exit
/// terminator with a trivial plan, no matter how deeply the terminator is
/// nested; only content-bearing divergences must be branches and recurse.
LogicalResult PhaseSorter::scanBlock(Operation *loopOp, Block &block, int16_t p,
                                     BodyPlan &plan) {
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
      // Non-exit CF at phase p was extracted by the innermost-first walk
      // already; leftover later-phase content here is a compiler bug.
      if (hasLaterContent(&blockOp, p))
        return blockOp.emitOpError()
               << "contains later-phase content that inner extraction did "
                  "not consume";
      continue;
    }

    // A return under a loop with later-phase content is a function-level exit
    // chain no matter how it is nested; those belong to distribution.
    for (auto *term : exitTerms)
      if (isa<ReturnOp>(term))
        return term->emitOpError()
               << "exits the function under a loop with later-phase content; "
                  "function-level exit chains are not implemented in phase "
                  "sorting yet";

    Divergence divergence;
    if (!hasLaterContent(&blockOp, p)) {
      // No later-phase content anywhere in the divergence: every exit path
      // contributes an empty chain link built at its terminator with a trivial
      // plan, no matter how deeply the terminator is nested.
      for (auto *term : exitTerms) {
        ExitArm arm;
        arm.terminator = term;
        arm.plan = std::make_unique<BodyPlan>();
        arm.plan->terminal = term;
        arm.plan->segments.emplace_back();
        divergence.exits.push_back(std::move(arm));
      }
    } else {
      // Later-phase content inside the divergence must be attributed to the
      // paths that run it, which requires a branch whose arms name the paths.
      auto ifOp = dyn_cast<IfOp>(&blockOp);
      if (!ifOp)
        return blockOp.emitOpError()
               << "carries a loop exit but is not a branch; cannot build "
                  "partial activation records";
      divergence.ifOp = ifOp;

      for (auto &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        auto &armBlock = region.front();
        auto *term = armBlock.getTerminator();
        auto termPhase = realPhaseOf(term);
        bool isExitTerm =
            termPhase && *termPhase <= p && isa<BreakOp, ContinueOp>(term);
        if (isExitTerm) {
          // An exit arm is a block like the loop body: decompose it recursively
          // so a nested exit-carrying branch inside it builds its own nested
          // chain link.
          ExitArm arm;
          arm.terminator = term;
          arm.plan = std::make_unique<BodyPlan>();
          if (failed(scanBlock(loopOp, armBlock, p, *arm.plan)))
            return failure();
          divergence.exits.push_back(std::move(arm));
          continue;
        }
        // The arm's terminal yields: it is a rejoining candidate. Absent a
        // nested exit it is a pure rejoining arm whose top-level later-phase
        // content is choice-ified (the base case); content nested inside
        // arm-interior CF is caught by the nested-phase rejection later.
        if (!findExitTerminator(region, p)) {
          RejoinArm rejoin;
          rejoin.region = &region;
          rejoin.content = collectLaterContent(armBlock, p);
          divergence.rejoins.push_back(std::move(rejoin));
          continue;
        }

        // The rejoining arm hides a nested exit: it is a block like the loop
        // body, so decompose it recursively. Its nested exits end the
        // activation through the outer loop (exit records) and its
        // fall-through, if reachable, contributes to the continued link. A
        // nested return is a function-level exit chain and is rejected inside
        // the recursive scan.
        auto subPlan = std::make_unique<BodyPlan>();
        if (failed(scanBlock(loopOp, armBlock, p, *subPlan)))
          return failure();

        // If every path through the arm exits (its yield is unreachable), it is
        // a pure exit-capable arm: its content rides the exit records and it
        // contributes nothing to the continued link. Otherwise its fall-through
        // is reachable and its content must be cleanly separable from its exit
        // structure.
        if (blockAllExits(armBlock, p)) {
          ExitArm arm;
          arm.terminator = term;
          arm.plan = std::move(subPlan);
          divergence.exits.push_back(std::move(arm));
          continue;
        }

        // A reachable fall-through carrying a nested exit: the exit records and
        // the fall-through replay must move disjoint content into disjoint
        // receiving blocks. Content before or between the exit divergences runs
        // on both an exit path and the fall-through (shared prefix), and a
        // nested rejoining divergence in the fall-through would need its own
        // arm-choice. Both require duplicating content across receiving blocks;
        // reject them.
        for (size_t s = 0; s + 1 < subPlan->segments.size(); ++s)
          if (!subPlan->segments[s].empty())
            return blockOp.emitOpError()
                   << "has a rejoining arm with later-phase content before a "
                      "nested exit; shared content across an exit and its fall-"
                      "through is not implemented in phase sorting yet";
        for (auto &subDiv : subPlan->divergences)
          if (!subDiv.rejoins.empty())
            return blockOp.emitOpError()
                   << "has a rejoining arm whose fall-through carries a nested "
                      "rejoining divergence; not implemented in phase sorting "
                      "yet";

        // Split the arm: its final segment is the fall-through content (moved
        // to a rejoining arm), and the emptied sub-plan is the exit structure.
        // The exit records then carry no fall-through content, and the replay
        // moves it only into the continued arm-choice alternative.
        RejoinArm rejoin;
        rejoin.region = &region;
        rejoin.content = std::move(subPlan->segments.back());
        subPlan->segments.back().clear();
        divergence.rejoins.push_back(std::move(rejoin));
        ExitArm arm;
        arm.terminator = term;
        arm.plan = std::move(subPlan);
        divergence.exits.push_back(std::move(arm));
      }
    }

    // A demanded later-phase result of the divergence is reconstructed by the
    // arm-pick that replays the rejoining arms' later content: each rejoining
    // arm must produce every such result within its own replayed content, so
    // the arm-pick's alternative can yield it. A later result the arm-pick
    // cannot rebuild — a divergence with no rejoining content, or a value from
    // outside the providing arm's content — needs plain single-provider result
    // forwarding, which is not implemented.
    SmallVector<unsigned> laterResultIdx;
    for (auto result : blockOp.getResults()) {
      auto it = analysis.actualPhase.find(result);
      if (it != analysis.actualPhase.end() && isRealPhase(it->second) &&
          it->second > p)
        laterResultIdx.push_back(result.getResultNumber());
    }
    if (!laterResultIdx.empty()) {
      bool reconstructible = !divergence.rejoins.empty();
      for (auto &rejoin : divergence.rejoins) {
        if (!reconstructible)
          break;
        DenseSet<Operation *> contentSet(rejoin.content.begin(),
                                         rejoin.content.end());
        auto yieldOp = cast<YieldOp>(rejoin.region->front().getTerminator());
        for (auto idx : laterResultIdx) {
          auto *defOp = idx < yieldOp.getValues().size()
                            ? yieldOp.getValues()[idx].getDefiningOp()
                            : nullptr;
          if (!defOp || !contentSet.contains(defOp)) {
            reconstructible = false;
            break;
          }
        }
      }
      if (!reconstructible)
        return blockOp.emitOpError()
               << "carries a loop exit and a later-phase result its rejoining "
                  "content does not produce; single-provider result forwarding "
                  "is not implemented in phase sorting yet";
    }

    plan.divergences.push_back(std::move(divergence));
    plan.segments.emplace_back();
  }
  return success();
}

/// Collect every op the plan moves into the receiving replicate — segment
/// content, exit-arm sub-plan content (recursively), and rejoining-arm content.
/// These count as moved for the crossing analysis, so a value one segment
/// samples that another produces does not spuriously become a record field.
void PhaseSorter::collectPlanContent(BodyPlan &plan,
                                     SmallVectorImpl<Operation *> &content) {
  for (auto &segment : plan.segments)
    content.append(segment.begin(), segment.end());
  for (auto &divergence : plan.divergences) {
    for (auto &arm : divergence.exits)
      collectPlanContent(*arm.plan, content);
    for (auto &rejoin : divergence.rejoins)
      content.append(rejoin.content.begin(), rejoin.content.end());
  }
}

/// Collect the crossing values of every segment, exit-arm sub-plan, and
/// rejoining arm of `plan`, deduplicating each level's fields against the
/// values already visible from its enclosing levels. `visible` is the set of
/// values the enclosing continued spine already carries; the spine extends it
/// segment by segment, and each exit arm receives a copy so its own extension
/// stays local to its subtree (sibling exits and the continued path never see
/// an exit arm's interior fields).
void PhaseSorter::collectPlanFields(Operation *loopOp, int16_t p,
                                    BodyPlan &plan,
                                    ArrayRef<Operation *> allMoved,
                                    llvm::SetVector<Value> visible) {
  plan.segFields.assign(plan.segments.size(), {});
  size_t numDivs = plan.divergences.size();
  for (auto [k, segment] : llvm::enumerate(plan.segments)) {
    plan.segFields[k] = collectCrossingValues(loopOp, p, segment, allMoved);
    plan.segFields[k].set_subtract(visible);
    visible.set_union(plan.segFields[k]);
    // The previous divergence's rejoining content runs in this level (its
    // `continued` alternative), nested under the levels up to and including
    // this one; collect each rejoining arm's samples now that those fields are
    // visible. Every rejoining arm of one divergence samples at the same level,
    // so they dedup against the same visible set (a copy each keeps sibling
    // arms' interior fields local).
    if (k >= 1)
      for (auto &rejoin : plan.divergences[k - 1].rejoins) {
        rejoin.fields =
            collectCrossingValues(loopOp, p, rejoin.content, allMoved);
        rejoin.fields.set_subtract(visible);
      }
    // Each exit arm of this level's divergence is nested under the levels up to
    // and including this one; scan its sub-plan against a copy of the current
    // visible set.
    if (k < numDivs)
      for (auto &arm : plan.divergences[k].exits)
        collectPlanFields(loopOp, p, *arm.plan, allMoved, visible);
  }
}

/// Rebuild every exit-carrying `if` in `plan` whose rejoining arm holds
/// later-phase content so it yields an arm-choice, recursing into exit-arm
/// sub-plans first so a nested divergence's `if` is rebuilt before its
/// enclosing arm moves under a new parent.
void PhaseSorter::rebuildRejoins(BodyPlan &plan, int16_t p) {
  for (auto &divergence : plan.divergences) {
    for (auto &arm : divergence.exits)
      rebuildRejoins(*arm.plan, p);
    if (llvm::any_of(divergence.rejoins,
                     [](RejoinArm &r) { return !r.content.empty(); }))
      divergence.armChoice = rebuildRejoiningIf(divergence, p);
  }
}

// NOLINTEND(misc-no-recursion)

/// Rebuild an exit-carrying `if` whose rejoining arms hold later-phase content
/// so each rejoining arm yields an arm-choice capturing that content's samples.
///
/// The arm-choice is a single phase-p value the rebuilt `if` delivers as its
/// last result: each rejoining arm yields `choice tag(samples…)`, where the tag
/// is the arm's index among the rejoining arms and the payload is the values
/// its fall-through content samples. Reaching the divergence's `continued` link
/// means exactly one rejoining arm ran, so the tag names it — no `not_ran`
/// alternative is needed. Each choice is built inside its arm, right before the
/// yield, so it captures arm-interior samples that are out of scope at the
/// activation-end record push. Exit arms are untouched: they break or continue
/// and never reach a yield, so they provide no arm-choice tag and the choice is
/// consumed only on the `continued` path where a rejoining arm ran. Returns the
/// arm-choice value.
///
/// A demanded later-phase result of the `if` (a value the rejoining arms yield
/// at phase p+1) cannot be delivered by a phase-p branch. Such results are
/// dropped from the rebuilt `if`; each rejoining arm's yield operand for a
/// later result is recorded so the receiving arm-pick reconstructs it (the
/// operand's producer is that arm's later-phase content, replayed there). The
/// original results are replaced by throw-away placeholders that the arm-pick
/// build later rewrites to its own results and erases.
Value PhaseSorter::rebuildRejoiningIf(Divergence &divergence, int16_t p) {
  IfOp ifOp = divergence.ifOp;
  auto loc = ifOp.getLoc();
  auto anyTy = hir::AnyType::get(ifOp.getContext());

  // Partition the results: those at phase p stay on the rebuilt `if`; those at
  // a later phase are reconstructed by the receiving arm-pick.
  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : ifOp.getResults()) {
    int16_t vph = analysis.actualPhase.at(result);
    if (isRealPhase(vph) && vph > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }

  OpBuilder builder(ifOp);
  auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
  setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
  Value choiceType = choiceTypeOp.getResult();

  // Build the arm-choice inside each rejoining arm, right before its yield, and
  // rewrite the yield to carry the kept values plus the choice, dropping the
  // later-result operands (recorded per arm for the arm-pick reconstruction).
  // The tag is the arm's index among the rejoining arms.
  for (auto [tag, rejoin] : llvm::enumerate(divergence.rejoins)) {
    auto &armBlock = rejoin.region->front();
    auto yieldOp = cast<YieldOp>(armBlock.getTerminator());
    for (auto idx : laterIdx) {
      assert(idx < yieldOp.getValues().size() &&
             "rejoining arm underprovides a demanded later result");
      rejoin.laterYieldOperands.push_back(yieldOp.getValues()[idx]);
    }
    OpBuilder armBuilder(yieldOp);
    auto choice = hir::ChoiceOp::create(armBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        rejoin.fields.getArrayRef());
    setPhase(choice, p);
    SmallVector<Value> yieldValues, yieldTypes;
    for (auto idx : keptIdx) {
      yieldValues.push_back(yieldOp.getValues()[idx]);
      yieldTypes.push_back(yieldOp.getTypeOfValues()[idx]);
    }
    yieldValues.push_back(choice.getResult());
    yieldTypes.push_back(choiceType);
    auto newYield = YieldOp::create(armBuilder, loc, yieldValues, yieldTypes);
    setPhase(newYield, p);
    // A rejoining arm that also hides a nested exit keeps its exit structure in
    // `exits`; that sub-plan's terminal is this same yield, so refresh it to
    // the replacement before the old yield is erased.
    Operation *oldYield = yieldOp.getOperation();
    for (auto &exitArm : divergence.exits)
      if (exitArm.plan->terminal == oldYield)
        exitArm.plan->terminal = newYield.getOperation();
    forget(yieldOp);
    yieldOp.erase();
  }

  // Rebuild the `if` with the kept results plus one extra result for the
  // arm-choice. Every if result is an `!hir.any`; the `resultTypes` operand
  // carries the per-result type operands.
  SmallVector<Type> resultTypes(keptIdx.size() + 1, anyTy);
  SmallVector<Value> typeOperands;
  for (auto idx : keptIdx)
    typeOperands.push_back(ifOp.getResultTypes()[idx]);
  typeOperands.push_back(choiceType);
  auto newIf = IfOp::create(builder, loc, resultTypes, ifOp.getCondition(),
                            typeOperands);
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  setPhase(newIf, p);
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newIf.getResult(i)] =
        analysis.actualPhase.at(ifOp.getResult(idx));
    ifOp.getResult(idx).replaceAllUsesWith(newIf.getResult(i));
  }
  // The dropped later results have no phase-p producer to point at yet; stand
  // them up with placeholders the arm-pick build replaces and erases.
  for (auto idx : laterIdx) {
    auto placeholder =
        hir::OpaquePackOp::create(builder, loc, anyTy, ValueRange{});
    setPhase(placeholder, static_cast<int16_t>(p + 1));
    ifOp.getResult(idx).replaceAllUsesWith(placeholder.getResult());
    divergence.laterResults.push_back(placeholder.getResult());
  }
  Value armChoice = newIf.getResult(newIf.getNumResults() - 1);
  analysis.actualPhase[armChoice] = p;
  divergence.ifOp = newIf;
  forget(ifOp);
  ifOp.erase();
  return armChoice;
}

//===----------------------------------------------------------------------===//
// Break-Value Extraction
//===----------------------------------------------------------------------===//
//
// # Extract Later-Phase Loop Results as Break-Value Choices
//
// A `uir.loop` at phase p whose break carries a later-phase value cannot
// deliver that value at phase p: the value materializes once the loop has run
// to completion, in a later phase (causality, sampling-model.md §7). Break-
// value extraction replaces the value at each break site with a phase-p
// `hir.choice` tagging the site and carrying, as its payload, the loop-local
// values the value's computation samples. The loop is rebuilt with the choice
// as an extra phase-p result, and one `hir.pick` at phase p+1 after the loop
// reconstructs the value: each site's value slice moves into that site's
// alternative, which recomputes the value from the choice payload.
//
// The choice is an ordinary phase-p value; the cross rule yields it out of the
// loop's phase and the pick consumes it in phase p+1 like any other crossing.
// Only the plain shape is lifted: a value slice shared between multiple break
// sites, or between a break value and other later-phase content, is rejected.

/// Extract the later-phase results of a phase-p `uir.loop` as break-value
/// choices. See the section header for the shape and invariants. Returns the
/// rebuilt loop; the loop is returned unchanged when it has no later-phase
/// result.
FailureOr<LoopOp> PhaseSorter::extractBreakValues(LoopOp loopOp, int16_t p) {
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

  // Collect this loop's break sites, not descending into nested loops (whose
  // breaks target the inner loop).
  SmallVector<BreakOp> breaks;
  std::function<void(Block &)> collectBreaks = [&](Block &block) {
    for (auto &blockOp : block) {
      if (isa<LoopOp>(&blockOp))
        continue;
      if (auto breakOp = dyn_cast<BreakOp>(&blockOp)) {
        breaks.push_back(breakOp);
        continue;
      }
      for (auto &region : blockOp.getRegions())
        for (auto &inner : region)
          collectBreaks(inner);
    }
  };
  collectBreaks(loopOp.getBody().front());

  // Per break site: the later-phase values it provides, the transitive later-
  // phase ops computing them (the value slice, which moves into the pick
  // alternative), and the crossing values that become the choice payload.
  struct SiteInfo {
    SmallVector<Value> laterValues;
    SmallVector<Operation *> slice;
    llvm::SetVector<Value> fields;
  };
  SmallVector<SiteInfo> sites;
  DenseSet<Operation *> allSliceOps;
  for (auto breakOp : breaks) {
    SiteInfo info;
    // Break arity is not a verifier invariant against the loop's results, so a
    // break need not carry every result. Every later result (reconstructed by
    // the pick) and every kept result (rebuilt onto the new break) is indexed
    // positionally below, so require the break to supply all of them here.
    for (auto result : op->getResults()) {
      unsigned idx = result.getResultNumber();
      if (idx >= breakOp.getValues().size()) {
        breakOp.emitOpError() << "does not provide the loop's result " << idx;
        return failure();
      }
    }
    for (auto idx : laterIdx)
      info.laterValues.push_back(breakOp.getValues()[idx]);

    // Backward closure over later-phase ops inside the loop feeding the value.
    DenseSet<Operation *> sliceSet;
    SmallVector<Value> worklist(info.laterValues);
    while (!worklist.empty()) {
      auto *defOp = worklist.pop_back_val().getDefiningOp();
      if (!defOp || !op->isProperAncestor(defOp))
        continue;
      if (auto ph = realPhaseOf(defOp); !ph || *ph <= p)
        continue;
      if (!sliceSet.insert(defOp).second)
        continue;
      // A slice op shared with another break site is rejected by the shared-
      // content check below on the earlier site (the op's value feeds that
      // other exit, which is content outside this site's slice), so this site
      // never reaches an op already claimed by an earlier one.
      bool fresh = allSliceOps.insert(defOp).second;
      assert(fresh && "shared break-value slice escaped the shared-content "
                      "check on an earlier break site");
      (void)fresh;
      defOp->walk([&](Operation *inner) {
        for (auto operand : inner->getOperands())
          worklist.push_back(operand);
      });
    }

    // Slice ops in program order, outermost units only; reject slices
    // entangled with content that stays behind.
    loopOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *walked) {
      if (sliceSet.contains(walked)) {
        info.slice.push_back(walked);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto *sliceOp : info.slice)
      for (auto *user : sliceOp->getUsers())
        if (user != breakOp && !sliceSet.contains(user) &&
            llvm::none_of(sliceSet,
                          [&](Operation *s) { return s->isAncestor(user); })) {
          sliceOp->emitOpError()
              << "is shared between a break value and other later-phase "
                 "content; not implemented in phase sorting yet";
          return failure();
        }

    info.fields = collectCrossingValues(op, p, info.slice, info.slice);
    for (auto value : info.laterValues) {
      auto *defOp = value.getDefiningOp();
      if (defOp && op->isProperAncestor(defOp) && !sliceSet.contains(defOp) &&
          isRealPhase(analysis.opPhases.at(defOp)))
        info.fields.insert(value);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto *owner = blockArg.getOwner()->getParentOp();
        if (owner == op || op->isProperAncestor(owner))
          info.fields.insert(value);
      }
    }
    sites.push_back(std::move(info));
  }

  // Construct the per-site choice and rewrite each break to carry it in place
  // of the later-phase values.
  OpBuilder builder(op);
  auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
  setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
  Value choiceType = choiceTypeOp.getResult();

  for (auto [tag, breakOp] : llvm::enumerate(breaks)) {
    OpBuilder breakBuilder(breakOp);
    auto choice = hir::ChoiceOp::create(breakBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        sites[tag].fields.getArrayRef());
    setPhase(choice, p);
    SmallVector<Value> values, typeOfValues;
    for (auto idx : keptIdx) {
      values.push_back(breakOp.getValues()[idx]);
      typeOfValues.push_back(breakOp.getTypeOfValues()[idx]);
    }
    values.push_back(choice.getResult());
    typeOfValues.push_back(choiceType);
    auto newBreak = BreakOp::create(breakBuilder, loc, values, typeOfValues);
    setPhase(newBreak, p);
    forget(breakOp);
    breakOp.erase();
  }

  // Rebuild the loop with the kept results plus the choice result.
  SmallVector<Type> newResultTypes(keptIdx.size() + 1, anyTy);
  SmallVector<Value> newTypeOperands;
  for (auto idx : keptIdx)
    newTypeOperands.push_back(loopOp.getResultTypes()[idx]);
  newTypeOperands.push_back(choiceType);
  auto newLoop = LoopOp::create(builder, loc, newResultTypes, loopOp.getInits(),
                                loopOp.getInitTypes(), newTypeOperands);
  newLoop.getBody().takeBody(loopOp.getBody());
  setPhase(newLoop, p);
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newLoop.getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newLoop.getResult(i));
  }
  Value choiceResult = newLoop.getResult(keptIdx.size());
  analysis.actualPhase[choiceResult] = p;

  // Create the receiving pick right after the loop. Record extraction (in
  // extractFromLoop below) inserts its replicate between the loop and the
  // pick, so the break values are reconstructed after all records.
  builder.setInsertionPointAfter(newLoop);
  SmallVector<Type> pickResultTypes(laterIdx.size(), anyTy);
  auto pick = hir::PickOp::create(builder, loc, pickResultTypes, choiceResult,
                                  sites.size());
  analysis.opPhases[pick] = p + 1;
  for (auto [i, idx] : llvm::enumerate(laterIdx)) {
    analysis.actualPhase[pick.getResult(i)] =
        analysis.actualPhase.at(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(pick.getResult(i));
  }

  for (auto [tag, site] : llvm::enumerate(sites)) {
    auto &altBlock = pick.getAlternatives()[tag].emplaceBlock();
    for (unsigned i = 0; i < site.fields.size(); ++i)
      altBlock.addArgument(anyTy, loc);
    for (auto *sliceOp : site.slice)
      sliceOp->moveBefore(&altBlock, altBlock.end());
    OpBuilder altBuilder(&altBlock, altBlock.end());
    auto altYield = hir::YieldOp::create(altBuilder, loc, site.laterValues);
    analysis.opPhases[altYield] = p + 1;
    remapFields(altBlock, site.fields);
    cloneFloatingUses(newLoop, altBlock);
  }

  forget(op);
  op->erase();
  return newLoop;
}

/// Extract the later-phase content of a phase-p `uir.loop` into a hit list and
/// a receiving `hir.replicate` at phase p+1. See the section header for the
/// shape and invariants. Every activation pushes exactly one record, built in
/// place at the terminator that ends it: the record chains a `continued` link
/// per divergence the activation fell past and ends in the taken exit path's
/// choice, whose payload is empty when that path ran no later-phase content.
LogicalResult PhaseSorter::extractFromLoop(LoopOp loopOp, int16_t p) {
  auto loc = loopOp.getLoc();
  auto anyTy = hir::AnyType::get(loopOp.getContext());

  // Later-phase loop results are break-value choices: replace each break site's
  // later value with a phase-p choice and reconstruct it in a receiving pick
  // after the loop. Extract these first, so their value slices are out of the
  // way of the record analysis below; the loop is rebuilt in the process.
  auto rebuilt = extractBreakValues(loopOp, p);
  if (failed(rebuilt))
    return failure();
  loopOp = *rebuilt;
  Operation *op = loopOp;

  // Break-value extraction may have consumed the only later-phase content (the
  // whole later part was the break value); then nothing remains to record-
  // extract.
  if (!hasLaterContent(op, p))
    return success();

  // A loop body whose activation ends in a return is a function-level exit
  // chain, not loop-record extraction. Catch it before scanning; nested returns
  // are caught inside `scanBlock`.
  auto *bodyTerm = loopOp.getBody().front().getTerminator();
  if (isa<ReturnOp>(bodyTerm))
    return bodyTerm->emitOpError()
           << "terminates a loop body with later-phase content; "
              "function-level exit chains are not implemented in phase "
              "sorting yet";
  if (!isa<BreakOp, ContinueOp, UnreachableOp>(bodyTerm))
    return bodyTerm->emitOpError()
           << "unsupported loop body terminator for record extraction";

  // Decompose the loop body into the recursive plan of content segments and
  // exit-carrying divergences, then collect the crossing values every segment,
  // exit-arm sub-plan, and rejoining arm samples. Everything the plan moves
  // counts as moved for the crossing analysis.
  BodyPlan plan;
  if (failed(scanBlock(op, loopOp.getBody().front(), p, plan)))
    return failure();

  SmallVector<Operation *> allMoved;
  collectPlanContent(plan, allMoved);
  collectPlanFields(op, p, plan, allMoved, {});

  // With the fields known, rebuild every exit-carrying `if` whose rejoining arm
  // holds later-phase content so it yields an arm-choice capturing that
  // content's samples; the choice rides the `continued` links below.
  rebuildRejoins(plan, p);

  // Hit list before the op.
  OpBuilder builder(op);
  auto list = hir::OpaqueListCreateOp::create(builder, loc, anyTy);
  setPhase(list, p);

  // # Build the record pushes
  //
  // Every terminator that ends an activation — each exit arm's break/continue,
  // however deeply nested, and the loop's own back edge — pushes exactly one
  // record, built in place. The record encodes the statically-known path from
  // the loop body root to that terminator: the root segment's samples, then one
  // chain link per divergence the path fell past. An exit path's link is a
  // choice tagged by the taken arm; its payload is the arm's own record body
  // (the arm sub-plan's first-segment samples plus, recursively, the arm's
  // nested chain). A fully-continued path's link is the `continued` tag (one
  // past the exit tags) carrying the following segment's samples and, for a
  // divergence with rejoining content, its arm-choice.
  //
  // `wrapContinued` wraps a chain that is divergence `level`'s choice with the
  // `continued` links of the earlier divergences of `pl`, yielding the chain at
  // the plan's outermost divergence. `emit` walks a plan, recursing into each
  // exit arm with a wrapper that turns the arm's chain into the enclosing
  // exited choice and threads it up to the enclosing plan's record.
  auto wrapContinued = [&](BodyPlan &pl, int level, Value chain,
                           OpBuilder &b) -> Value {
    for (int i = level - 1; i >= 0; --i) {
      SmallVector<Value> link(pl.segFields[i + 1].getArrayRef());
      if (pl.divergences[i].armChoice)
        link.push_back(pl.divergences[i].armChoice);
      link.push_back(chain);
      auto continued = hir::ChoiceOp::create(
          b, loc, anyTy, static_cast<uint32_t>(pl.divergences[i].exits.size()),
          link);
      setPhase(continued, p);
      chain = continued.getResult();
    }
    return chain;
  };

  using Wrap = std::function<void(OpBuilder &, Value)>;
  auto rootWrap = [&](OpBuilder &b, Value chain) {
    SmallVector<Value> recordFields(plan.segFields[0].getArrayRef());
    if (chain)
      recordFields.push_back(chain);
    auto record = hir::OpaquePackOp::create(b, loc, anyTy, recordFields);
    setPhase(record, p);
    auto push = hir::OpaqueListPushOp::create(b, loc, list.getResult(),
                                              record.getResult());
    setPhase(push, p);
  };

  std::function<void(BodyPlan &, const Wrap &)> emit = [&](BodyPlan &pl,
                                                           const Wrap &wrap) {
    size_t numDivs = pl.divergences.size();
    // Each exit arm's terminators wrap their sub-plan chain into this
    // divergence's exited choice, then up through this plan's `continued`
    // links.
    for (size_t k = 0; k < numDivs; ++k) {
      auto &divergence = pl.divergences[k];
      for (size_t a = 0; a < divergence.exits.size(); ++a) {
        BodyPlan &armPlan = *divergence.exits[a].plan;
        Wrap armWrap = [&, k, a](OpBuilder &b, Value subChain) {
          SmallVector<Value> payload(armPlan.segFields[0].getArrayRef());
          if (subChain)
            payload.push_back(subChain);
          auto exited = hir::ChoiceOp::create(
              b, loc, anyTy, static_cast<uint32_t>(a), payload);
          setPhase(exited, p);
          Value chain =
              wrapContinued(pl, static_cast<int>(k), exited.getResult(), b);
          wrap(b, chain);
        };
        emit(armPlan, armWrap);
      }
    }
    // The plan's own terminal ends the fully-continued path; an unreachable
    // terminal (a loop whose every path exits) pushes no record there.
    if (!isa<BreakOp, ContinueOp>(pl.terminal))
      return;
    OpBuilder b(pl.terminal);
    if (numDivs == 0) {
      wrap(b, Value());
      return;
    }
    auto &last = pl.divergences[numDivs - 1];
    SmallVector<Value> payload(pl.segFields[numDivs].getArrayRef());
    if (last.armChoice)
      payload.push_back(last.armChoice);
    auto tail = hir::ChoiceOp::create(
        b, loc, anyTy, static_cast<uint32_t>(last.exits.size()), payload);
    setPhase(tail, p);
    Value chain =
        wrapContinued(pl, static_cast<int>(numDivs) - 1, tail.getResult(), b);
    wrap(b, chain);
  };
  emit(plan, rootWrap);

  // # Build the receiving replicate
  //
  // The replicate mirrors the record shape: one level block per segment along
  // the continued spine, connected by picks over the chain links. A divergence
  // pick's exit alternative is the exit arm sub-plan's own level-0 block —
  // recursively built — and its `continued` alternative is the next spine
  // level. In the flat case (an exit arm with no nested divergence) the
  // recursion bottoms out at a single-segment sub-plan: the alternative simply
  // holds the arm's content and yields.
  builder.setInsertionPointAfter(op);
  auto replicate = hir::ReplicateOp::create(builder, loc, list.getResult());
  analysis.opPhases[replicate] = p + 1;

  SmallVector<std::pair<Block *, const llvm::SetVector<Value> *>> remapBlocks;
  std::function<void(BodyPlan &, Block *)> buildLevels = [&](BodyPlan &pl,
                                                             Block *block0) {
    Block *levelBlock = block0;
    size_t numDivs = pl.divergences.size();
    for (size_t k = 0; k < pl.segments.size(); ++k) {
      bool hasChain = k < numDivs;
      // Block args mirror the record link this level consumes: the segment's
      // samples, then (for a `continued` alternative whose divergence has
      // rejoining content) the arm-choice, then the chain into the next level.
      for (unsigned i = 0; i < pl.segFields[k].size(); ++i)
        levelBlock->addArgument(anyTy, loc);
      Divergence *rejoin = (k >= 1 && pl.divergences[k - 1].armChoice)
                               ? &pl.divergences[k - 1]
                               : nullptr;
      Value armChoiceArg;
      if (rejoin) {
        armChoiceArg = levelBlock->addArgument(anyTy, loc);
        analysis.actualPhase[armChoiceArg] = p;
      }
      Value chainArg;
      if (hasChain) {
        chainArg = levelBlock->addArgument(anyTy, loc);
        analysis.actualPhase[chainArg] = p;
      }
      remapBlocks.push_back({levelBlock, &pl.segFields[k]});

      // Replay the previous divergence's rejoining content before this segment,
      // via a nested pick over its arm-choice: one alternative per rejoining
      // arm, holding that arm's fall-through content. Reaching this `continued`
      // link means exactly one rejoining arm ran, so its alternative fires.
      // When the divergence carried demanded later-phase results, the arm-pick
      // is valued: each alternative yields that arm's operands for those
      // results, and the pick's results reconstruct them for the following
      // segment.
      if (rejoin) {
        OpBuilder rejoinBuilder(levelBlock, levelBlock->end());
        SmallVector<Type> armPickResultTypes(rejoin->laterResults.size(),
                                             anyTy);
        auto armPick =
            hir::PickOp::create(rejoinBuilder, loc, armPickResultTypes,
                                armChoiceArg, rejoin->rejoins.size());
        analysis.opPhases[armPick] = p + 1;
        for (auto [tag, rejoinArm] : llvm::enumerate(rejoin->rejoins)) {
          auto &altBlock = armPick.getAlternatives()[tag].emplaceBlock();
          for (unsigned i = 0; i < rejoinArm.fields.size(); ++i)
            altBlock.addArgument(anyTy, loc);
          remapBlocks.push_back({&altBlock, &rejoinArm.fields});
          for (auto *moved : rejoinArm.content)
            moved->moveBefore(&altBlock, altBlock.end());
          OpBuilder altBuilder(&altBlock, altBlock.end());
          auto altYield = hir::YieldOp::create(altBuilder, loc,
                                               rejoinArm.laterYieldOperands);
          analysis.opPhases[altYield] = p + 1;
        }
        // Replace each dropped later result's placeholder with the arm-pick's
        // reconstructed result and drop the placeholder op.
        for (auto [j, placeholder] : llvm::enumerate(rejoin->laterResults)) {
          Operation *placeholderOp = placeholder.getDefiningOp();
          analysis.actualPhase[armPick.getResult(j)] =
              static_cast<int16_t>(p + 1);
          placeholder.replaceAllUsesWith(armPick.getResult(j));
          forget(placeholderOp);
          placeholderOp->erase();
        }
      }

      for (auto *moved : pl.segments[k])
        moved->moveBefore(levelBlock, levelBlock->end());

      Block *nextLevel = nullptr;
      if (hasChain) {
        auto &divergence = pl.divergences[k];
        OpBuilder levelBuilder(levelBlock, levelBlock->end());
        auto pick = hir::PickOp::create(levelBuilder, loc, TypeRange{},
                                        chainArg, divergence.exits.size() + 1);
        analysis.opPhases[pick] = p + 1;
        for (size_t a = 0; a < divergence.exits.size(); ++a) {
          auto &altBlock = pick.getAlternatives()[a].emplaceBlock();
          buildLevels(*divergence.exits[a].plan, &altBlock);
        }
        nextLevel =
            &pick.getAlternatives()[divergence.exits.size()].emplaceBlock();
      }
      OpBuilder levelBuilder(levelBlock, levelBlock->end());
      auto levelYield = hir::YieldOp::create(levelBuilder, loc, ValueRange{});
      analysis.opPhases[levelYield] = p + 1;
      levelBlock = nextLevel;
    }
  };
  buildLevels(plan, &replicate.getBody().emplaceBlock());

  // Remap fields outermost-first so that deduplicated inner uses resolve to
  // the enclosing level's arguments, then clone floating trees once over the
  // whole structure.
  for (auto [block, fields] : remapBlocks)
    remapFields(*block, *fields);
  cloneFloatingUses(op, replicate.getBody().front());
  return success();
}

/// Extract later-phase content out of every phase-p `uir.loop` whose interior
/// or results demand it, innermost first so a nested loop becomes ordinary
/// later-phase content of its enclosing loop's records. A later-phase result
/// is an extraction obligation on its own: its break values need not be
/// computed inside the loop body.
LogicalResult PhaseSorter::extractLoops() {
  SmallVector<LoopOp> loops;
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](LoopOp op) {
    if (realPhaseOf(op))
      loops.push_back(op);
  });

  for (auto loopOp : loops) {
    int16_t p = *realPhaseOf(loopOp);
    bool hasLaterResult = llvm::any_of(loopOp->getResults(), [&](Value r) {
      int16_t vph = analysis.actualPhase.at(r);
      return isRealPhase(vph) && vph > p;
    });
    if (!hasLaterResult && !hasLaterContent(loopOp, p))
      continue;
    if (failed(extractFromLoop(loopOp, p)))
      return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Reject Unimplemented Sorting Machinery
//===----------------------------------------------------------------------===//

/// Format a phase for diagnostics, naming the sentinels like the phase
/// analysis annotation does.
static std::string phaseName(int16_t phase) {
  if (phase == PhaseAnalysis::kFloatPhase)
    return "float";
  if (phase == PhaseAnalysis::kNeverPhase)
    return "never";
  return std::to_string(phase);
}

/// Reject inputs that need sorting machinery beyond wrapping, call
/// decomposition, and forward crossings, with explicit diagnostics instead of
/// sorting them incorrectly:
///
/// - Early function exits (a `uir.return` anywhere but as the body region's
///   terminator, including returns still riding inside undissolved expr
///   carriers), which need function-level exit chains.
/// - Ops delivering a result at a phase different from their own phase by a
///   transparent forwarding of a single-provider value (the other arms exit).
///   These need result forwarding. Choice-ification obligations (a result
///   provided by more than one terminator at a later phase) are already
///   lifted by branch and break-value extraction above, and never-produced
///   results are skipped here (their consumers are already
///   `uir.unreachable_value`, so the result is unused), so the residual
///   divergence is single-provider results, plus the receiving picks of
///   crossings more than one phase wide (whose re-extraction at the next
///   boundary this pass does not do).
/// - Anchored content whose phase differs from its parent op's phase, which
///   needs hoisting across the phase boundary (branch content is handled by
///   branch extraction and single-boundary loop content by loop-record
///   extraction above; content more than one phase later than its loop is
///   left here for the next-boundary re-extraction that this pass does not do).
///
/// Together these establish that every op and every value inside anchored
/// content is at that content's phase, which is what makes the wrap and
/// crossing rules below exact.
LogicalResult PhaseSorter::rejectUnimplemented() {
  auto *bodyTerminator = funcOp.getBody().front().getTerminator();
  auto result = funcOp->walk([&](Operation *op) {
    if (op == funcOp)
      return WalkResult::advance();
    if (isa<ReturnOp>(op) && op != bodyTerminator) {
      op->emitOpError() << "is an early function exit; function-level exit "
                           "chains are not implemented in phase sorting yet";
      return WalkResult::interrupt();
    }

    auto phaseIt = analysis.opPhases.find(op);
    assert(phaseIt != analysis.opPhases.end() && "op phase not computed");
    int16_t opPhase = phaseIt->second;

    // Reject results delivered at a phase other than the op's own.
    if (isRealPhase(opPhase)) {
      for (auto [index, value] : llvm::enumerate(op->getResults())) {
        auto valueIt = analysis.actualPhase.find(value);
        assert(valueIt != analysis.actualPhase.end() &&
               "value phase not computed");
        // A never result stays on its producer: its real-phase consumers were
        // rewritten to `uir.unreachable_value` by never-forwarding, so the
        // result is unused (or used only by dead float content) and needs no
        // forwarding of its own.
        if (valueIt->second == PhaseAnalysis::kNeverPhase)
          continue;
        if (valueIt->second != opPhase) {
          op->emitOpError()
              << "delivers result #" << index << " at phase "
              << phaseName(valueIt->second) << " instead of its own phase "
              << phaseName(opPhase)
              << "; result forwarding (single-provider control-flow results, "
                 "and receivers of crossings more than one phase wide) is not "
                 "implemented in phase sorting yet";
          return WalkResult::interrupt();
        }
      }
    }

    // Reject anchored content nested inside differently-phased content.
    // Every frame now ends with a real phase, so a parent — always a
    // region-bearing frame — is never floating: content whose phase differs
    // from its parent's needs hoisting or extraction across the boundary,
    // which this pass does not implement.
    if (auto *parent = op->getParentOp(); parent != funcOp) {
      auto parentIt = analysis.opPhases.find(parent);
      assert(parentIt != analysis.opPhases.end() && "op phase not computed");
      assert(isRealPhase(parentIt->second) &&
             "a region-bearing parent is never floating");
      if (isRealPhase(opPhase) && opPhase != parentIt->second) {
        op->emitOpError()
            << "at phase " << opPhase << " is nested inside phase "
            << parentIt->second
            << " content; hoisting and extraction across phase "
               "boundaries are not implemented in phase sorting yet";
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
  // ops go to their phase, sentinel-phase ops sink after any operand that
  // moves, and stay at the top level otherwise. The map is ordered so phase
  // ops are later created in ascending phase order.
  std::map<int16_t, SmallVector<Operation *>> phaseContent;
  DenseMap<Operation *, int16_t> placement;
  for (auto &op : block.without_terminator()) {
    std::optional<int16_t> target = realPhaseOf(&op);
    if (!target) {
      // Sink after the latest operand that moves into a phase region. Also
      // consider operands of nested ops, whose defs may sit at the top
      // level.
      op.walk([&](Operation *user) {
        for (auto operand : user->getOperands()) {
          auto *defOp = operand.getDefiningOp();
          if (!defOp)
            continue;
          auto *ancestor = block.findAncestorOpInBlock(*defOp);
          assert(ancestor && "operand defined outside the function region");
          if (ancestor == &op)
            continue;
          auto it = placement.find(ancestor);
          if (it != placement.end())
            target = std::max(target.value_or(it->second), it->second);
        }
      });
    }
    if (target) {
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
  // phase ops. With divergent results and nested phase mismatches rejected
  // up front, the only uses left at a phase before the defining op's
  // placement are inside floating (dead) content nested under
  // earlier-phase control flow, which cannot sink and is rejected.
  // Insertion follows program order of the defining ops, keeping the yield
  // lists deterministic.
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
                 "floating op using a placed value must have sunk");
          usePhase = it->second;
        } else if (returnOp) {
          auto numValues = returnOp.getValues().size();
          auto index = use.getOperandNumber();
          usePhase = static_cast<int16_t>(
              resultPhases[index < numValues ? index : index - numValues]);
        }
        if (usePhase == defPhase)
          continue;
        if (usePhase && *usePhase < defPhase) {
          auto diag = use.getOwner()->emitOpError()
                      << "uses a value from phase " << defPhase
                      << " inside phase " << *usePhase
                      << " content; backward crossings are not implemented "
                         "in phase sorting yet";
          diag.attachNote(value.getLoc()) << "value defined here";
          return failure();
        }
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
  if (failed(extractBranches()))
    return failure();
  if (failed(extractLoops()))
    return failure();
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
