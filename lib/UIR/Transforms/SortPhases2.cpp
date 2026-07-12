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
//   content, or whose result selects between distinct later-phase values, is
//   rebuilt at phase p with a per-arm `hir.choice` as an extra result; the
//   later-phase content and results move into a `hir.pick` at phase p+1 that
//   dispatches on the choice. The choice crosses phases as an ordinary
//   crossing. Only genuine choice-ification is lifted this way; identity-
//   provided and never-produced divergent results stay rejected below.
//
// - **Extract loops**: a `uir.loop` at phase p whose body holds later-phase
//   content is rebuilt at phase p pushing one record per activation onto a hit
//   list, and the later-phase content moves into a `hir.replicate` at phase
//   p+1 that replays the records. Loop-phase early exits (`break`/`continue`
//   in branch arms) make the record end in a nested exit-path choice that a
//   nested `hir.pick` in the replicate dispatches on. The list crosses phases
//   as an ordinary crossing. A loop delivering a later-phase result first has
//   that break value choice-ified: each break site carries a `hir.choice` of
//   the value's snapshots in place of the value, and a `hir.pick` at phase p+1
//   after the loop reconstructs it (composing with record extraction).
//
// The pass covers straight-line content, single-boundary branch extraction,
// single-boundary loop-record extraction, and break-value choices. Inputs that
// need the remaining sorting machinery are rejected with explicit diagnostics
// instead of being sorted incorrectly: early function exits (function-level
// exit chains), results delivered at a phase different from their op's own
// phase by identity-provided or never-produced control flow (which need result
// forwarding and unreachable-consumer handling), hoisting and content more
// than one phase away from its control flow, and backward uses from floating
// content nested under earlier-phase control flow.
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
/// One exit path of a divergence point: a break/continue terminator at the
/// loop's phase, together with the later-phase content on its path and the
/// crossing values that content samples; both are empty for paths that run
/// no later-phase content.
struct ExitPath {
  Operation *terminator;
  SmallVector<Operation *> content;
  llvm::SetVector<Value> fields;
};

/// An exit-carrying op in a loop body with later-phase content: activations
/// diverge here in how much of that content they instantiate, and every
/// exit path contributes one chain link to the activation's record.
struct Divergence {
  SmallVector<ExitPath> exits;
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
  LogicalResult extractBranches();
  LogicalResult extractFromBranch(IfOp ifOp, int16_t p);
  LogicalResult extractLoops();
  LogicalResult extractFromLoop(LoopOp loopOp, int16_t p);
  FailureOr<LoopOp> extractBreakValues(LoopOp loopOp, int16_t p);
  LogicalResult
  scanLoopBody(Operation *op, Block &body, int16_t p,
               SmallVectorImpl<SmallVector<Operation *>> &segments,
               SmallVectorImpl<Divergence> &divergences);
  LogicalResult rejectUnimplemented();
  LogicalResult sortRegion(Region &region);

  bool hasLaterContent(Operation *root, int16_t p);
  SmallVector<Operation *> collectLaterContent(Block &block, int16_t p);
  Operation *findExitTerminator(Region &region, int16_t p);
  void collectExitTerminators(Region &region, int16_t p,
                              SmallVectorImpl<Operation *> &exits);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp,
                                               ArrayRef<Operation *> movedOps);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp,
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
// Branch Extraction
//===----------------------------------------------------------------------===//
//
// # Extract Later-Phase Content Out of Earlier-Phase Branches
//
// A `uir.if` at phase p whose arms hold later-phase content, or whose result
// selects between distinct later-phase values, cannot wrap wholesale into
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
// when its providing arm yields select distinct SSA values. An identity-
// provided result (all arms yield the same value) or a never-produced result
// is not extracted; the residual if still delivers it divergently, so the
// rejection below reports it. Extraction thus lifts exactly the
// choice-ification case and leaves the rest rejected.

/// Whether any op transitively inside `root` sits at a real phase later than
/// p.
bool PhaseSorter::hasLaterContent(Operation *root, int16_t p) {
  auto result = root->walk([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    if (auto ph = realPhaseOf(op); ph && *ph > p)
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
PhaseSorter::collectCrossingValues(Operation *regionOp,
                                   ArrayRef<Operation *> movedOps) {
  return collectCrossingValues(regionOp, movedOps, movedOps);
}

/// Collect the values defined inside `regionOp` that the `userOps` use and
/// that are not themselves produced by any of the `movedOps`: these cross the
/// phase boundary and become record fields / choice payloads. `movedOps` lists
/// everything that leaves the region; with partial activations it is more than
/// the `userOps` whose uses are being collected here (a segment's uses are
/// collected while all segments and exit paths count as moved).
llvm::SetVector<Value>
PhaseSorter::collectCrossingValues(Operation *regionOp,
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

/// Clone the defining tree of a floating pure value at the builder's
/// insertion point.
static Value cloneFloatingTree(Value value, OpBuilder &builder,
                               IRMapping &mapping) {
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;
  auto *defOp = value.getDefiningOp();
  assert(defOp && "floating value must have a defining op");
  for (auto operand : defOp->getOperands())
    cloneFloatingTree(operand, builder, mapping);
  builder.clone(*defOp, mapping);
  return mapping.lookup(value);
}

/// Clone floating trees that moved content still references from inside the
/// (former) region of `regionOp`. A floating value defined inside an arm is
/// not visible from the receiving pick once the arm wraps into its own phase,
/// so it is duplicated in front of its user; any other remaining use of a
/// region-interior value is a causality violation the analysis rejects.
void PhaseSorter::cloneFloatingUses(Operation *regionOp, Block &rootBlock) {
  rootBlock.walk([&](Operation *user) {
    for (auto &use : user->getOpOperands()) {
      auto *defOp = use.get().getDefiningOp();
      if (!defOp || !regionOp->isProperAncestor(defOp))
        continue;
      assert(!isRealPhase(analysis.opPhases.at(defOp)) &&
             hir::isEffectivelyPure(defOp) &&
             "moved op uses non-floating, non-crossing value from region");
      OpBuilder builder(user);
      IRMapping mapping;
      auto clone = cloneFloatingTree(use.get(), builder, mapping);
      clone.getDefiningOp()->walk([&](Operation *cloned) {
        setPhase(cloned, PhaseAnalysis::kFloatPhase);
      });
      use.set(clone);
    }
  });
}

/// Whether the arm yields providing result `idx` of `ifOp` select between
/// distinct SSA values. Only distinct selections are choice-ification
/// obligations; a result forwarded identically by all arms is delivered at
/// the value's own phase and is not extracted here. An arm terminated by an
/// exit (return/break/continue) rather than a yield does not provide the
/// result; such an if is not choice-ified but left for the early-exit
/// rejection, so a single yielding provider never counts as a selection.
static bool selectsDistinctValues(IfOp ifOp, unsigned idx) {
  SmallVector<Value, 2> provided;
  for (auto *region : {&ifOp.getThenRegion(), &ifOp.getElseRegion()}) {
    if (region->empty())
      continue;
    auto yieldOp = dyn_cast<YieldOp>(region->front().getTerminator());
    if (yieldOp && idx < yieldOp.getValues().size())
      provided.push_back(yieldOp.getValues()[idx]);
  }
  return provided.size() > 1 &&
         !llvm::all_of(provided, [&](Value v) { return v == provided[0]; });
}

/// Extract later-phase content out of every phase-p `uir.if` whose interior
/// or results demand it, innermost first so a nested if becomes ordinary
/// later-phase content of its enclosing branch.
///
/// An if is extracted only when every one of its results is deliverable at
/// its own phase after extraction: results already at phase p stay, and
/// results at a later phase are removed and reconstructed by the pick — but
/// only if they choice-ify (distinct arm values). A result diverging from the
/// op's phase in any other way (identity-provided earlier or later, or
/// never-produced) is left in place for the divergence rejection; such an if
/// is not extracted.
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
          selectsDistinctValues(ifOp, result.getResultNumber())) {
        hasChoiceResult = true;
        continue; // Choice-ification obligation.
      }
      extractable = false; // Identity-provided or never-produced divergence.
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
    info.fields = collectCrossingValues(op, info.moved);

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
// Only single-boundary loop-record extraction is handled here (later-phase
// loop results are choice-ified by break-value extraction first): returns
// under loops (function-level exit chains), rejoining-arm content and nested
// divergence inside exit arms, and content more than one phase later (which
// needs the replicate re-extracted at the next boundary) are rejected below
// and by `rejectUnimplemented`.

/// Decompose a loop body's top level into content segments separated by
/// exit-carrying divergence points. `segments` holds the later-phase ops
/// between consecutive divergences in program order (one more entry than
/// `divergences`); each divergence records its exit terminators and the
/// later-phase content on their paths. Exit paths that run no later-phase
/// content get an empty chain link, so every activation pushes exactly one
/// record and the hit list length always equals the activation count.
LogicalResult
PhaseSorter::scanLoopBody(Operation *op, Block &body, int16_t p,
                          SmallVectorImpl<SmallVector<Operation *>> &segments,
                          SmallVectorImpl<Divergence> &divergences) {
  segments.emplace_back();
  for (auto &blockOp : body) {
    if (&blockOp == body.getTerminator())
      break;
    if (auto ph = realPhaseOf(&blockOp); ph && *ph > p) {
      segments.back().push_back(&blockOp);
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

    // Results whose phase was never demanded have no consumers and need no
    // reconstruction; only demanded later-phase results are unimplemented.
    for (auto result : blockOp.getResults()) {
      auto it = analysis.actualPhase.find(result);
      if (it != analysis.actualPhase.end() && isRealPhase(it->second) &&
          it->second > p)
        return blockOp.emitOpError()
               << "carries a loop exit and has later-phase results; exit-"
                  "carrying branches with demanded later-phase results are not "
                  "implemented in phase sorting yet";
    }

    Divergence divergence;
    if (!hasLaterContent(&blockOp, p)) {
      // No later-phase content anywhere in the divergence: every exit path
      // contributes an empty chain link built at its terminator, no matter
      // how deeply the terminator is nested.
      for (auto *term : exitTerms) {
        ExitPath exit;
        exit.terminator = term;
        divergence.exits.push_back(std::move(exit));
      }
    } else {
      // Later-phase content inside the divergence must be attributed to the
      // paths that run it; only the flat form (a branch whose exit arms are
      // straight-line up to their terminator) is implemented.
      auto ifOp = dyn_cast<IfOp>(&blockOp);
      if (!ifOp)
        return blockOp.emitOpError()
               << "carries a loop exit but is not a branch; cannot build "
                  "partial activation records";
      for (auto &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        auto &armBlock = region.front();
        auto *term = armBlock.getTerminator();
        auto termPhase = realPhaseOf(term);
        bool isExitTerm =
            termPhase && *termPhase <= p && isa<BreakOp, ContinueOp>(term);
        if (isExitTerm) {
          ExitPath exit;
          exit.terminator = term;
          exit.content = collectLaterContent(armBlock, p);
          // The arm must be straight-line up to its exit: nested divergence
          // (further exits or later-phase content inside arm-interior CF) is
          // not implemented.
          for (auto &armOp : armBlock) {
            if (&armOp == term || llvm::is_contained(exit.content, &armOp))
              continue;
            bool nestedExit = false;
            for (auto &inner : armOp.getRegions())
              nestedExit |= findExitTerminator(inner, p) != nullptr;
            if (nestedExit || hasLaterContent(&armOp, p))
              return armOp.emitOpError()
                     << "diverges inside an exit-carrying arm; nested "
                        "divergence is not implemented in phase sorting yet";
          }
          divergence.exits.push_back(std::move(exit));
        } else {
          // Rejoining arm: its content would need a sub-choice covering the
          // arm-local divergence, which is not implemented.
          if (findExitTerminator(region, p))
            return blockOp.emitOpError()
                   << "has a rejoining arm containing a nested exit; nested "
                      "divergence is not implemented in phase sorting yet";
          bool armHasContent = !collectLaterContent(armBlock, p).empty() ||
                               llvm::any_of(armBlock, [&](Operation &armOp) {
                                 return hasLaterContent(&armOp, p);
                               });
          if (armHasContent)
            return blockOp.emitOpError()
                   << "has later-phase content on a rejoining arm of an "
                      "exit-carrying branch; not implemented in phase sorting "
                      "yet";
        }
      }
    }

    divergences.push_back(std::move(divergence));
    segments.emplace_back();
  }
  return success();
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

    info.fields = collectCrossingValues(op, info.slice, info.slice);
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

  Block *body = &loopOp.getBody().front();
  SmallVector<SmallVector<Operation *>> segments;
  SmallVector<Divergence> divergences;
  if (failed(scanLoopBody(op, *body, p, segments, divergences)))
    return failure();
  size_t numDivs = divergences.size();

  // Collect the crossing values per segment and per exit path. Everything that
  // leaves the region counts as moved for the crossing analysis. A segment's
  // fields are deduplicated against all enclosing segments (their
  // receiving-block arguments dominate the nested levels); an exit path's
  // fields against the segments enclosing its pick.
  SmallVector<Operation *> allMoved;
  for (auto &segment : segments)
    allMoved.append(segment);
  for (auto &divergence : divergences)
    for (auto &exit : divergence.exits)
      allMoved.append(exit.content);

  SmallVector<llvm::SetVector<Value>> segFields(segments.size());
  llvm::SetVector<Value> prefixFields;
  for (auto [k, segment] : llvm::enumerate(segments)) {
    segFields[k] = collectCrossingValues(op, segment, allMoved);
    segFields[k].set_subtract(prefixFields);
    prefixFields.set_union(segFields[k]);
    if (k < numDivs) {
      for (auto &exit : divergences[k].exits) {
        exit.fields = collectCrossingValues(op, exit.content, allMoved);
        exit.fields.set_subtract(prefixFields);
      }
    }
  }

  // Hit list before the op.
  OpBuilder builder(op);
  auto list = hir::OpaqueListCreateOp::create(builder, loc, anyTy);
  setPhase(list, p);

  // Build one record push per exit path, in place at its terminator. The
  // record is the shared prefix segment plus, for each divergence the path
  // fell past, a `continued` link carrying that segment's samples, ending in
  // the path's own exit tag. The `continued` tag of each divergence is one
  // past its exit tags.
  auto buildRecordPush = [&](Operation *beforeTerm, int exitLevel, uint32_t tag,
                             ArrayRef<Value> payload) {
    OpBuilder b(beforeTerm);
    Value chain;
    if (exitLevel >= 0) {
      auto choice = hir::ChoiceOp::create(b, loc, anyTy, tag, payload);
      setPhase(choice, p);
      chain = choice.getResult();
      for (int i = exitLevel - 1; i >= 0; --i) {
        SmallVector<Value> link(segFields[i + 1].getArrayRef());
        link.push_back(chain);
        auto continued = hir::ChoiceOp::create(
            b, loc, anyTy, static_cast<uint32_t>(divergences[i].exits.size()),
            link);
        setPhase(continued, p);
        chain = continued.getResult();
      }
    }
    SmallVector<Value> recordFields(segFields[0].getArrayRef());
    if (chain)
      recordFields.push_back(chain);
    auto record = hir::OpaquePackOp::create(b, loc, anyTy, recordFields);
    setPhase(record, p);
    auto push = hir::OpaqueListPushOp::create(b, loc, list.getResult(),
                                              record.getResult());
    setPhase(push, p);
  };

  for (auto [k, divergence] : llvm::enumerate(divergences))
    for (auto [tag, exit] : llvm::enumerate(divergence.exits))
      buildRecordPush(exit.terminator, k, tag, exit.fields.getArrayRef());

  // A break/continue body terminator ends a full activation just like a
  // replicate body's yield; the record with the full chain is pushed there.
  auto *bodyTerm = body->getTerminator();
  if (isa<BreakOp, ContinueOp>(bodyTerm)) {
    if (numDivs == 0)
      buildRecordPush(bodyTerm, -1, 0, {});
    else
      buildRecordPush(bodyTerm, numDivs - 1,
                      divergences[numDivs - 1].exits.size(),
                      segFields[numDivs].getArrayRef());
  } else if (isa<ReturnOp>(bodyTerm)) {
    return bodyTerm->emitOpError()
           << "terminates a loop body with later-phase content; "
              "function-level exit chains are not implemented in phase "
              "sorting yet";
  } else if (!isa<UnreachableOp>(bodyTerm)) {
    return bodyTerm->emitOpError()
           << "unsupported loop body terminator for record extraction";
  }

  // Receiving replicate after the op, mirroring the record shape: one level
  // per segment, connected by picks over the chain links. Level k holds
  // segment k's content; its pick's exit alternatives hold the arm content,
  // and the `continued` alternative is level k+1.
  builder.setInsertionPointAfter(op);
  auto replicate = hir::ReplicateOp::create(builder, loc, list.getResult());
  analysis.opPhases[replicate] = p + 1;

  SmallVector<std::pair<Block *, const llvm::SetVector<Value> *>> remapBlocks;
  Block *levelBlock = &replicate.getBody().emplaceBlock();
  for (auto [k, segment] : llvm::enumerate(segments)) {
    bool hasChain = k < numDivs;
    for (unsigned i = 0; i < segFields[k].size(); ++i)
      levelBlock->addArgument(anyTy, loc);
    Value chainArg;
    if (hasChain) {
      chainArg = levelBlock->addArgument(anyTy, loc);
      analysis.actualPhase[chainArg] = p;
    }
    remapBlocks.push_back({levelBlock, &segFields[k]});
    for (auto *moved : segment)
      moved->moveBefore(levelBlock, levelBlock->end());

    Block *nextLevel = nullptr;
    if (hasChain) {
      auto &divergence = divergences[k];
      OpBuilder levelBuilder(levelBlock, levelBlock->end());
      auto pick = hir::PickOp::create(levelBuilder, loc, TypeRange{}, chainArg,
                                      divergence.exits.size() + 1);
      analysis.opPhases[pick] = p + 1;
      for (auto [tag, exit] : llvm::enumerate(divergence.exits)) {
        auto &altBlock = pick.getAlternatives()[tag].emplaceBlock();
        for (unsigned i = 0; i < exit.fields.size(); ++i)
          altBlock.addArgument(anyTy, loc);
        remapBlocks.push_back({&altBlock, &exit.fields});
        for (auto *moved : exit.content)
          moved->moveBefore(&altBlock, altBlock.end());
        OpBuilder altBuilder(&altBlock, altBlock.end());
        auto altYield = hir::YieldOp::create(altBuilder, loc, ValueRange{});
        analysis.opPhases[altYield] = p + 1;
      }
      nextLevel =
          &pick.getAlternatives()[divergence.exits.size()].emplaceBlock();
    }
    OpBuilder levelBuilder(levelBlock, levelBlock->end());
    auto levelYield = hir::YieldOp::create(levelBuilder, loc, ValueRange{});
    analysis.opPhases[levelYield] = p + 1;
    levelBlock = nextLevel;
  }

  // Remap fields outermost-first so that deduplicated inner uses resolve to
  // the enclosing level's arguments, then clone floating trees once over the
  // whole structure.
  for (auto [block, fields] : remapBlocks)
    remapFields(*block, *fields);
  cloneFloatingUses(op, replicate.getBody().front());

  return success();
}

/// Extract later-phase content out of every phase-p `uir.loop` whose interior
/// demands it, innermost first so a nested loop becomes ordinary later-phase
/// content of its enclosing loop's records.
LogicalResult PhaseSorter::extractLoops() {
  SmallVector<LoopOp> loops;
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](LoopOp op) {
    if (realPhaseOf(op))
      loops.push_back(op);
  });

  for (auto loopOp : loops) {
    int16_t p = *realPhaseOf(loopOp);
    if (!hasLaterContent(loopOp, p))
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
///   transparent forwarding of an identity-provided value or a never-produced
///   result. These need result forwarding and unreachable-consumer handling.
///   Choice-ification obligations (a control-flow op selecting distinct
///   later-phase values) are already lifted by branch extraction above, so
///   the residual divergence here is only identity-provided or never-produced.
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
        if (valueIt->second != opPhase) {
          op->emitOpError()
              << "delivers result #" << index << " at phase "
              << phaseName(valueIt->second) << " instead of its own phase "
              << phaseName(opPhase)
              << "; result forwarding (identity-provided or never-produced "
                 "control-flow results) is not implemented in phase sorting "
                 "yet";
          return WalkResult::interrupt();
        }
      }
    }

    // Reject anchored content nested inside differently-phased content.
    // Floating (dead) content may sit inside anchored content, but never the
    // other way around: demand or anchoring on nested content always anchors
    // its container.
    if (auto *parent = op->getParentOp(); parent != funcOp) {
      auto parentIt = analysis.opPhases.find(parent);
      assert(parentIt != analysis.opPhases.end() && "op phase not computed");
      if (isRealPhase(parentIt->second)) {
        if (isRealPhase(opPhase) && opPhase != parentIt->second) {
          op->emitOpError()
              << "at phase " << opPhase << " is nested inside phase "
              << parentIt->second
              << " content; hoisting and extraction across phase "
                 "boundaries are not implemented in phase sorting yet";
          return WalkResult::interrupt();
        }
      } else {
        assert(!isRealPhase(opPhase) &&
               "anchored content inside floating content");
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
