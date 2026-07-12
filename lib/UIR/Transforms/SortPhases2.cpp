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
// is computed.
//
// The pass currently covers straight-line content. Inputs that need the
// remaining sorting machinery are rejected with explicit diagnostics
// instead of being sorted incorrectly: call decomposition into
// `uir.phase_call`, early function exits (function-level exit chains),
// results delivered at a phase different from their op's own phase
// (identity-provided, choice-ified, and never-produced control-flow
// results, which need result forwarding and unreachable-consumer
// handling), hoisting/extraction of content nested under
// differently-phased control flow, and backward uses from floating content
// nested under earlier-phase control flow.
//
//===----------------------------------------------------------------------===//

#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
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
/// Sorts the content of a single `uir.func` into per-phase `uir.phase` ops
/// and maintains the phase map for every op it erases or creates.
struct PhaseSorter {
  PhaseSorter(FuncOp funcOp, PhaseAnalysis &analysis)
      : funcOp(funcOp), analysis(analysis) {}

  LogicalResult run();

private:
  FuncOp funcOp;
  PhaseAnalysis &analysis;

  LogicalResult rejectSortedVocabulary();
  LogicalResult checkCoverage();
  LogicalResult dissolveExprsAndPins();
  LogicalResult rejectUnimplemented();
  LogicalResult sortRegion(Region &region);

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

/// Reject inputs that need sorting machinery beyond wrapping and forward
/// crossings, with explicit diagnostics instead of sorting them incorrectly:
///
/// - `uir.call` decomposition into `uir.phase_call` entries.
/// - Early function exits (a `uir.return` anywhere but as the body region's
///   terminator, including returns still riding inside undissolved expr
///   carriers), which need function-level exit chains.
/// - Ops delivering a result at a phase different from their own phase: a
///   transparent forwarding of an identity-provided value, a
///   choice-ification obligation (a control-flow op selecting distinct
///   later-phase values), or a never-produced result. These need result
///   forwarding and unreachable-consumer handling.
/// - Anchored content whose phase differs from its parent op's phase, which
///   needs hoisting or extraction across the phase boundary.
///
/// Together these establish that every op and every value inside anchored
/// content is at that content's phase, which is what makes the wrap and
/// crossing rules below exact.
LogicalResult PhaseSorter::rejectUnimplemented() {
  auto *bodyTerminator = funcOp.getBody().front().getTerminator();
  auto result = funcOp->walk([&](Operation *op) {
    if (op == funcOp)
      return WalkResult::advance();
    if (isa<CallOp>(op)) {
      op->emitOpError() << "decomposition into 'uir.phase_call' entries is "
                           "not implemented in phase sorting yet";
      return WalkResult::interrupt();
    }
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
              << "; result forwarding (identity-provided, choice-ified, or "
                 "never-produced control-flow results) is not implemented in "
                 "phase sorting yet";
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
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      PhaseAnalysis analysis(funcOp);
      if (failed(analysis.run()))
        return signalPassFailure();
      if (failed(PhaseSorter(funcOp, analysis).run()))
        return signalPassFailure();
    }
  }
};
} // namespace
