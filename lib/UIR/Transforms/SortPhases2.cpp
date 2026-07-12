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
//   the latest such phase region, after its operands. Such ops are dead
//   (any live consumer would have anchored them through demand), so their
//   placement is only a dominance concern.
//
// - **Return**: the body's `uir.return` dissolves. The value and type
//   operand of result i move onto the `uir.phase_return` of the phase op
//   for the function's declared phase of result i, creating the phase op
//   if it does not exist: a phase return carrying function results is
//   content, so a result-only phase still gets its op, while contentless
//   phases get none. The body terminator becomes `uir.unreachable`. A
//   `uir.signature` terminator stays in place.
//
// Before sorting, `uir.expr` bodies are inlined and `uir.pin` outputs are
// replaced by their inputs: the grouping ops exist to communicate phase
// constraints to PhaseAnalysis and only stand in the way once the phase map
// is computed.
//
// The pass currently covers straight-line content whose values stay within
// their own phase. Inputs that need the remaining sorting machinery are
// rejected with explicit diagnostics instead of being sorted incorrectly:
// cross-phase value crossings (which will become phase-op results), call
// decomposition into `uir.phase_call`, early function exits (function-level
// exit chains), and hoisting/extraction of content nested under
// differently-phased control flow.
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
  LogicalResult checkRegion(Region &region);

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

/// Reject inputs that need sorting machinery beyond the straight-line wrap:
/// call decomposition into `uir.phase_call` and function-level exit chains
/// (a `uir.return` anywhere but as the body region's terminator, including
/// returns still riding inside undissolved expr carriers).
LogicalResult PhaseSorter::rejectUnimplemented() {
  auto *bodyTerminator = funcOp.getBody().front().getTerminator();
  auto result = funcOp->walk([&](Operation *op) {
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
  auto returnOp = dyn_cast<ReturnOp>(terminator);
  if (returnOp) {
    auto resultPhases = funcOp.getResultPhases();
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

  // Create the phase ops in ascending phase order before the terminator and
  // move each phase's content into them, preserving program order. Remaining
  // top-level ops all precede the phase ops, so direct uses of them from
  // inside the phase regions dominate.
  OpBuilder builder(terminator);
  for (auto &[phase, content] : phaseContent) {
    auto loc =
        content.empty() ? terminator->getLoc() : content.front()->getLoc();
    auto phaseOp = PhaseOp::create(builder, loc, TypeRange{}, phase);
    setPhase(phaseOp, phase);
    auto &phaseBlock = phaseOp.getBody().emplaceBlock();
    for (auto *op : content)
      op->moveBefore(&phaseBlock, phaseBlock.end());

    auto returnBuilder = OpBuilder::atBlockEnd(&phaseBlock);
    auto resultsIt = phaseResults.find(phase);
    auto phaseReturnOp =
        resultsIt != phaseResults.end()
            ? PhaseReturnOp::create(
                  returnBuilder, terminator->getLoc(), resultsIt->second.values,
                  resultsIt->second.typeOfValues, ValueRange{})
            : PhaseReturnOp::create(returnBuilder, loc, ValueRange{},
                                    ValueRange{}, ValueRange{});
    setPhase(phaseReturnOp, phase);
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
// Check the Sorted Region
//===----------------------------------------------------------------------===//

/// Check the sorted region against the parts of the phase-op contract this
/// pass cannot yet establish, and reject violating inputs loudly:
///
/// - Every op inside a `uir.phase` at a real phase must be at that phase;
///   a mismatch means the input needs hoisting or extraction of content
///   nested under differently-phased control flow.
/// - Every use must stay within the defining value's phase region; uses
///   from another phase op or from the region top level are cross-phase
///   crossings, which will become phase-op results.
///
/// Uses of region-top-level values (function arguments and floating ops)
/// from inside any phase are legal sorted form and pass the check.
LogicalResult PhaseSorter::checkRegion(Region &region) {
  auto &block = region.front();

  // Report a use of a phase-interior value from outside that phase.
  auto emitCrossingError = [&](Operation *user, Value value,
                               PhaseOp defPhaseOp) {
    auto diag = user->emitOpError()
                << "uses a value defined in phase " << defPhaseOp.getPhase()
                << "; cross-phase value crossings are not implemented in "
                   "phase sorting yet";
    diag.attachNote(value.getLoc()) << "value defined here";
    return diag;
  };

  // The phase op enclosing an operand's definition at this region's top
  // level, or null for block arguments and top-level floating ops.
  auto definingPhaseOp = [&](Value value) -> PhaseOp {
    auto *defOp = value.getDefiningOp();
    if (!defOp)
      return {};
    auto *ancestor = block.findAncestorOpInBlock(*defOp);
    assert(ancestor && "operand defined outside the function region");
    return dyn_cast<PhaseOp>(ancestor);
  };

  for (auto &topOp : block) {
    if (auto phaseOp = dyn_cast<PhaseOp>(&topOp)) {
      auto result = phaseOp.getBody().walk([&](Operation *op) {
        auto it = analysis.opPhases.find(op);
        assert(it != analysis.opPhases.end() && "op phase not computed");
        if (isRealPhase(it->second) && it->second != phaseOp.getPhase()) {
          op->emitOpError()
              << "at phase " << it->second << " is nested inside phase "
              << phaseOp.getPhase()
              << " content; hoisting and extraction across phase "
                 "boundaries are not implemented in phase sorting yet";
          return WalkResult::interrupt();
        }
        for (auto operand : op->getOperands()) {
          auto defPhaseOp = definingPhaseOp(operand);
          if (defPhaseOp && defPhaseOp != phaseOp) {
            emitCrossingError(op, operand, defPhaseOp);
            return WalkResult::interrupt();
          }
        }
        return WalkResult::advance();
      });
      if (result.wasInterrupted())
        return failure();
      continue;
    }

    // Top-level ops (floating ops and the region terminator) must not
    // reach into any phase region.
    auto result = topOp.walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        if (auto defPhaseOp = definingPhaseOp(operand)) {
          emitCrossingError(op, operand, defPhaseOp);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return failure();
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
  if (failed(checkRegion(funcOp.getSignature())))
    return failure();
  if (failed(checkRegion(funcOp.getBody())))
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
