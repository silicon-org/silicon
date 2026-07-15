//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "phase-analysis2"
#define INDENT llvm::indent(depth * 2)

/// Check whether an operand is a "type operand" — a value that describes a
/// type rather than carrying data. Type operands must be available one phase
/// before the value they describe (the "-1 rule").
///
/// Ops declare their type operands via `TypeOperandsOpInterface`; all
/// operands of ops without the interface are plain value operands.
static bool isTypeOperand(OpOperand &operand) {
  if (auto iface = dyn_cast<hir::TypeOperandsOpInterface>(operand.getOwner()))
    return iface.isTypeOperand(operand);
  return false;
}

/// Walk the function IR and populate the terminator mappings. Uses manual
/// recursion with a `currentLoop` context to associate break/continue ops with
/// their enclosing loop without walking the parent chain at each use.
void PhaseAnalysis::collectTerminators() {
  std::function<void(Block &, LoopOp)> walkBlock = [&](Block &block,
                                                       LoopOp currentLoop) {
    for (auto &op : block) {
      if (isa<ReturnOp, SignatureOp>(&op)) {
        funcTerminators.push_back(&op);
        continue;
      }
      if (auto breakOp = dyn_cast<BreakOp>(&op)) {
        assert(currentLoop && "break outside of loop");
        loopBreaks[currentLoop].push_back(breakOp);
        breakToLoop[breakOp] = currentLoop;
        continue;
      }
      if (auto continueOp = dyn_cast<ContinueOp>(&op)) {
        assert(currentLoop && "continue outside of loop");
        continueToLoop[continueOp] = currentLoop;
        continue;
      }
      // Regions the analysis does not model are opaque: terminators inside
      // them belong to that op's own construct, not to this function's
      // loops or interface.
      if (!isModeledRegionOp(&op))
        continue;
      auto loopOp = dyn_cast<LoopOp>(&op);
      for (auto &region : op.getRegions())
        for (auto &innerBlock : region)
          walkBlock(innerBlock, loopOp ? loopOp : currentLoop);
    }
  };

  for (auto &block : funcOp.getSignature())
    walkBlock(block, /*currentLoop=*/nullptr);
  for (auto &block : funcOp.getBody())
    walkBlock(block, /*currentLoop=*/nullptr);
}

/// Run the phase analysis. The algorithm has six steps:
/// 1. Pre-walk: collect terminators.
/// 2. Constrain blocks containing function terminators to phase ≤ 0.
/// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
/// 4. Demand-driven constraints: push from declared arg/result phases.
///    Then default undemanded floating exprs with anchored content to their
///    block phase.
/// 5. Final recompute: reschedule visited pure ops to their earliest phase,
///    derive phases for unvisited pure ops from their operands, and refresh
///    or fill in region result phases from their providers.
/// 6. Coverage stamp: anything still unresolved (operand-free dead pure
///    trees, dead terminators, dead block arguments) gets the float
///    sentinel, so afterwards only contents of unmodeled regions lack
///    entries.
LogicalResult PhaseAnalysis::run() {
  LLVM_DEBUG(dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                    << "\n");

  opPhases.insert({funcOp, 0});

  // Pre-populate actualPhase for body and signature block arguments.
  auto bodyArgs = funcOp.getBody().getArguments();
  auto sigArgs = funcOp.getSignature().getArguments();
  for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
    int16_t argPhase = static_cast<int16_t>(phase);
    actualPhase[bodyArgs[idx]] = argPhase;
    actualPhase[sigArgs[idx]] = argPhase;
    LLVM_DEBUG(if (argPhase != 0) dbgs()
               << "  arg " << idx << " has phase " << argPhase << "\n");
  }

  // Step 1: Pre-walk — collect terminators.
  collectTerminators();

  // Step 2: Constrain blocks containing function terminators.
  LLVM_DEBUG(dbgs() << "  step 2: constrain function terminator blocks\n");
  for (auto *term : funcTerminators)
    if (failed(constrainBlock(*term->getBlock(), 0)))
      return failure();

  // Step 3: Structural setup — process blocks.
  LLVM_DEBUG(dbgs() << "  step 3: structural setup\n");
  if (failed(processBlock(funcOp.getSignature().front(), 0)))
    return failure();
  if (failed(processBlock(funcOp.getBody().front(), 0)))
    return failure();

  // Step 3b: Result phase floor for function returns. A result provided by
  // more than one return op is a selection at the body phase (0), regardless
  // of the values' identity, so a declared result phase < 0 is an error.
  {
    auto resultPhases = funcOp.getResultPhases();
    for (unsigned i = 0; i < resultPhases.size(); ++i) {
      int16_t rp = static_cast<int16_t>(resultPhases[i]);
      if (rp >= 0)
        continue;

      unsigned numProviders = 0;
      for (auto *term : funcTerminators)
        if (auto returnOp = dyn_cast<ReturnOp>(term))
          if (i < returnOp.getValues().size())
            ++numProviders;

      if (numProviders <= 1)
        continue;

      auto diag = emitError(funcOp.getLoc())
                  << "result " << i << " at phase " << rp
                  << " is selected by control flow at phase 0"
                  << "; result phase must be >= 0";
      for (auto *term : funcTerminators)
        if (auto returnOp = dyn_cast<ReturnOp>(term))
          diag.attachNote(returnOp.getLoc()) << "return value provided here";
      return failure();
    }
  }

  // Step 4: Demand-driven constraints — function interface.
  LLVM_DEBUG(dbgs() << "  step 4: demand-driven constraints\n");
  for (auto *term : funcTerminators) {
    if (auto sigOp = dyn_cast<SignatureOp>(term)) {
      auto argPhases = funcOp.getArgPhases();
      auto resultPhases = funcOp.getResultPhases();

      for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfArgs())) {
        int16_t latest = static_cast<int16_t>(argPhases[i]) - 1;
        if (failed(constrainValue(typeVal, latest))) {
          emitRemark(sigOp.getLoc())
              << "required by signature type of arg " << i;
          return failure();
        }
      }
      for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfResults())) {
        int16_t latest = static_cast<int16_t>(resultPhases[i]) - 1;
        if (failed(constrainValue(typeVal, latest))) {
          emitRemark(sigOp.getLoc())
              << "required by signature type of result " << i;
          return failure();
        }
      }
    } else if (auto returnOp = dyn_cast<ReturnOp>(term)) {
      auto resultPhases = funcOp.getResultPhases();

      for (auto [i, value] : llvm::enumerate(returnOp.getValues())) {
        int16_t latest = static_cast<int16_t>(resultPhases[i]);
        if (failed(constrainValue(value, latest))) {
          emitRemark(returnOp.getLoc())
              << "required by return value at phase " << latest;
          return failure();
        }
      }
      for (auto [i, typeVal] : llvm::enumerate(returnOp.getTypeOfValues())) {
        int16_t typeLatest = static_cast<int16_t>(resultPhases[i]) - 1;
        if (failed(constrainValue(typeVal, typeLatest))) {
          emitRemark(returnOp.getLoc()) << "required by return type operand";
          return failure();
        }
      }
    }
  }

  // Step 4b: Floating exprs that no demand ever reached have no phase. Pure
  // contents are dead code and are left for the step-6 float stamp, but
  // anchored contents (side effects, pins) must execute at a well-defined
  // phase regardless of whether any result is used: such exprs fall back to
  // their block phase, as if the expression statement stood alone. Walk in
  // pre-order so a nested floating expr sees its parent's (possibly just
  // defaulted) phase.
  {
    // Whether the expr transitively contains anchored (non-floating) content.
    // A break/continue whose target loop lies outside `exprOp` transfers
    // control out of the frame: like a side effect, it is observable and
    // must execute at a definite phase, so it anchors the expr. An exit
    // targeting a loop inside `exprOp` (or any other terminator) does not.
    auto hasAnchoredContent = [&](ExprOp exprOp) {
      auto anchorsExpr = [&](Operation *targetLoop) {
        return !exprOp->isProperAncestor(targetLoop);
      };
      auto inner = exprOp->walk([&](Operation *op) -> WalkResult {
        if (auto breakOp = dyn_cast<BreakOp>(op))
          return anchorsExpr(breakToLoop.at(breakOp)) ? WalkResult::interrupt()
                                                      : WalkResult::advance();
        if (auto continueOp = dyn_cast<ContinueOp>(op))
          return anchorsExpr(continueToLoop.at(continueOp))
                     ? WalkResult::interrupt()
                     : WalkResult::advance();
        if (op == exprOp || hir::isEffectivelyPure(op) ||
            op->hasTrait<OpTrait::ConstantLike>() ||
            op->hasTrait<OpTrait::IsTerminator>() || isModeledRegionOp(op))
          return WalkResult::advance();
        return WalkResult::interrupt();
      });
      return inner.wasInterrupted();
    };

    auto walkResult =
        funcOp->walk<WalkOrder::PreOrder>([&](ExprOp exprOp) -> WalkResult {
          if (exprOp.getPin() || opPhases.count(exprOp) ||
              !hasAnchoredContent(exprOp))
            return WalkResult::advance();

          // The fallback phase is the parent block's phase: 0 for the func
          // body, otherwise the parent op's phase (already resolved — the
          // pre-order walk defaults enclosing floating exprs first).
          auto *parentOp = exprOp->getParentOp();
          int16_t blockPhase =
              isa<FuncOp>(parentOp) ? 0 : opPhases.find(parentOp)->second;
          LLVM_DEBUG(dbgs() << "    defaulting floating expr to block phase "
                            << blockPhase << "\n");
          opPhases[exprOp] = blockPhase;

          for (auto typeVal : exprOp.getResultTypes()) {
            if (failed(constrainValue(typeVal, blockPhase - 1))) {
              emitRemark(exprOp.getLoc())
                  << "required by expr result type at phase " << blockPhase - 1;
              return WalkResult::interrupt();
            }
          }
          if (failed(processBlock(exprOp.getBody().front(), blockPhase)))
            return WalkResult::interrupt();
          return WalkResult::advance();
        });
    if (walkResult.wasInterrupted())
      return failure();
  }

  // Step 5: Final recompute and derivation. During the DFS, pure ops were
  // assigned their demanded phase (`latest`) and region result phases were
  // recorded from demand-time value phases; both can be tighter in the final
  // solution. Content no demand ever reached has no entry yet; since
  // undemanded means latest = +inf, no error is possible and phases derive
  // from operands alone: pure ops as earliest = max(operand phases), and
  // undemanded floating exprs like pure ops, from the bounds their anchored
  // contents impose (deriveFloatingExpr). Region results are refreshed or
  // filled in from their providing terminator values. Demand thus never
  // influences where a pure op, expr, or region result materializes, and
  // dead results choice-ify exactly like live ones. The walk covers exactly
  // the blocks the analysis models — the entry blocks of signature and body
  // and, recursively, of modeled region ops — in post-order, so providers
  // are final before their consumers: region contents before the region op
  // itself, block ops in order.
  LLVM_DEBUG(dbgs() << "  step 5: recompute pure ops and region results\n");
  recomputeBlock(funcOp.getSignature().front());
  recomputeBlock(funcOp.getBody().front());

  // Step 6: Coverage stamp. Everything the analysis models — the entry
  // blocks of the function's signature and body, and recursively the entry
  // blocks of the region ops it models — ends up with an entry in both
  // maps. Ops and values still unresolved here are dead code with no
  // real-phase operand bound (operand-free pure trees, dead terminators,
  // dead block arguments); they get the float sentinel. Contents of regions
  // the analysis does not model remain absent from the maps: absence of an
  // entry is the defined signal that an op or value is outside the
  // analysis' model.
  LLVM_DEBUG(dbgs() << "  step 6: stamp unresolved pure content\n");
  std::function<void(Block &)> stamp = [&](Block &block) {
    for (auto arg : block.getArguments())
      actualPhase.try_emplace(arg, kFloatPhase);
    for (auto &op : block) {
      opPhases.try_emplace(&op, kFloatPhase);
      if (isModeledRegionOp(&op))
        for (auto &region : op.getRegions())
          if (!region.empty())
            stamp(region.front());
      for (auto result : op.getResults())
        actualPhase.try_emplace(result, kFloatPhase);
    }
  };
  stamp(funcOp.getSignature().front());
  stamp(funcOp.getBody().front());

  return success();
}

//===----------------------------------------------------------------------===//
// Final Recompute and Derivation (step 5)
//===----------------------------------------------------------------------===//
//
// The recompute pass finalizes the phases the demand DFS left loose. Pure
// ops reschedule to earliest = max(operand phases); region results follow
// their providers. Content no demand ever reached derives the same way:
// undemanded pure ops adopt earliest directly, and undemanded floating
// exprs derive in two passes. The scan pass derives the expr's pure content
// (with the expr's own frame still unknown, so iteration arguments of loops
// inside contribute nothing yet) and collects the lower bounds the anchored
// content imposes on the expr's phase: a CF condition or loop init consumed
// at the frame phase forces the frame to be at least the operand's phase,
// and type operands force one more. The assign pass then anchors CF ops,
// terminators, pinned exprs, and loop iteration arguments at the derived
// phase and re-derives the pure content and region results — values built
// from iteration arguments can only rise to the frame phase they were
// pegged to, which every bound they participate in already accommodates, so
// no iteration is needed. With no real bound anywhere, the expr is an
// operand-free tree: it stays unresolved and floats in step 6.

// NOLINTBEGIN(misc-no-recursion)

void PhaseAnalysis::recomputeBlock(Block &block) {
  for (auto &op : block) {
    if (isModeledRegionOp(&op)) {
      auto exprOp = dyn_cast<ExprOp>(&op);
      if (exprOp && !exprOp.getPin() && !opPhases.count(&op)) {
        // Undemanded floating expr: derive it like a pure op; the
        // derivation recomputes its interior itself.
        deriveFloatingExpr(exprOp);
      } else {
        assert(opPhases.count(&op) &&
               "region op in a processed block must have a phase");
        for (auto &region : op.getRegions())
          if (!region.empty())
            recomputeBlock(region.front());
      }
      fillRegionResults(&op, /*overwrite=*/false);
      continue;
    }

    if (!hir::isEffectivelyPure(&op) || op.hasTrait<OpTrait::IsTerminator>() ||
        op.getNumRegions() > 0)
      continue;
    derivePure(&op, /*overwrite=*/false);
  }
}

void PhaseAnalysis::derivePure(Operation *op, bool overwrite) {
  int16_t earliest = INT16_MIN;
  if (isa<hir::TypeOfOp>(op)) {
    auto valIt = actualPhase.find(op->getOperand(0));
    if (valIt != actualPhase.end()) {
      earliest = valIt->second;
      if (earliest > kNeverPhase)
        earliest -= 1;
    }
  } else {
    for (auto &operand : op->getOpOperands()) {
      auto valIt = actualPhase.find(operand.get());
      if (valIt == actualPhase.end())
        continue;
      int16_t opActual = valIt->second;
      // The ±1 shifts only apply to real phases, not the sentinels.
      if (isTypeOperand(operand) && opActual > kNeverPhase)
        opActual += 1;
      earliest = std::max(earliest, opActual);
    }
  }

  auto phaseIt = opPhases.find(op);
  if (phaseIt == opPhases.end()) {
    // Unvisited: no demand imposes a latest bound; adopt earliest when some
    // operand contributes a real phase, otherwise stay unresolved (float in
    // step 6).
    if (earliest <= kNeverPhase)
      return;
  } else if (!overwrite) {
    // Visited: the DFS assigned the demanded phase; tighten to earliest.
    if (earliest >= phaseIt->second)
      return;
  } else if (earliest <= kNeverPhase) {
    // Overwrite mode never loses a real phase: operand entries only appear
    // or rise between the scan and assign passes.
    return;
  }

  LLVM_DEBUG({
    dbgs() << "    ";
    if (phaseIt != opPhases.end())
      dbgs() << phaseIt->second << " => " << earliest;
    else
      dbgs() << "derived " << earliest;
    dbgs() << ": ";
    op->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });
  opPhases[op] = earliest;
  for (auto r : op->getResults())
    actualPhase[r] = earliest;
}

void PhaseAnalysis::fillRegionResults(Operation *op, bool overwrite) {
  for (auto result : op->getResults()) {
    auto providers = resultProviders(op, result.getResultNumber());
    int16_t recomputed = kNeverPhase;
    if (!providers.empty()) {
      recomputed = kFloatPhase;
      for (auto &[value, term] : providers) {
        auto valIt = actualPhase.find(value);
        recomputed =
            std::max(recomputed,
                     valIt != actualPhase.end() ? valIt->second : kFloatPhase);
      }
      if (providers.size() > 1) {
        auto opIt = opPhases.find(op);
        recomputed = std::max(recomputed, opIt != opPhases.end() ? opIt->second
                                                                 : kFloatPhase);
      }
    }
    auto resultIt = actualPhase.find(result);
    if (resultIt == actualPhase.end()) {
      actualPhase[result] = recomputed;
      continue;
    }
    if (recomputed == resultIt->second)
      continue;
    assert((overwrite || recomputed < resultIt->second) &&
           "region result phases only tighten");
    LLVM_DEBUG({
      dbgs() << "    result " << result.getResultNumber() << ": "
             << resultIt->second << " => " << recomputed << ": ";
      op->print(dbgs(), OpPrintingFlags().skipRegions());
      dbgs() << "\n";
    });
    resultIt->second = recomputed;
  }
}

void PhaseAnalysis::scanFloatingContent(Block &block, int16_t shift,
                                        int16_t &bound) {
  // An anchored op consuming `value` at the frame phase (expr + relShift)
  // forces expr >= phase(value) - relShift; type operands are consumed one
  // phase earlier and force one more. Sentinel and unresolved operands
  // contribute no bound.
  auto addBound = [&](Value value, int16_t relShift, bool isType) {
    auto valIt = actualPhase.find(value);
    if (valIt == actualPhase.end() || valIt->second <= kNeverPhase)
      return;
    bound =
        std::max<int16_t>(bound, valIt->second + (isType ? 1 : 0) - relShift);
  };

  for (auto &op : block) {
    if (auto exprOp = dyn_cast<ExprOp>(&op)) {
      if (exprOp.getPin()) {
        int16_t inner = shift + exprOp.getPhaseShift();
        for (auto typeVal : exprOp.getResultTypes())
          addBound(typeVal, inner, /*isType=*/true);
        scanFloatingContent(exprOp.getBody().front(), inner, bound);
      } else {
        // A nested floating expr is an independent frame: it derives its own
        // phase from its own anchored interior right here, and its result
        // imposes no bound on this frame (a floating expr's yield constrains
        // nothing unless demanded). Deriving it during the scan makes nested
        // frames resolve regardless of whether this outer frame does.
        if (!opPhases.count(exprOp))
          deriveFloatingExpr(exprOp);
      }
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    if (auto ifOp = dyn_cast<IfOp>(&op)) {
      addBound(ifOp.getCondition(), shift, /*isType=*/false);
      for (auto typeVal : ifOp.getResultTypes())
        addBound(typeVal, shift, /*isType=*/true);
      scanFloatingContent(ifOp.getThenRegion().front(), shift, bound);
      if (!ifOp.getElseRegion().empty())
        scanFloatingContent(ifOp.getElseRegion().front(), shift, bound);
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    if (auto loopOp = dyn_cast<LoopOp>(&op)) {
      for (auto init : loopOp.getInits())
        addBound(init, shift, /*isType=*/false);
      for (auto typeVal : loopOp.getInitTypes())
        addBound(typeVal, shift, /*isType=*/true);
      for (auto typeVal : loopOp.getResultTypes())
        addBound(typeVal, shift, /*isType=*/true);
      scanFloatingContent(loopOp.getBody().front(), shift, bound);
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    // Break/continue exits. An exit whose target loop lies outside the frame
    // — control must cross a `uir.expr` boundary to reach it — is frame-locked
    // at that loop's (real) phase, since the exit's phase aliases the target
    // loop's phase. It therefore bounds the frame to the loop's phase (less
    // the pin shift to the exit's own frame). An in-frame exit aliases an
    // inner floating loop's phase and imposes nothing, except that in-frame
    // continue values feed the loop's iteration arguments at the frame phase
    // (their type operands one phase earlier); break values are delivered
    // through the loop's result and impose no frame bound.
    if (isa<BreakOp, ContinueOp>(&op)) {
      auto targetLoop = isa<BreakOp>(&op)
                            ? breakToLoop.at(cast<BreakOp>(&op))
                            : continueToLoop.at(cast<ContinueOp>(&op));
      bool outTargeting = false;
      for (Operation *p = op.getParentOp(); p != targetLoop.getOperation();
           p = p->getParentOp())
        if (isa<ExprOp>(p)) {
          outTargeting = true;
          break;
        }
      if (outTargeting) {
        auto loopIt = opPhases.find(targetLoop);
        if (loopIt != opPhases.end() && loopIt->second > kNeverPhase)
          bound = std::max<int16_t>(bound, loopIt->second - shift);
      } else if (auto continueOp = dyn_cast<ContinueOp>(&op)) {
        for (auto value : continueOp.getValues())
          addBound(value, shift, /*isType=*/false);
        for (auto typeVal : continueOp.getTypeOfValues())
          addBound(typeVal, shift, /*isType=*/true);
      }
      continue;
    }

    if (op.hasTrait<OpTrait::IsTerminator>())
      continue;

    // Pure region-bearing ops outside the model (e.g. hir.pick): their
    // interiors carry no entries and they derive no phase.
    if (op.getNumRegions() > 0) {
      assert(hir::isEffectivelyPure(&op) &&
             "effectful op inside an undemanded floating expr");
      continue;
    }

    assert(
        (hir::isEffectivelyPure(&op) || op.hasTrait<OpTrait::ConstantLike>()) &&
        "effectful op inside an undemanded floating expr; step 4b "
        "defaults exprs with anchored content");
    derivePure(&op, /*overwrite=*/true);

    // An internal crossing bounds the frame. When this op sits at a real
    // phase but consumes an operand at a strictly earlier real phase, the
    // earlier value must be captured and carried forward inside the frame.
    // A crossing exists only between materialized phases, so the frame
    // cannot float wholesale: it is bounded to this op's phase (less the pin
    // shift to the op's own frame). A frame whose interior content all sits
    // at a single phase (no lower operand anywhere) has no crossing and
    // still floats. Iteration arguments carry no phase during the scan, so
    // they contribute no spurious crossing.
    auto opIt = opPhases.find(&op);
    if (opIt != opPhases.end() && opIt->second > kNeverPhase)
      for (auto operand : op.getOperands()) {
        auto vIt = actualPhase.find(operand);
        if (vIt != actualPhase.end() && vIt->second > kNeverPhase &&
            vIt->second < opIt->second) {
          bound = std::max<int16_t>(bound, opIt->second - shift);
          break;
        }
      }
  }
}

void PhaseAnalysis::assignDerivedBlock(Block &block, int16_t blockPhase) {
  for (auto &op : block) {
    if (auto exprOp = dyn_cast<ExprOp>(&op)) {
      if (exprOp.getPin()) {
        int16_t exprPhase = blockPhase + exprOp.getPhaseShift();
        opPhases[&op] = exprPhase;
        assignDerivedBlock(exprOp.getBody().front(), exprPhase);
      }
      // A nested floating expr is an independent frame already derived in
      // the scan pass; it does not re-anchor to this frame's phase.
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    if (auto ifOp = dyn_cast<IfOp>(&op)) {
      opPhases[&op] = blockPhase;
      assignDerivedBlock(ifOp.getThenRegion().front(), blockPhase);
      if (!ifOp.getElseRegion().empty())
        assignDerivedBlock(ifOp.getElseRegion().front(), blockPhase);
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    if (auto loopOp = dyn_cast<LoopOp>(&op)) {
      opPhases[&op] = blockPhase;
      // Iteration arguments live at the loop body's phase.
      for (auto arg : loopOp.getBody().front().getArguments())
        actualPhase[arg] = blockPhase;
      assignDerivedBlock(loopOp.getBody().front(), blockPhase);
      fillRegionResults(&op, /*overwrite=*/true);
      continue;
    }

    if (op.hasTrait<OpTrait::IsTerminator>()) {
      opPhases[&op] = blockPhase;
      continue;
    }

    // Pure region-bearing ops outside the model stay unresolved.
    if (op.getNumRegions() > 0)
      continue;

    derivePure(&op, /*overwrite=*/true);
  }
}

void PhaseAnalysis::deriveFloatingExpr(ExprOp exprOp) {
  assert(!exprOp.getPin() && !opPhases.count(exprOp) &&
         "only undemanded floating exprs are derived");

  int16_t bound = kFloatPhase;
  scanFloatingContent(exprOp.getBody().front(), /*shift=*/0, bound);
  // The expr's result types are consumed one phase before the expr, like a
  // pure op's type operands.
  for (auto typeVal : exprOp.getResultTypes()) {
    auto valIt = actualPhase.find(typeVal);
    if (valIt != actualPhase.end() && valIt->second > kNeverPhase)
      bound = std::max<int16_t>(bound, valIt->second + 1);
  }

  if (bound <= kNeverPhase)
    return;

  LLVM_DEBUG({
    dbgs() << "    derived floating expr at " << bound << ": ";
    exprOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });
  opPhases[exprOp] = bound;
  assignDerivedBlock(exprOp.getBody().front(), bound);
}

// NOLINTEND(misc-no-recursion)

/// Collect the (value, terminator) pairs that provide a given result of a
/// region-bearing op.
///
/// A result is provided only by terminators that hand control to the op's
/// continuation carrying the value: a `uir.yield` for exprs and ifs, and a
/// `uir.break` of that loop for loops. Any other arm terminator diverts
/// control elsewhere — an if/expr arm that breaks, continues, returns, or is
/// unreachable never delivers a value to the op's result, so its operands (a
/// break's values belong to the enclosing loop's result, not this op's) are
/// not providers. Result-index positions a provider does not carry are
/// omitted.
SmallVector<std::pair<Value, Operation *>, 2>
PhaseAnalysis::resultProviders(Operation *regionOp, unsigned resultIdx) {
  SmallVector<std::pair<Value, Operation *>, 2> providers;
  auto addYield = [&](Operation *term) {
    auto yieldOp = dyn_cast<YieldOp>(term);
    if (!yieldOp)
      return;
    if (resultIdx < yieldOp.getValues().size())
      providers.push_back({yieldOp.getValues()[resultIdx], term});
  };

  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    addYield(exprOp.getBody().front().getTerminator());
  } else if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    addYield(ifOp.getThenRegion().front().getTerminator());
    if (!ifOp.getElseRegion().empty())
      addYield(ifOp.getElseRegion().front().getTerminator());
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end())
      for (auto breakOp : it->second)
        if (resultIdx < breakOp.getValues().size())
          providers.push_back({breakOp.getValues()[resultIdx], breakOp});
  } else {
    llvm_unreachable("unexpected region op in resultProviders");
  }
  return providers;
}

// NOLINTBEGIN(misc-no-recursion)

/// Consumer demands value at phase ≤ latest. Dispatches based on the value's
/// kind. Returns the resolved phase, or failure if the demand cannot be
/// satisfied (diagnostics already emitted).
FailureOr<int16_t> PhaseAnalysis::constrainValue(Value value, int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Block arguments have fixed phases.
  if (isa<BlockArgument>(value)) {
    int16_t phase = actualPhase.at(value);
    LLVM_DEBUG(dbgs() << INDENT << "constrainValue(blockArg, <=" << latest
                      << "): actual " << phase << "\n");
    if (phase > latest) {
      emitError(value.getLoc())
          << "value at phase " << phase
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return phase;
  }

  auto result = cast<OpResult>(value);
  auto *op = result.getOwner();
  unsigned resultIdx = result.getResultNumber();

  // ConstantLike ops float to any phase.
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    LLVM_DEBUG(dbgs() << INDENT << "constrainValue(constant, <=" << latest
                      << "): float\n");
    opPhases[op] = kFloatPhase;
    for (auto r : op->getResults())
      actualPhase[r] = kFloatPhase;
    return kFloatPhase;
  }

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainValue(<=" << latest << "): ";
    value.printAsOperand(dbgs(), OpPrintingFlags());
    dbgs() << "\n";
  });

  // hir.type_of: result is at p(operand) - 1.
  if (auto typeOfOp = dyn_cast<hir::TypeOfOp>(op)) {
    auto inputPhase = constrainValue(typeOfOp.getInput(), latest + 1);
    if (failed(inputPhase))
      return failure();
    // Sentinel phases (float/never) pass through unshifted.
    int16_t resolved =
        (*inputPhase <= kNeverPhase) ? *inputPhase : *inputPhase - 1;
    LLVM_DEBUG(dbgs() << INDENT << "  => phase " << resolved << "\n");
    opPhases[op] = resolved;
    actualPhase[typeOfOp.getResult()] = resolved;
    return resolved;
  }

  // Region ops: push per-result constraint through the yield.
  if (isModeledRegionOp(op)) {
    if (failed(constrainRegionResult(op, resultIdx, latest)))
      return failure();
    int16_t actual = actualPhase.at(value);
    if (actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // Call results: anchored at blockPhase + resultOffset. constrainBlock ensures
  // the enclosing block is processed (triggers processBlock for floating
  // exprs).
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto resultPhases = callOp.getResultPhases();
    int16_t resultOffset = static_cast<int16_t>(resultPhases[resultIdx]);
    int16_t demandedBlockPhase = latest - resultOffset;

    if (failed(constrainBlock(*op->getBlock(), demandedBlockPhase))) {
      emitRemark(op->getLoc())
          << "required by call result " << resultIdx << " at phase " << latest;
      return failure();
    }

    int16_t actual = actualPhase.at(value);
    if (actual > latest) {
      emitError(op->getLoc())
          << "call result " << resultIdx << " at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // PinOp: anchored at blockPhase + offset. constrainBlock ensures the
  // enclosing block is processed.
  if (auto pinOp = dyn_cast<PinOp>(op)) {
    int16_t offset = pinOp.getPhaseOffset();
    int16_t demandedBlockPhase = latest - offset;

    if (failed(constrainBlock(*op->getBlock(), demandedBlockPhase))) {
      emitRemark(op->getLoc()) << "required by pin at phase " << latest;
      return failure();
    }

    int16_t actual = actualPhase.at(value);
    if (actual > latest) {
      emitError(op->getLoc())
          << "pinned value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // Pure ops: push constraints to operands, assign `latest` as the phase.
  if (hir::isEffectivelyPure(op)) {
    for (auto &operand : op->getOpOperands()) {
      int16_t opLatest = isTypeOperand(operand) ? (latest - 1) : latest;
      if (failed(constrainValue(operand.get(), opLatest))) {
        emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
        return failure();
      }
    }

    auto it = opPhases.find(op);
    if (it == opPhases.end() || it->second > latest) {
      LLVM_DEBUG(dbgs() << INDENT << "  => phase " << latest << "\n");
      opPhases[op] = latest;
      for (auto r : op->getResults())
        actualPhase[r] = latest;
    }

    return actualPhase.at(value);
  }

  // Other side-effecting ops: anchored at block phase. constrainBlock ensures
  // the enclosing block is processed.
  if (failed(constrainBlock(*op->getBlock(), latest))) {
    emitRemark(op->getLoc())
        << "required by side-effecting op at phase " << latest;
    return failure();
  }

  int16_t actual = actualPhase.at(value);
  if (actual > latest) {
    emitError(op->getLoc())
        << "value at phase " << actual
        << " cannot satisfy requirement for phase " << latest;
    return failure();
  }
  return actual;
}

/// Someone needs this block at phase ≤ demanded. Dispatches on the parent op:
/// - Floating expr: tighten block phase, re-process.
/// - Pinned expr at offset N: propagate to parent block at demanded - N.
/// - Anchored CF (if/loop): propagate to parent block.
/// - Function body/signature: fixed at 0.
///
/// Returns the resolved block phase.
FailureOr<int16_t> PhaseAnalysis::constrainBlock(Block &block,
                                                 int16_t demandedPhase) {
  SaveAndRestore guard(depth, depth + 2);
  auto *parentOp = block.getParentOp();
  assert(parentOp && "block has no parent op");

  if (isa<FuncOp>(parentOp)) {
    LLVM_DEBUG(dbgs() << INDENT << "constrainBlock(<=" << demandedPhase
                      << "): func body, fixed at 0\n");
    return int16_t(0);
  }

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainBlock(<=" << demandedPhase << "): ";
    parentOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  if (auto exprOp = dyn_cast<ExprOp>(parentOp)) {
    if (exprOp.getPin()) {
      auto parentPhase = constrainBlock(*parentOp->getBlock(),
                                        demandedPhase - exprOp.getPhaseShift());
      if (failed(parentPhase))
        return failure();
      return int16_t(*parentPhase + exprOp.getPhaseShift());
    }

    // Floating expr: tighten block phase if needed, then re-process.
    auto it = opPhases.find(parentOp);
    if (it != opPhases.end() && it->second <= demandedPhase)
      return it->second;

    LLVM_DEBUG(dbgs() << INDENT << "  floating expr => phase " << demandedPhase
                      << "\n");
    opPhases[parentOp] = demandedPhase;
    for (auto typeVal : exprOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, demandedPhase - 1))) {
        emitRemark(exprOp.getLoc())
            << "required by expr result type at phase " << demandedPhase - 1;
        return failure();
      }
    }
    if (failed(processBlock(exprOp.getBody().front(), demandedPhase)))
      return failure();
    return demandedPhase;
  }

  // Anchored CF (if/loop): propagate to parent block.
  if (isa<IfOp, LoopOp>(parentOp))
    return constrainBlock(*parentOp->getBlock(), demandedPhase);

  llvm_unreachable("unexpected parent op in constrainBlock");
}

/// Push demand through yield/break to a specific result of a region-bearing op.
/// Yields and breaks forward the demand to the corresponding operand.
///
/// For merging CF ops (IfOp, LoopOp), the result phase floor applies: a
/// result provided by more than one terminator is a selection the CF op
/// performs at its own phase — regardless of the values' identity — so the
/// result cannot be delivered earlier. A demand below the op phase is an
/// error, and the recorded result phase is the max over the contributing
/// values, floored at the op phase.
///
/// A recorded result phase *later* than the op phase is a choice-ification
/// obligation: SortPhases replaces the value by a choice crossing the phase
/// boundary and reconstructs the selection with `hir.pick` in the receiving
/// phase. SortPhases derives these obligations directly from the phase map.
LogicalResult PhaseAnalysis::constrainRegionResult(Operation *regionOp,
                                                   unsigned resultIdx,
                                                   int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Skip if this result already satisfies the demand.
  auto result = regionOp->getResult(resultIdx);
  auto resultIt = actualPhase.find(result);
  if (resultIt != actualPhase.end() && resultIt->second <= latest)
    return success();

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainRegionResult(result " << resultIdx
           << ", <=" << latest << "): ";
    regionOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  // Constrain the value and type operand a terminator provides for this
  // result. Returns the value's resolved phase, or nullopt if the terminator
  // does not carry this result (e.g. uir.return, uir.unreachable).
  auto constrainTermOperand =
      [&](Operation *term) -> FailureOr<std::optional<int16_t>> {
    // Only the terminator kind that hands control to this op's continuation
    // provides its results: a yield for exprs/ifs, a break for loops. An
    // if/expr arm terminated otherwise (a break diverting to an enclosing
    // loop, a continue, a return) delivers no value to this op's result.
    auto getOperands = [&](Operation *t) -> std::pair<ValueRange, ValueRange> {
      if (isa<ExprOp, IfOp>(regionOp)) {
        if (auto yieldOp = dyn_cast<YieldOp>(t))
          return {yieldOp.getValues(), yieldOp.getTypeOfValues()};
      } else if (isa<LoopOp>(regionOp)) {
        if (auto breakOp = dyn_cast<BreakOp>(t))
          return {breakOp.getValues(), breakOp.getTypeOfValues()};
      }
      return {{}, {}};
    };

    auto [values, typeOfValues] = getOperands(term);

    std::optional<int16_t> valuePhase;
    if (resultIdx < values.size()) {
      auto phase = constrainValue(values[resultIdx], latest);
      if (failed(phase)) {
        emitRemark(term->getLoc()) << "required by yield operand";
        return failure();
      }
      valuePhase = *phase;
    }

    if (resultIdx < typeOfValues.size()) {
      if (failed(constrainValue(typeOfValues[resultIdx], latest - 1))) {
        emitRemark(term->getLoc()) << "required by yield type operand";
        return failure();
      }
    }

    return valuePhase;
  };

  // Exprs are transparent single-terminator conduits.
  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    auto phase = constrainTermOperand(exprOp.getBody().front().getTerminator());
    if (failed(phase))
      return failure();
    assert(*phase && "expr yield must carry all results");
    actualPhase[result] = **phase;
    return success();
  }

  // Collect the terminators feeding this result of a merging CF op.
  SmallVector<Operation *, 2> terms;
  if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    terms.push_back(ifOp.getThenRegion().front().getTerminator());
    if (!ifOp.getElseRegion().empty())
      terms.push_back(ifOp.getElseRegion().front().getTerminator());
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end())
      for (auto breakOp : it->second)
        terms.push_back(breakOp);
  } else {
    llvm_unreachable("unexpected region op in constrainRegionResult");
  }

  // Gather the values provided for this result. A result provided by more
  // than one terminator is a selection, regardless of the values' identity,
  // and the result phase floor applies.
  auto termEntries = resultProviders(regionOp, resultIdx);

  // A result that no terminator provides (a loop without breaks, an if whose
  // branches all return) never materializes: every consumer is unreachable at
  // runtime, like Rust's `!`. It gets the bottom phase, which satisfies any
  // demand.
  if (termEntries.empty()) {
    LLVM_DEBUG(dbgs() << INDENT << "  => never (no providers)\n");
    actualPhase[result] = kNeverPhase;
    return success();
  }

  bool selects = termEntries.size() > 1;

  // Make sure the CF op has a resolved phase. A result can be demanded before
  // the op's block was processed when the op is nested in a floating expr;
  // constraining the enclosing block at the demanded phase resolves the
  // floating exprs on the way and processes their contents.
  if (opPhases.find(regionOp) == opPhases.end()) {
    if (failed(constrainBlock(*regionOp->getBlock(), latest)))
      return failure();
    assert(opPhases.find(regionOp) != opPhases.end() &&
           "region op must be processed after constraining its block");
  }
  int16_t opPhase = opPhases.find(regionOp)->second;

  // Result phase floor: a selection cannot be delivered earlier than the
  // phase at which the CF op performs it.
  if (selects && latest < opPhase) {
    auto diag = emitError(regionOp->getLoc())
                << "result at phase " << latest
                << " is selected by control flow at phase " << opPhase
                << "; result phase must be >= " << opPhase;
    for (auto &[val, term] : termEntries)
      diag.attachNote(term->getLoc()) << "value provided here";
    return failure();
  }

  // Constrain all contributing terminator operands and record the result
  // phase: the max over the contributing values, floored at the op phase for
  // selections.
  std::optional<int16_t> resultPhase;
  for (auto *term : terms) {
    auto phase = constrainTermOperand(term);
    if (failed(phase))
      return failure();
    if (*phase)
      resultPhase = std::max(resultPhase.value_or(INT16_MIN), **phase);
  }
  if (selects)
    resultPhase = std::max(resultPhase.value_or(INT16_MIN), opPhase);
  if (resultPhase)
    actualPhase[result] = *resultPhase;
  return success();
}

/// Push blockPhase down to all ops in a block (including the terminator).
LogicalResult PhaseAnalysis::processBlock(Block &block, int16_t blockPhase) {
  SaveAndRestore guard(depth, depth + 2);
  LLVM_DEBUG(dbgs() << INDENT << "processBlock(phase " << blockPhase << ")\n");

  for (auto &op : block)
    if (failed(processOp(&op, blockPhase)))
      return failure();

  return success();
}

/// Assign phase to a single op and push constraints to operands/regions.
/// Skips floating exprs, pure ops, and constants (demand-driven only).
/// Validates terminator phase constraints.
LogicalResult PhaseAnalysis::processOp(Operation *op, int16_t blockPhase) {
  // Floating uir.expr: demand-driven only.
  if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && !exprOp.getPin())
    return success();

  // Pure ops and constants impose no latest bound on themselves, regardless
  // of use count: their latest comes from demands (+inf when undemanded),
  // and their final phase is derived in the step-5 recompute. Terminators
  // are control flow and stay anchored even when memory-effect free.
  if (!isa<ExprOp>(op) && op->getNumRegions() == 0 &&
      !op->hasTrait<OpTrait::IsTerminator>() &&
      (hir::isEffectivelyPure(op) || op->hasTrait<OpTrait::ConstantLike>()))
    return success();

  SaveAndRestore guard(depth, depth + 2);

  // Pinned expr: phase = blockPhase + offset.
  if (auto exprOp = dyn_cast<ExprOp>(op)) {
    assert(exprOp.getPin() && "floating expr should have been skipped");
    int16_t exprPhase = blockPhase + exprOp.getPhaseShift();

    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= exprPhase)
      return success();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(pinnedExpr, phase " << exprPhase
                      << ")\n");
    opPhases[op] = exprPhase;

    for (auto typeVal : exprOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, exprPhase - 1))) {
        emitRemark(exprOp.getLoc())
            << "required by expr result type at phase " << exprPhase - 1;
        return failure();
      }
    }

    return processBlock(exprOp.getBody().front(), exprPhase);
  }

  // PinOp: phase = blockPhase + offset.
  if (auto pinOp = dyn_cast<PinOp>(op)) {
    int16_t pinPhase = blockPhase + pinOp.getPhaseOffset();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(pin, phase " << pinPhase
                      << ")\n");
    opPhases[op] = pinPhase;
    for (auto r : op->getResults())
      actualPhase[r] = pinPhase;

    for (auto input : pinOp.getInputs()) {
      if (failed(constrainValue(input, pinPhase))) {
        emitRemark(pinOp.getLoc()) << "required by pin at phase " << pinPhase;
        return failure();
      }
    }
    return success();
  }

  // IfOp: anchored at blockPhase.
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return success();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(if, phase " << blockPhase
                      << ")\n");
    opPhases[op] = blockPhase;

    if (failed(constrainValue(ifOp.getCondition(), blockPhase))) {
      emitRemark(ifOp.getLoc())
          << "required by if condition at phase " << blockPhase;
      return failure();
    }
    for (auto typeVal : ifOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1))) {
        emitRemark(ifOp.getLoc())
            << "required by if result type at phase " << blockPhase - 1;
        return failure();
      }
    }

    if (failed(processBlock(ifOp.getThenRegion().front(), blockPhase)))
      return failure();
    if (!ifOp.getElseRegion().empty())
      if (failed(processBlock(ifOp.getElseRegion().front(), blockPhase)))
        return failure();
    return success();
  }

  // LoopOp: anchored at blockPhase. Loop-carried iteration arguments live at
  // the loop body's phase; their initial values are constrained to be
  // available there, and the type operands one phase earlier.
  if (auto loopOp = dyn_cast<LoopOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return success();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(loop, phase " << blockPhase
                      << ")\n");
    opPhases[op] = blockPhase;

    for (auto typeVal : loopOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1))) {
        emitRemark(loopOp.getLoc())
            << "required by loop result type at phase " << blockPhase - 1;
        return failure();
      }
    }

    // Iteration arguments are available at the loop body's phase.
    for (auto arg : loopOp.getBody().front().getArguments())
      actualPhase[arg] = blockPhase;

    for (auto init : loopOp.getInits()) {
      if (failed(constrainValue(init, blockPhase))) {
        emitRemark(loopOp.getLoc())
            << "required by loop init value at phase " << blockPhase;
        return failure();
      }
    }
    for (auto typeVal : loopOp.getInitTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1))) {
        emitRemark(loopOp.getLoc())
            << "required by loop init type at phase " << blockPhase - 1;
        return failure();
      }
    }

    return processBlock(loopOp.getBody().front(), blockPhase);
  }

  // CallOp: anchored at blockPhase.
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return success();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(call, phase " << blockPhase
                      << ")\n");
    opPhases[op] = blockPhase;

    auto resultPhases = callOp.getResultPhases();
    for (auto [i, r] : llvm::enumerate(callOp.getResults()))
      actualPhase[r] = blockPhase + static_cast<int16_t>(resultPhases[i]);

    auto argPhases = callOp.getArgPhases();
    for (auto [i, arg] : llvm::enumerate(callOp.getArguments())) {
      int16_t argLatest = blockPhase + static_cast<int16_t>(argPhases[i]);
      if (failed(constrainValue(arg, argLatest))) {
        emitRemark(callOp.getLoc())
            << "required by call argument " << i << " at phase " << argLatest;
        return failure();
      }
    }

    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfArgs())) {
      int16_t typeLatest = blockPhase + static_cast<int16_t>(argPhases[i]) - 1;
      if (failed(constrainValue(typeVal, typeLatest))) {
        emitRemark(callOp.getLoc())
            << "required by type of call argument " << i;
        return failure();
      }
    }

    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfResults())) {
      int16_t typeLatest =
          blockPhase + static_cast<int16_t>(resultPhases[i]) - 1;
      if (failed(constrainValue(typeVal, typeLatest))) {
        emitRemark(callOp.getLoc()) << "required by type of call result " << i;
        return failure();
      }
    }
    return success();
  }

  // Terminators: assign block phase and validate phase-shift constraints.
  if (isa<YieldOp, UnreachableOp>(op)) {
    opPhases[op] = blockPhase;
    return success();
  }

  if (isa<ReturnOp>(op)) {
    opPhases[op] = blockPhase;
    if (blockPhase != 0) {
      emitError(op->getLoc())
          << "return from a phase-shifted block is not allowed";
      return failure();
    }
    return success();
  }

  if (isa<SignatureOp>(op)) {
    opPhases[op] = blockPhase;
    // The verifier pins uir.signature to the func's signature region, which is
    // always processed at phase 0.
    assert(blockPhase == 0 && "signature region must be at phase 0");
    return success();
  }

  if (auto breakOp = dyn_cast<BreakOp>(op)) {
    opPhases[op] = blockPhase;
    int16_t loopPhase = getPhase(breakToLoop.at(breakOp));
    if (blockPhase != loopPhase) {
      emitError(breakOp.getLoc())
          << "break from a phase-shifted block is not allowed";
      return failure();
    }
    return success();
  }

  if (auto continueOp = dyn_cast<ContinueOp>(op)) {
    opPhases[op] = blockPhase;
    int16_t loopPhase = getPhase(continueToLoop.at(continueOp));
    if (blockPhase != loopPhase) {
      emitError(continueOp.getLoc())
          << "continue from a phase-shifted block is not allowed";
      return failure();
    }
    // The carried values flow into the loop's iteration arguments, which live
    // at the loop's phase; their type operands one phase earlier.
    for (auto value : continueOp.getValues()) {
      if (failed(constrainValue(value, blockPhase))) {
        emitRemark(continueOp.getLoc())
            << "required by continue value at phase " << blockPhase;
        return failure();
      }
    }
    for (auto typeVal : continueOp.getTypeOfValues()) {
      if (failed(constrainValue(typeVal, blockPhase - 1))) {
        emitRemark(continueOp.getLoc()) << "required by continue type operand";
        return failure();
      }
    }
    return success();
  }

  // Remaining side-effecting ops: anchored at the block phase.
  auto it = opPhases.find(op);
  if (it != opPhases.end() && it->second <= blockPhase)
    return success();

  LLVM_DEBUG({
    dbgs() << INDENT << "processOp(generic, phase " << blockPhase << "): ";
    op->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });
  opPhases[op] = blockPhase;
  for (auto r : op->getResults())
    actualPhase[r] = blockPhase;

  for (auto &operand : op->getOpOperands()) {
    int16_t opLatest = isTypeOperand(operand) ? (blockPhase - 1) : blockPhase;
    if (failed(constrainValue(operand.get(), opLatest))) {
      emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
      return failure();
    }
  }
  return success();
}

// NOLINTEND(misc-no-recursion)

//===----------------------------------------------------------------------===//
// Accessors
//===----------------------------------------------------------------------===//

int16_t PhaseAnalysis::getPhase(Operation *op) const {
  auto it = opPhases.find(op);
  assert(it != opPhases.end() && "op phase not computed");
  return it->second;
}

int16_t PhaseAnalysis::getValuePhase(Value value) const {
  auto it = actualPhase.find(value);
  assert(it != actualPhase.end() && "value phase not computed");
  return it->second;
}

/// Annotate every op in the function with its resolved phase and the phases
/// of its operands and results. After the coverage stamp, only ops inside
/// regions the analysis does not model lack an opPhases entry; those still
/// get operand/result annotations.
void PhaseAnalysis::annotate() const {
  FuncOp func = funcOp;
  auto *ctx = func.getContext();

  auto phaseAttr = [&](int16_t phase) -> StringAttr {
    if (phase == kFloatPhase)
      return StringAttr::get(ctx, "float");
    if (phase == kNeverPhase)
      return StringAttr::get(ctx, "never");
    return StringAttr::get(ctx, std::to_string(phase));
  };

  auto valuePhases = [&](auto values) -> ArrayAttr {
    SmallVector<Attribute> attrs;
    for (Value v : values) {
      auto it = actualPhase.find(v);
      if (it != actualPhase.end())
        attrs.push_back(phaseAttr(it->second));
      else
        attrs.push_back(StringAttr::get(ctx, "?"));
    }
    return ArrayAttr::get(ctx, attrs);
  };

  for (auto &region : func->getRegions()) {
    region.walk([&](Operation *op) {
      auto it = opPhases.find(op);
      if (it != opPhases.end())
        op->setAttr("pa.phase", phaseAttr(it->second));
      if (op->getNumOperands() > 0)
        op->setAttr("pa.operands", valuePhases(op->getOperands()));
      if (op->getNumResults() > 0)
        op->setAttr("pa.results", valuePhases(op->getResults()));
    });
  }
}
