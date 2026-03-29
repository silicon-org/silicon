//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "phase-analysis2"

//===----------------------------------------------------------------------===//
// Type Operand Classification
//
// Determines which operands of an op are "type operands" — values that describe
// types rather than carry data. Type operands must be available one phase
// before the value they describe (the "-1 rule").
//
// For each op, we compute a bitmask where bit i indicates that operand i is a
// type operand. processOp and constrainValue push `phase - 1` to type operands
// instead of `phase`.
//
// NOTE: When adding a new HIR op with type operands, this function must be
// updated. See also the corresponding op definitions in HIR/Ops.td.
//===----------------------------------------------------------------------===//

/// Check whether an operand is a "type operand" — a value that describes a
/// type rather than carrying data. Uses tablegen-generated Mutable accessors
/// to identify type operands without hardcoding indices.
///
/// NOTE: When adding a new HIR op with type operands, add a case here.
/// See also the corresponding op definitions in HIR/Ops.td.
static bool isTypeOperand(OpOperand &operand) {
  return TypeSwitch<Operation *, bool>(operand.getOwner())
      // HIR binary ops and others with $resultType.
      .Case<hir::AddOp, hir::SubOp, hir::MulOp, hir::DivOp, hir::ModOp,
            hir::AndOp, hir::OrOp, hir::XorOp, hir::ShlOp, hir::ShrOp,
            hir::EqOp, hir::NeqOp, hir::LtOp, hir::GtOp, hir::GeqOp,
            hir::LeqOp>(
          [&](auto op) { return &operand == &op.getResultTypeMutable(); })
      // Ops with $typeOperand.
      .Case<hir::ConstantIntOp, hir::CoerceTypeOp>(
          [&](auto op) { return &operand == &op.getTypeOperandMutable(); })
      // hir.store: $valueType.
      .Case<hir::StoreOp>(
          [&](auto op) { return &operand == &op.getValueTypeMutable(); })
      .Default([](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

template <typename OpTy>
OpTy PhaseAnalysis::findEnclosing(Operation *from) {
  for (auto *op = from->getParentOp(); op; op = op->getParentOp()) {
    if (auto target = dyn_cast<OpTy>(op))
      return target;
  }
  llvm_unreachable("enclosing op not found");
}

//===----------------------------------------------------------------------===//
// Entry Point
//
// The algorithm has four steps:
// 1. Pre-walk: collect terminators.
// 2. Constrain blocks containing function-interface terminators to phase ≤ 0.
// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
// 4. Demand-driven constraints: push from declared arg/result phases.
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::run() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                          << "\n");

  // Seed the function at phase 0.
  opPhases.insert({funcOp, 0});

  // Pre-populate actualPhase for body and signature block arguments. Both
  // regions have matching block args corresponding to the function arguments.
  auto bodyArgs = funcOp.getBody().getArguments();
  auto sigArgs = funcOp.getSignature().getArguments();
  for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
    int16_t argPhase = static_cast<int16_t>(phase);
    actualPhase[bodyArgs[idx]] = argPhase;
    actualPhase[sigArgs[idx]] = argPhase;
    LLVM_DEBUG(if (argPhase != 0) llvm::dbgs()
               << "- Arg " << idx << " has phase " << argPhase << "\n");
  }

  // Step 1: Pre-walk — collect terminators.
  funcOp.walk([&](Operation *op) {
    if (auto breakOp = dyn_cast<BreakOp>(op)) {
      auto loopOp = findEnclosing<LoopOp>(breakOp);
      loopBreaks[loopOp].push_back(breakOp);
    } else if (isa<ReturnOp, SignatureOp>(op)) {
      funcInterfaceTerminators.push_back(op);
    }
  });

  // Step 2: Constrain blocks containing function-interface terminators.
  // This ensures any floating expr containing a return or signature gets
  // constrained to phase ≤ 0.
  for (auto *term : funcInterfaceTerminators)
    constrainBlock(*term->getBlock(), 0);

  // Step 3: Structural setup — process blocks.
  processBlock(funcOp.getSignature().front(), 0);
  processBlock(funcOp.getBody().front(), 0);

  // Step 4: Demand-driven constraints — function interface.
  for (auto *term : funcInterfaceTerminators) {
    if (auto sigOp = dyn_cast<SignatureOp>(term)) {
      auto enclosingFunc = findEnclosing<FuncOp>(sigOp);
      auto argPhases = enclosingFunc.getArgPhases();
      auto resultPhases = enclosingFunc.getResultPhases();

      for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfArgs())) {
        int16_t latest = static_cast<int16_t>(argPhases[i]) - 1;
        if (failed(constrainValue(typeVal, latest)))
          emitRemark(sigOp.getLoc())
              << "required by signature type of arg " << i;
      }
      for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfResults())) {
        int16_t latest = static_cast<int16_t>(resultPhases[i]) - 1;
        if (failed(constrainValue(typeVal, latest)))
          emitRemark(sigOp.getLoc())
              << "required by signature type of result " << i;
      }
    } else if (auto returnOp = dyn_cast<ReturnOp>(term)) {
      auto enclosingFunc = findEnclosing<FuncOp>(returnOp);
      auto resultPhases = enclosingFunc.getResultPhases();

      for (auto [i, value] : llvm::enumerate(returnOp.getValues())) {
        int16_t latest = static_cast<int16_t>(resultPhases[i]);
        if (failed(constrainValue(value, latest)))
          emitRemark(returnOp.getLoc())
              << "required by return value at phase " << latest;
      }
      for (auto [i, typeVal] : llvm::enumerate(returnOp.getTypeOfValues())) {
        int16_t typeLatest = static_cast<int16_t>(resultPhases[i]) - 1;
        if (failed(constrainValue(typeVal, typeLatest)))
          emitRemark(returnOp.getLoc()) << "required by return type operand";
      }
    }
  }

  return anyErrors ? failure() : success();
}

//===----------------------------------------------------------------------===//
// constrainValue — Consumer Demand on a Value
//
// Consumer demands value at phase ≤ latest. Dispatches based on the value's
// kind:
// - Block arg: check actualPhase ≤ latest.
// - Constant-like result: resolve op, always satisfies.
// - Pure op result: push to operands, schedule at earliest.
// - Region op result (expr/if/loop): constrainRegionResult.
// - Call result: push to call op with result offset.
// - Other side-effecting result: constrainBlock(parentBlock, latest).
//===----------------------------------------------------------------------===//

FailureOr<int16_t> PhaseAnalysis::constrainValue(Value value, int16_t latest) {
  // Block arguments have fixed phases.
  if (isa<BlockArgument>(value)) {
    auto it = actualPhase.find(value);
    assert(it != actualPhase.end() && "block arg phase not seeded");
    int16_t phase = it->second;
    if (phase != INT16_MIN && phase > latest) {
      emitError(value.getLoc())
          << "value at phase " << phase
          << " cannot satisfy requirement for phase " << latest;
      anyErrors = true;
      return failure();
    }
    return phase;
  }

  auto *op = value.getDefiningOp();
  assert(op && "expected defining op for non-block-arg value");

  unsigned resultIdx = cast<OpResult>(value).getResultNumber();

  // ConstantLike ops float to any phase.
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    opPhases[op] = INT16_MIN;
    for (auto result : op->getResults())
      actualPhase[result] = INT16_MIN;
    return INT16_MIN;
  }

  // hir.type_of: result is at p(operand) - 1. Push latest + 1 to input.
  if (isa<hir::TypeOfOp>(op)) {
    assert(op->getNumOperands() == 1);
    (void)constrainValue(op->getOperand(0), latest + 1);
    int16_t inputPhase = actualPhase.lookup(op->getOperand(0));
    int16_t resolved = (inputPhase == INT16_MIN) ? INT16_MIN : inputPhase - 1;
    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << resolved << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = resolved;
    for (auto result : op->getResults())
      actualPhase[result] = resolved;
    return resolved;
  }

  // Region ops: push per-result constraint through the yield.
  if (isa<ExprOp, IfOp, LoopOp>(op)) {
    constrainRegionResult(op, resultIdx, latest);
    int16_t actual = actualPhase.lookup(value);
    if (actual != INT16_MIN && actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      anyErrors = true;
      return failure();
    }
    return actual;
  }

  // Call results: translate value-level constraint to op-level by subtracting
  // the result's phase offset. Calls are tightened directly (not through
  // constrainBlock), since they can be tightened independently.
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto resultPhases = callOp.getResultPhases();
    int16_t resultOffset = static_cast<int16_t>(resultPhases[resultIdx]);
    int16_t demandedCallPhase = latest - resultOffset;

    // Also constrain the parent block (for floating exprs).
    constrainBlock(*op->getBlock(), demandedCallPhase);

    // Tighten the call directly.
    auto callIt = opPhases.find(op);
    if (callIt == opPhases.end() || callIt->second > demandedCallPhase)
      processOp(op, demandedCallPhase);

    int16_t actual = actualPhase.lookup(value);
    if (actual != INT16_MIN && actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      anyErrors = true;
      return failure();
    }
    return actual;
  }

  // Pure ops (effectively pure): push to operands, then schedule at earliest.
  if (hir::isEffectivelyPure(op)) {
    // Push constraints to operands.
    for (auto &operand : op->getOpOperands()) {
      int16_t opLatest = isTypeOperand(operand) ? (latest - 1) : latest;
      if (failed(constrainValue(operand.get(), opLatest)))
        emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
    }

    // Post-order: schedule at max(operand actuals), accounting for the type
    // operand -1 rule.
    int16_t earliest = INT16_MIN;
    for (auto &operand : op->getOpOperands()) {
      int16_t opActual = actualPhase.lookup(operand.get());
      if (isTypeOperand(operand) && opActual != INT16_MIN)
        opActual += 1;
      earliest = std::max(earliest, opActual);
    }

    // Only update if this is a tighter (earlier) phase or first visit.
    auto it = opPhases.find(op);
    if (it == opPhases.end() || it->second > earliest) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Phase " << earliest << " for: ";
        op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
        llvm::dbgs() << "\n";
      });
      opPhases[op] = earliest;
      for (auto result : op->getResults())
        actualPhase[result] = earliest;
    }

    int16_t actual = actualPhase.lookup(value);
    if (actual != INT16_MIN && actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      anyErrors = true;
      return failure();
    }
    return actual;
  }

  // Other side-effecting ops: constrain the parent block and tighten the op.
  constrainBlock(*op->getBlock(), latest);
  auto seIt = opPhases.find(op);
  if (seIt == opPhases.end() || seIt->second > latest)
    processOp(op, latest);

  int16_t actual = actualPhase.lookup(value);
  if (actual != INT16_MIN && actual > latest) {
    emitError(op->getLoc())
        << "value at phase " << actual
        << " cannot satisfy requirement for phase " << latest;
    anyErrors = true;
    return failure();
  }
  return actual;
}

//===----------------------------------------------------------------------===//
// constrainBlock — Block Phase Demand
//
// Someone needs this block at phase ≤ demanded. Dispatches on the parent op:
// - Floating expr: tighten block phase, re-process.
// - Pinned expr at offset N: constrainBlock(parentBlock, demanded - N).
// - Anchored CF (if/loop): constrainBlock(parentBlock, demanded).
// - Function body/signature: phase is 0 (no action needed here).
//===----------------------------------------------------------------------===//

void PhaseAnalysis::constrainBlock(Block &block, int16_t demandedPhase) {
  auto *parentOp = block.getParentOp();
  if (!parentOp)
    return;

  // Function body/signature: phase is fixed at 0. Nothing to constrain.
  if (isa<FuncOp>(parentOp))
    return;

  if (auto exprOp = dyn_cast<ExprOp>(parentOp)) {
    if (exprOp.getPin()) {
      // Pinned expr at offset N: need p(parentBlock) + N ≤ demanded,
      // so constrainBlock(parentBlock, demanded - N).
      constrainBlock(*parentOp->getBlock(),
                     demandedPhase - exprOp.getPhaseShift());
    } else {
      // Floating expr: tighten block phase.
      auto it = opPhases.find(parentOp);
      if (it != opPhases.end() && it->second <= demandedPhase)
        return; // already tight enough
      // Will be processed when processBlock reaches it, or re-processed
      // if tightening.
    }
    return;
  }

  // Anchored CF (if/loop): demand propagates to parent block.
  if (isa<IfOp, LoopOp>(parentOp)) {
    constrainBlock(*parentOp->getBlock(), demandedPhase);
    return;
  }
}

//===----------------------------------------------------------------------===//
// constrainRegionResult — Per-Result Constraint Through Yield/Break
//
// Pushes a phase constraint onto a specific result of a region-bearing op.
// The constraint propagates through the yield (for expr/if) or break ops
// (for loop) to the corresponding operand. Enables sparse, per-result
// constraints.
//===----------------------------------------------------------------------===//

void PhaseAnalysis::constrainRegionResult(Operation *regionOp,
                                          unsigned resultIdx, int16_t latest) {
  // Initialize result constraints if needed.
  auto &constraints = resultConstraints[regionOp];
  if (constraints.empty())
    constraints.assign(regionOp->getNumResults(), kUnconstrained);
  assert(resultIdx < constraints.size());

  // Only tighten: skip if already at a tighter-or-equal constraint.
  if (constraints[resultIdx] != kUnconstrained &&
      constraints[resultIdx] <= latest)
    return;
  constraints[resultIdx] = latest;

  // Push constraint through yields/breaks.
  auto pushToYield = [&](Region &region) {
    auto *term = region.front().getTerminator();
    if (auto yieldOp = dyn_cast<YieldOp>(term)) {
      auto values = yieldOp.getValues();
      if (resultIdx < values.size()) {
        Value value = values[resultIdx];

        // For floating exprs, ensure the block has been processed at the
        // demanded phase. On first visit, this sets the phase and runs
        // processBlock. On re-visit with a tighter demand, it re-processes.
        bool isFloating =
            isa<ExprOp>(regionOp) && !cast<ExprOp>(regionOp).getPin();
        if (isFloating) {
          auto exprIt = opPhases.find(regionOp);
          if (exprIt == opPhases.end() || exprIt->second > latest) {
            auto exprOp = cast<ExprOp>(regionOp);
            LLVM_DEBUG({
              llvm::dbgs() << "- Phase " << latest << " for: ";
              regionOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
              llvm::dbgs() << "\n";
            });
            opPhases[regionOp] = latest;
            for (auto result : regionOp->getResults())
              actualPhase[result] = latest;
            for (auto typeVal : exprOp.getResultTypes()) {
              if (failed(constrainValue(typeVal, latest - 1)))
                emitRemark(exprOp.getLoc())
                    << "required by expr result type at phase " << latest - 1;
            }
            processBlock(exprOp.getBody().front(), latest);
          }
        }

        auto valIt = actualPhase.find(value);
        if (valIt == actualPhase.end()) {
          if (failed(constrainValue(value, latest)))
            emitRemark(yieldOp.getLoc()) << "required by yield operand";
          valIt = actualPhase.find(value);
        } else if (valIt->second != INT16_MIN && valIt->second > latest) {
          emitError(value.getLoc())
              << "value at phase " << valIt->second
              << " cannot satisfy requirement for phase " << latest;
          emitRemark(yieldOp.getLoc()) << "required by yield operand";
          anyErrors = true;
        }
        // Propagate actual phase to parent result.
        if (valIt != actualPhase.end())
          actualPhase[regionOp->getResult(resultIdx)] = valIt->second;
      }
      // Type operand for this result.
      auto typeOfValues = yieldOp.getTypeOfValues();
      if (resultIdx < typeOfValues.size()) {
        if (failed(constrainValue(typeOfValues[resultIdx], latest - 1)))
          emitRemark(yieldOp.getLoc()) << "required by yield type operand";
      }
    }
  };

  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    pushToYield(exprOp.getBody());
  } else if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    pushToYield(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      pushToYield(ifOp.getElseRegion());
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    // Loop results come from break ops, not yields.
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end()) {
      for (auto breakOp : it->second) {
        auto values = breakOp.getValues();
        if (resultIdx < values.size()) {
          Value value = values[resultIdx];
          auto valIt = actualPhase.find(value);
          if (valIt == actualPhase.end()) {
            if (failed(constrainValue(value, latest)))
              emitRemark(breakOp.getLoc()) << "required by break operand";
            valIt = actualPhase.find(value);
          } else if (valIt->second != INT16_MIN && valIt->second > latest) {
            emitError(value.getLoc())
                << "value at phase " << valIt->second
                << " cannot satisfy requirement for phase " << latest;
            emitRemark(breakOp.getLoc()) << "required by break operand";
            anyErrors = true;
          }
          // Propagate actual phase to loop result.
          if (valIt != actualPhase.end())
            actualPhase[regionOp->getResult(resultIdx)] = valIt->second;
        }
        // Type operands for break.
        auto typeOfValues = breakOp.getTypeOfValues();
        if (resultIdx < typeOfValues.size()) {
          if (failed(constrainValue(typeOfValues[resultIdx], latest - 1)))
            emitRemark(breakOp.getLoc()) << "required by break type operand";
        }
      }
    }
  }
}

//===----------------------------------------------------------------------===//
// processBlock — Push Block Phase Down to Ops
//
// Walks ops in block order, calling processOp for non-terminators. Then sets
// the terminator phase and validates phase equalities (return at 0, break/
// continue at loop phase).
//===----------------------------------------------------------------------===//

void PhaseAnalysis::processBlock(Block &block, int16_t blockPhase) {
  // Process non-terminator ops.
  for (auto &op : block.without_terminator())
    processOp(&op, blockPhase);

  // Process the terminator.
  auto *terminator = block.getTerminator();
  opPhases[terminator] = blockPhase;

  if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
    auto *parent = yieldOp->getParentOp();
    auto it = resultConstraints.find(parent);
    if (it != resultConstraints.end()) {
      auto &constraints = it->second;
      // Push constraints for results that have already been constrained.
      for (auto [i, value] : llvm::enumerate(yieldOp.getValues())) {
        if (i >= constraints.size() || constraints[i] == kUnconstrained)
          continue;
        if (failed(constrainValue(value, constraints[i])))
          emitRemark(yieldOp.getLoc()) << "required by yield operand";
        // Propagate actual phase to parent result.
        auto valIt = actualPhase.find(value);
        if (valIt != actualPhase.end() && parent->getNumResults() > i)
          actualPhase[parent->getResult(i)] = valIt->second;
      }
      for (auto [i, typeVal] : llvm::enumerate(yieldOp.getTypeOfValues())) {
        if (i >= constraints.size() || constraints[i] == kUnconstrained)
          continue;
        if (failed(constrainValue(typeVal, constraints[i] - 1)))
          emitRemark(yieldOp.getLoc()) << "required by yield type operand";
      }
    }

  } else if (isa<SignatureOp>(terminator)) {
    // Phase validation and operand constraints handled in Step 4 of run().

  } else if (auto returnOp = dyn_cast<ReturnOp>(terminator)) {
    int16_t funcBodyPhase = 0;
    if (blockPhase != funcBodyPhase) {
      emitError(returnOp.getLoc())
          << "return from a phase-shifted block is not allowed";
      anyErrors = true;
    }
    // Operand constraints handled in Step 4 of run().

  } else if (auto breakOp = dyn_cast<BreakOp>(terminator)) {
    auto enclosingLoop = findEnclosing<LoopOp>(breakOp);
    int16_t loopPhase = getPhase(enclosingLoop);
    if (blockPhase != loopPhase) {
      emitError(breakOp.getLoc())
          << "break from a phase-shifted block is not allowed";
      anyErrors = true;
    }
    // Break operand constraints are handled via constrainRegionResult.

  } else if (auto continueOp = dyn_cast<ContinueOp>(terminator)) {
    auto enclosingLoop = findEnclosing<LoopOp>(continueOp);
    int16_t loopPhase = getPhase(enclosingLoop);
    if (blockPhase != loopPhase) {
      emitError(continueOp.getLoc())
          << "continue from a phase-shifted block is not allowed";
      anyErrors = true;
    }

  } else if (isa<UnreachableOp>(terminator)) {
    // Nothing to check.
  } else {
    llvm_unreachable("unexpected terminator in UIR block");
  }
}

//===----------------------------------------------------------------------===//
// processOp — Push Block Phase to a Single Op
//
// Called by processBlock for each non-terminator. Sets the op's phase and
// pushes constraints to operands and regions. Skips floating exprs, pure ops,
// and constants (demand-driven only).
//===----------------------------------------------------------------------===//

void PhaseAnalysis::processOp(Operation *op, int16_t blockPhase) {
  // Floating uir.expr: demand-driven only. Skip.
  if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && !exprOp.getPin())
    return;

  // Pure ops and constants are demand-driven only — unless they are zero-use
  // expression statements, which are anchored at the block phase.
  if (!isa<ExprOp>(op) && op->getNumRegions() == 0 &&
      (hir::isEffectivelyPure(op) || op->hasTrait<OpTrait::ConstantLike>()) &&
      !op->use_empty())
    return;

  // Pinned expr: phase = blockPhase + offset.
  if (auto exprOp = dyn_cast<ExprOp>(op)) {
    assert(exprOp.getPin() && "floating expr should have been skipped");
    int16_t exprPhase = blockPhase + exprOp.getPhaseShift();

    // Tightening check.
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= exprPhase)
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << exprPhase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = exprPhase;
    for (auto result : op->getResults())
      actualPhase[result] = exprPhase;

    // Push result type constraints.
    for (auto typeVal : exprOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, exprPhase - 1)))
        emitRemark(exprOp.getLoc())
            << "required by expr result type at phase " << exprPhase - 1;
    }

    // Initialize result constraints for pinned exprs.
    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(exprOp.getNumResults(), exprPhase);

    processBlock(exprOp.getBody().front(), exprPhase);
    return;
  }

  // PinOp: phase = blockPhase + offset.
  if (auto pinOp = dyn_cast<PinOp>(op)) {
    int16_t pinPhase = blockPhase + pinOp.getPhaseOffset();

    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << pinPhase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = pinPhase;
    for (auto result : op->getResults())
      actualPhase[result] = pinPhase;

    for (auto input : pinOp.getInputs()) {
      if (failed(constrainValue(input, pinPhase)))
        emitRemark(pinOp.getLoc()) << "required by pin at phase " << pinPhase;
    }
    return;
  }

  // IfOp: anchored at blockPhase.
  if (auto ifOp = dyn_cast<IfOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << blockPhase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = blockPhase;
    for (auto result : op->getResults())
      actualPhase[result] = blockPhase;

    if (failed(constrainValue(ifOp.getCondition(), blockPhase)))
      emitRemark(ifOp.getLoc())
          << "required by if condition at phase " << blockPhase;
    for (auto typeVal : ifOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1)))
        emitRemark(ifOp.getLoc())
            << "required by if result type at phase " << blockPhase - 1;
    }

    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(ifOp.getNumResults(), kUnconstrained);

    processBlock(ifOp.getThenRegion().front(), blockPhase);
    if (!ifOp.getElseRegion().empty())
      processBlock(ifOp.getElseRegion().front(), blockPhase);
    return;
  }

  // LoopOp: anchored at blockPhase.
  if (auto loopOp = dyn_cast<LoopOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << blockPhase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = blockPhase;
    for (auto result : op->getResults())
      actualPhase[result] = blockPhase;

    for (auto typeVal : loopOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1)))
        emitRemark(loopOp.getLoc())
            << "required by loop result type at phase " << blockPhase - 1;
    }

    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(loopOp.getNumResults(), kUnconstrained);

    processBlock(loopOp.getBody().front(), blockPhase);
    return;
  }

  // CallOp: anchored at blockPhase.
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= blockPhase)
      return;

    LLVM_DEBUG({
      llvm::dbgs() << "- Phase " << blockPhase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases[op] = blockPhase;

    // Update result phases with per-result offsets.
    auto resultPhases = callOp.getResultPhases();
    for (auto [i, result] : llvm::enumerate(callOp.getResults()))
      actualPhase[result] = blockPhase + static_cast<int16_t>(resultPhases[i]);

    // Push to arguments with per-arg offsets.
    auto argPhases = callOp.getArgPhases();
    for (auto [i, arg] : llvm::enumerate(callOp.getArguments())) {
      int16_t argLatest = blockPhase + static_cast<int16_t>(argPhases[i]);
      if (failed(constrainValue(arg, argLatest)))
        emitRemark(callOp.getLoc())
            << "required by call argument " << i << " at phase " << argLatest;
    }
    // Type operands.
    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfArgs())) {
      int16_t typeLatest = blockPhase + static_cast<int16_t>(argPhases[i]) - 1;
      if (failed(constrainValue(typeVal, typeLatest)))
        emitRemark(callOp.getLoc()) << "required by call type of arg " << i;
    }
    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfResults())) {
      int16_t typeLatest =
          blockPhase + static_cast<int16_t>(resultPhases[i]) - 1;
      if (failed(constrainValue(typeVal, typeLatest)))
        emitRemark(callOp.getLoc()) << "required by call type of result " << i;
    }
    return;
  }

  // Zero-use expression statements and other side-effecting ops.
  auto it = opPhases.find(op);
  if (it != opPhases.end() && it->second <= blockPhase)
    return;

  LLVM_DEBUG({
    llvm::dbgs() << "- Phase " << blockPhase << " for: ";
    op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  opPhases[op] = blockPhase;
  for (auto result : op->getResults())
    actualPhase[result] = blockPhase;

  // Generic ops: push to operands with type operand -1 shift.
  for (auto &operand : op->getOpOperands()) {
    int16_t opLatest = isTypeOperand(operand) ? (blockPhase - 1) : blockPhase;
    if (failed(constrainValue(operand.get(), opLatest)))
      emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
  }
}

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
