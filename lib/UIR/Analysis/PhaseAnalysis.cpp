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
// type operand. resolveOp pushes `phase - 1` to type operands instead of
// `phase`.
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

  // Process signature and body uniformly.
  processBlock(funcOp.getSignature().front(), 0);
  processBlock(funcOp.getBody().front(), 0);

  return anyErrors ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Block Processing
//
// Walks ops in block order, identifying roots (pins, pinned exprs, zero-use
// statements) and resolving them top-to-bottom. Floating ops are only reached
// via use-def DFS from roots or the terminator.
//===----------------------------------------------------------------------===//

void PhaseAnalysis::processBlock(Block &block, int16_t blockPhase) {
  // Step 1: Process roots in block order.
  for (auto &op : block.without_terminator()) {
    if (auto pinOp = dyn_cast<PinOp>(&op)) {
      // Root: pinned at blockPhase + offset.
      int16_t pinPhase = blockPhase + pinOp.getPhaseOffset();
      resolveOp(pinOp, pinPhase);
    } else if (auto exprOp = dyn_cast<ExprOp>(&op); exprOp && exprOp.getPin()) {
      // Root: pinned expr at blockPhase + phaseShift.
      int16_t exprPhase = blockPhase + exprOp.getPhaseShift();
      resolveOp(exprOp, exprPhase);
    } else if (op.use_empty()) {
      // Root: expression statement (zero-use, pinned at blockPhase).
      resolveOp(&op, blockPhase);
    } else if (!isa<ExprOp>(&op) &&
               (!hir::isEffectivelyPure(&op) ||
                op.getNumRegions() > 0) &&
               !op.hasTrait<OpTrait::ConstantLike>()) {
      // Side-effecting ops (calls, CF) are anchored at the block phase.
      // Region-bearing ops (if, loop) are always anchored even if their
      // recursive memory effects happen to be pure — CF is side-effecting.
      // Only floating uir.expr, pure ops, and constants may float.
      resolveOp(&op, blockPhase);
    }
    // Floating uir.expr, pure ops, and constants are reached via use-def
    // from roots/terminator.
  }

  // Step 2: Process the terminator.
  auto *terminator = block.getTerminator();
  opPhases[terminator] = blockPhase;

  if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
    auto *parent = yieldOp->getParentOp();
    auto it = resultConstraints.find(parent);
    assert(it != resultConstraints.end() && "missing result constraints");
    auto &constraints = it->second;

    // Only push constraints for results that have been constrained.
    // Unconstrained results are skipped — their yield operands will be
    // resolved when a consumer demands them via constrainResult.
    for (auto [i, value] : llvm::enumerate(yieldOp.getValues())) {
      if (i >= constraints.size() || constraints[i] == kUnconstrained)
        continue;
      if (failed(resolveValue(value, constraints[i])))
        emitRemark(yieldOp.getLoc()) << "required by yield operand";
      // Propagate actual phase to parent result.
      auto valIt = actualPhase.find(value);
      if (valIt != actualPhase.end() && parent->getNumResults() > i)
        actualPhase[parent->getResult(i)] = valIt->second;
    }
    for (auto [i, typeVal] : llvm::enumerate(yieldOp.getTypeOfValues())) {
      if (i >= constraints.size() || constraints[i] == kUnconstrained)
        continue;
      if (failed(resolveValue(typeVal, constraints[i] - 1)))
        emitRemark(yieldOp.getLoc()) << "required by yield type operand";
    }

  } else if (auto sigOp = dyn_cast<SignatureOp>(terminator)) {
    auto enclosingFunc = findEnclosing<FuncOp>(sigOp);
    auto argPhases = enclosingFunc.getArgPhases();
    auto resultPhases = enclosingFunc.getResultPhases();

    for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfArgs())) {
      int16_t latest = static_cast<int16_t>(argPhases[i]) - 1;
      if (failed(resolveValue(typeVal, latest)))
        emitRemark(sigOp.getLoc()) << "required by signature type of arg " << i;
    }
    for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfResults())) {
      int16_t latest = static_cast<int16_t>(resultPhases[i]) - 1;
      if (failed(resolveValue(typeVal, latest)))
        emitRemark(sigOp.getLoc())
            << "required by signature type of result " << i;
    }

  } else if (auto returnOp = dyn_cast<ReturnOp>(terminator)) {
    int16_t funcBodyPhase = 0;
    if (blockPhase != funcBodyPhase) {
      emitError(returnOp.getLoc())
          << "return from a phase-shifted block is not allowed";
      anyErrors = true;
    }

    auto enclosingFunc = findEnclosing<FuncOp>(returnOp);
    auto resultPhases = enclosingFunc.getResultPhases();
    for (auto [i, value] : llvm::enumerate(returnOp.getValues())) {
      int16_t latest = static_cast<int16_t>(resultPhases[i]);
      if (failed(resolveValue(value, latest)))
        emitRemark(returnOp.getLoc())
            << "required by return value at phase " << latest;
    }
    for (auto [i, typeVal] : llvm::enumerate(returnOp.getTypeOfValues())) {
      int16_t typeLatest = static_cast<int16_t>(resultPhases[i]) - 1;
      if (failed(resolveValue(typeVal, typeLatest)))
        emitRemark(returnOp.getLoc()) << "required by return type operand";
    }

  } else if (auto breakOp = dyn_cast<BreakOp>(terminator)) {
    auto enclosingLoop = findEnclosing<LoopOp>(breakOp);
    int16_t loopPhase = getPhase(enclosingLoop);
    if (blockPhase != loopPhase) {
      emitError(breakOp.getLoc())
          << "break from a phase-shifted block is not allowed";
      anyErrors = true;
    }

    auto it = resultConstraints.find(enclosingLoop.getOperation());
    assert(it != resultConstraints.end() && "missing loop result constraints");
    auto &constraints = it->second;
    for (auto [i, value] : llvm::enumerate(breakOp.getValues())) {
      if (i >= constraints.size() || constraints[i] == kUnconstrained)
        continue;
      if (failed(resolveValue(value, constraints[i])))
        emitRemark(breakOp.getLoc()) << "required by break operand";
      // Propagate actual phase to loop result.
      auto valIt = actualPhase.find(value);
      if (valIt != actualPhase.end() &&
          enclosingLoop->getNumResults() > i)
        actualPhase[enclosingLoop->getResult(i)] = valIt->second;
    }
    for (auto [i, typeVal] : llvm::enumerate(breakOp.getTypeOfValues())) {
      if (i >= constraints.size() || constraints[i] == kUnconstrained)
        continue;
      if (failed(resolveValue(typeVal, constraints[i] - 1)))
        emitRemark(breakOp.getLoc()) << "required by break type operand";
    }

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
// resolveValue — Thin Value-Level Wrapper
//
// Translates a consumer's phase constraint on a value to an op-level
// resolveOp call, then checks the resolved actual phase against the
// constraint. For call results, applies the result offset before delegating.
//===----------------------------------------------------------------------===//

FailureOr<int16_t> PhaseAnalysis::resolveValue(Value value, int16_t latest) {
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

  // For call results, translate the value-level constraint to an op-level
  // constraint by subtracting the result's phase offset.
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto resultPhases = callOp.getResultPhases();
    int16_t resultOffset = static_cast<int16_t>(resultPhases[resultIdx]);
    resolveOp(callOp, latest - resultOffset);
  } else if (isa<ExprOp, IfOp, LoopOp>(op)) {
    // Region ops: ensure the op has been processed (block entered), then
    // push per-result constraint through the yield.
    if (!opPhases.count(op))
      resolveOp(op, latest);
    constrainResult(op, resultIdx, latest);
  } else {
    resolveOp(op, latest);
  }

  // Check the resolved actual phase against the consumer's constraint.
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
// resolveOp — Op-Level Phase Resolution
//
// The single place for all op-level phase logic. Handles:
// - Tightening check (skip if already at tighter-or-equal phase)
// - Pinned ops (never tighten, authoritative phase)
// - ConstantLike (float at INT16_MIN)
// - hir.type_of (+1 shift on input, result at p(input) - 1)
// - Calls (per-arg offsets, type operand -1)
// - Region ops (if/loop/expr: condition, resultTypes -1, enter regions)
// - Generic ops (type operand mask, pure op earliest scheduling)
//===----------------------------------------------------------------------===//

void PhaseAnalysis::resolveOp(Operation *op, int16_t phase) {
  auto it = opPhases.find(op);
  if (it != opPhases.end()) {
    // Already resolved. Pinned ops never tighten.
    bool isPinned =
        isa<PinOp>(op) || (isa<ExprOp>(op) && cast<ExprOp>(op).getPin());
    if (isPinned || it->second <= phase)
      return;
  }

  // ConstantLike ops float to any phase.
  if (op->hasTrait<OpTrait::ConstantLike>()) {
    opPhases[op] = INT16_MIN;
    for (auto result : op->getResults())
      actualPhase[result] = INT16_MIN;
    return;
  }

  // hir.type_of: result is at p(operand) - 1. Push latest + 1 to input.
  if (isa<hir::TypeOfOp>(op)) {
    assert(op->getNumOperands() == 1);
    (void)resolveValue(op->getOperand(0), phase + 1);
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
    return;
  }

  // Set the op's phase and result phases.
  LLVM_DEBUG({
    llvm::dbgs() << "- Phase " << phase << " for: ";
    op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  opPhases[op] = phase;
  for (auto result : op->getResults())
    actualPhase[result] = phase;

  // Push constraints to operands, dispatch by op type.
  if (auto pinOp = dyn_cast<PinOp>(op)) {
    for (auto input : pinOp.getInputs()) {
      if (failed(resolveValue(input, phase)))
        emitRemark(pinOp.getLoc()) << "required by pin at phase " << phase;
    }

  } else if (auto exprOp = dyn_cast<ExprOp>(op)) {
    for (auto typeVal : exprOp.getResultTypes()) {
      if (failed(resolveValue(typeVal, phase - 1)))
        emitRemark(exprOp.getLoc())
            << "required by expr result type at phase " << phase - 1;
    }
    // Pinned exprs impose the pinned phase as the result constraint.
    // Floating exprs start unconstrained — consumers push per-result
    // constraints via constrainResult.
    auto &constraints = resultConstraints[exprOp.getOperation()];
    if (constraints.empty()) {
      if (exprOp.getPin())
        constraints.assign(exprOp.getNumResults(), phase);
      else
        constraints.assign(exprOp.getNumResults(), kUnconstrained);
    }
    processBlock(exprOp.getBody().front(), phase);

  } else if (auto ifOp = dyn_cast<IfOp>(op)) {
    if (failed(resolveValue(ifOp.getCondition(), phase)))
      emitRemark(ifOp.getLoc())
          << "required by if condition at phase " << phase;
    for (auto typeVal : ifOp.getResultTypes()) {
      if (failed(resolveValue(typeVal, phase - 1)))
        emitRemark(ifOp.getLoc())
            << "required by if result type at phase " << phase - 1;
    }
    auto &constraints = resultConstraints[ifOp.getOperation()];
    if (constraints.empty())
      constraints.assign(ifOp.getNumResults(), kUnconstrained);
    processBlock(ifOp.getThenRegion().front(), phase);
    if (!ifOp.getElseRegion().empty())
      processBlock(ifOp.getElseRegion().front(), phase);

  } else if (auto loopOp = dyn_cast<LoopOp>(op)) {
    for (auto typeVal : loopOp.getResultTypes()) {
      if (failed(resolveValue(typeVal, phase - 1)))
        emitRemark(loopOp.getLoc())
            << "required by loop result type at phase " << phase - 1;
    }
    auto &constraints = resultConstraints[loopOp.getOperation()];
    if (constraints.empty())
      constraints.assign(loopOp.getNumResults(), kUnconstrained);
    processBlock(loopOp.getBody().front(), phase);

  } else if (auto callOp = dyn_cast<CallOp>(op)) {
    // Update result phases with per-result offsets.
    auto resultPhases = callOp.getResultPhases();
    for (auto [i, result] : llvm::enumerate(callOp.getResults()))
      actualPhase[result] = phase + static_cast<int16_t>(resultPhases[i]);

    // Push to arguments with per-arg offsets.
    auto argPhases = callOp.getArgPhases();
    for (auto [i, arg] : llvm::enumerate(callOp.getArguments())) {
      int16_t argLatest = phase + static_cast<int16_t>(argPhases[i]);
      if (failed(resolveValue(arg, argLatest)))
        emitRemark(callOp.getLoc())
            << "required by call argument " << i << " at phase " << argLatest;
    }
    // Type operands: type of arg i at callPhase + argOffset - 1, etc.
    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfArgs())) {
      int16_t typeLatest = phase + static_cast<int16_t>(argPhases[i]) - 1;
      if (failed(resolveValue(typeVal, typeLatest)))
        emitRemark(callOp.getLoc()) << "required by call type of arg " << i;
    }
    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfResults())) {
      int16_t typeLatest = phase + static_cast<int16_t>(resultPhases[i]) - 1;
      if (failed(resolveValue(typeVal, typeLatest)))
        emitRemark(callOp.getLoc()) << "required by call type of result " << i;
    }

  } else {
    // Generic ops: push to operands with type operand -1 shift.
    // Pure ops get post-order earliest scheduling.
    for (auto &operand : op->getOpOperands()) {
      int16_t opLatest = isTypeOperand(operand) ? (phase - 1) : phase;
      if (failed(resolveValue(operand.get(), opLatest)))
        emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
    }

    // Post-order: pure ops schedule at max(operand actuals), accounting for
    // the type operand -1 rule. Type operands contribute actualPhase + 1 to
    // earliest, since the op must be at least typePhase + 1 for the type to
    // be available one phase before the op.
    if (hir::isEffectivelyPure(op)) {
      int16_t earliest = INT16_MIN;
      for (auto &operand : op->getOpOperands()) {
        int16_t opActual = actualPhase.lookup(operand.get());
        if (isTypeOperand(operand) && opActual != INT16_MIN)
          opActual += 1;
        earliest = std::max(earliest, opActual);
      }
      opPhases[op] = earliest;
      for (auto result : op->getResults())
        actualPhase[result] = earliest;
    }
  }
}

//===----------------------------------------------------------------------===//
// constrainResult — Per-Result Constraint Push
//
// Pushes a phase constraint onto a specific result of a region-bearing op.
// The constraint propagates through the yield to the corresponding operand.
// This enables sparse, per-result constraints: different consumers can demand
// different results at different phases, and only constrained results push
// into the DFS.
//===----------------------------------------------------------------------===//

void PhaseAnalysis::constrainResult(Operation *regionOp, unsigned resultIdx,
                                    int16_t latest) {
  // Lazily initialize result constraints if this region op hasn't been
  // processed by resolveOp yet (consumer reached it before a root did).
  auto &constraints = resultConstraints[regionOp];
  if (constraints.empty())
    constraints.assign(regionOp->getNumResults(), kUnconstrained);
  assert(resultIdx < constraints.size());

  // Only tighten: skip if already at a tighter-or-equal constraint.
  if (constraints[resultIdx] != kUnconstrained &&
      constraints[resultIdx] <= latest)
    return;
  constraints[resultIdx] = latest;

  // Find the yield(s) in the region(s) and push the constraint to the
  // corresponding operand. For IfOp, push into both then/else regions.
  auto pushToYield = [&](Region &region) {
    auto *term = region.front().getTerminator();
    if (auto yieldOp = dyn_cast<YieldOp>(term)) {
      auto values = yieldOp.getValues();
      if (resultIdx < values.size()) {
        Value value = values[resultIdx];
        // If the value hasn't been resolved yet, push the constraint to
        // resolve it. If already resolved, just check the constraint.
        bool isFloating =
            isa<ExprOp>(regionOp) && !cast<ExprOp>(regionOp).getPin();

        // For floating exprs, tighten the block phase first so that
        // side-effecting ops inside re-anchor at the tighter phase.
        // This must happen before resolving the yield operand, because
        // the operand may depend on a call anchored at the block phase.
        if (isFloating) {
          auto exprIt = opPhases.find(regionOp);
          if (exprIt != opPhases.end() && exprIt->second > latest)
            resolveOp(regionOp, latest);
        }

        auto valIt = actualPhase.find(value);
        if (valIt == actualPhase.end()) {
          // Not yet resolved — push the constraint to resolve it.
          if (failed(resolveValue(value, latest)))
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
        if (failed(resolveValue(typeOfValues[resultIdx], latest - 1)))
          emitRemark(yieldOp.getLoc()) << "required by yield type operand";
      }
    }
    // BreakOp in loops is handled separately via processBlock.
  };

  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    pushToYield(exprOp.getBody());
  } else if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    pushToYield(ifOp.getThenRegion());
    if (!ifOp.getElseRegion().empty())
      pushToYield(ifOp.getElseRegion());
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    // Loop results come from break ops, not directly from yields.
    // The break handler reads resultConstraints, so updating the constraint
    // above is sufficient — break ops will pick up the tightened value when
    // the block is (re-)processed.
    // However, we may need to re-process the loop body if the constraint
    // tightened and there are break ops that haven't seen the new value.
    // For now, the break handler in processBlock already reads from
    // resultConstraints, so it will use the latest value.
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
