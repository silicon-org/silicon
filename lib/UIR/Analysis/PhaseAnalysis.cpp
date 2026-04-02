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
// Debug Helpers
//===----------------------------------------------------------------------===//

/// Print indentation for the current DFS depth.
#define INDENT llvm::indent(depth * 2)

//===----------------------------------------------------------------------===//
// Pre-walk: Collect Terminators
//
// Walks the function IR with manual recursion, passing a `currentLoop` context
// down through regions. This lets us associate break/continue ops with their
// enclosing loop without walking the parent chain at each use.
//===----------------------------------------------------------------------===//

void PhaseAnalysis::collectTerminators() {
  std::function<void(Block &, LoopOp)> walkBlock = [&](Block &block,
                                                       LoopOp currentLoop) {
    for (auto &op : block) {
      if (isa<ReturnOp, SignatureOp>(&op)) {
        funcInterfaceTerminators.push_back(&op);
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
      if (auto loopOp = dyn_cast<LoopOp>(&op)) {
        for (auto &region : op.getRegions())
          for (auto &innerBlock : region)
            walkBlock(innerBlock, loopOp);
      } else {
        for (auto &region : op.getRegions())
          for (auto &innerBlock : region)
            walkBlock(innerBlock, currentLoop);
      }
    }
  };

  for (auto &block : funcOp.getSignature())
    walkBlock(block, /*currentLoop=*/nullptr);
  for (auto &block : funcOp.getBody())
    walkBlock(block, /*currentLoop=*/nullptr);
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::run() {
  LLVM_DEBUG(dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                    << "\n");

  // Seed the function at phase 0.
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

  // Step 2: Constrain blocks containing function-interface terminators.
  LLVM_DEBUG(dbgs() << "  step 2: constrain function-interface blocks\n");
  for (auto *term : funcInterfaceTerminators)
    if (failed(constrainBlock(*term->getBlock(), 0)))
      return failure();

  // Step 3: Structural setup — process blocks.
  LLVM_DEBUG(dbgs() << "  step 3: structural setup\n");
  if (failed(processBlock(funcOp.getSignature().front(), 0)))
    return failure();
  if (failed(processBlock(funcOp.getBody().front(), 0)))
    return failure();

  // Step 4: Demand-driven constraints — function interface.
  LLVM_DEBUG(dbgs() << "  step 4: demand-driven constraints\n");
  for (auto *term : funcInterfaceTerminators) {
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

  return success();
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
// - Call/pin/side-effecting result: constrainBlock, then check actualPhase.
//===----------------------------------------------------------------------===//

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
    opPhases[op] = INT16_MIN;
    for (auto result : op->getResults())
      actualPhase[result] = INT16_MIN;
    return INT16_MIN;
  }

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainValue(<=" << latest << "): ";
    op->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  // hir.type_of: result is at p(operand) - 1. Push latest + 1 to input.
  if (isa<hir::TypeOfOp>(op)) {
    assert(op->getNumOperands() == 1);
    auto inputPhase = constrainValue(op->getOperand(0), latest + 1);
    if (failed(inputPhase))
      return failure();
    int16_t resolved = (*inputPhase == INT16_MIN) ? INT16_MIN : *inputPhase - 1;
    LLVM_DEBUG(dbgs() << INDENT << "  => phase " << resolved << "\n");
    opPhases[op] = resolved;
    for (auto result : op->getResults())
      actualPhase[result] = resolved;
    return resolved;
  }

  // Region ops: push per-result constraint through the yield.
  if (isa<ExprOp, IfOp, LoopOp>(op)) {
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

  // Pure ops (effectively pure): push to operands, then schedule at earliest.
  if (hir::isEffectivelyPure(op)) {
    // Push constraints to operands and collect their resolved phases.
    SmallVector<int16_t> operandPhases;
    for (auto &operand : op->getOpOperands()) {
      int16_t opLatest = isTypeOperand(operand) ? (latest - 1) : latest;
      auto phase = constrainValue(operand.get(), opLatest);
      if (failed(phase)) {
        emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
        return failure();
      }
      operandPhases.push_back(*phase);
    }

    // Post-order: schedule at max(operand actuals), accounting for the type
    // operand -1 rule.
    int16_t earliest = INT16_MIN;
    for (auto [i, operand] : llvm::enumerate(op->getOpOperands())) {
      int16_t opActual = operandPhases[i];
      if (isTypeOperand(operand) && opActual != INT16_MIN)
        opActual += 1;
      earliest = std::max(earliest, opActual);
    }

    // Only update if this is a tighter (earlier) phase or first visit.
    auto it = opPhases.find(op);
    if (it == opPhases.end() || it->second > earliest) {
      LLVM_DEBUG(dbgs() << INDENT << "  => phase " << earliest << "\n");
      opPhases[op] = earliest;
      for (auto result : op->getResults())
        actualPhase[result] = earliest;
    }

    // After constraining all operands successfully, the op's phase must
    // satisfy the demand.
    assert(actualPhase.at(value) <= latest &&
           "pure op phase exceeds demand after successful operand constraints");
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

//===----------------------------------------------------------------------===//
// constrainBlock — Block Phase Demand
//
// Someone needs this block at phase ≤ demanded. Dispatches on the parent op:
// - Floating expr: tighten block phase, re-process.
// - Pinned expr at offset N: constrainBlock(parentBlock, demanded - N).
// - Anchored CF (if/loop): constrainBlock(parentBlock, demanded).
// - Function body/signature: phase is 0 (no action needed here).
//===----------------------------------------------------------------------===//

FailureOr<int16_t> PhaseAnalysis::constrainBlock(Block &block,
                                                 int16_t demandedPhase) {
  SaveAndRestore guard(depth, depth + 2);
  auto *parentOp = block.getParentOp();
  assert(parentOp && "block has no parent op");

  // Function body/signature: phase is fixed at 0.
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
      // Pinned expr at offset N: propagate to parent block with adjusted
      // demand. Block phase = parentBlockPhase + offset.
      auto parentPhase = constrainBlock(*parentOp->getBlock(),
                                        demandedPhase - exprOp.getPhaseShift());
      if (failed(parentPhase))
        return failure();
      return int16_t(*parentPhase + exprOp.getPhaseShift());
    }

    // Floating expr: tighten block phase if needed, then re-process.
    auto it = opPhases.find(parentOp);
    if (it != opPhases.end() && it->second <= demandedPhase)
      return it->second; // already tight enough

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

  // Anchored CF (if/loop): demand propagates to parent block. The block phase
  // is the same as the parent block phase.
  if (isa<IfOp, LoopOp>(parentOp))
    return constrainBlock(*parentOp->getBlock(), demandedPhase);

  llvm_unreachable("unexpected parent op in constrainBlock");
}

//===----------------------------------------------------------------------===//
// constrainRegionResult — Per-Result Constraint Through Yield/Break
//
// Pushes a phase constraint onto a specific result of a region-bearing op by
// forwarding constrainValue to the corresponding yield/break operand. The
// resolved actual phase is propagated back to the region result. Yields and
// breaks are pure conduits — they don't create errors or inspect phases
// themselves.
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::constrainRegionResult(Operation *regionOp,
                                                   unsigned resultIdx,
                                                   int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Skip if this result already has a phase that satisfies the demand.
  auto resultIt = actualPhase.find(regionOp->getResult(resultIdx));
  if (resultIt != actualPhase.end() && resultIt->second <= latest)
    return success();

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainRegionResult(result " << resultIdx
           << ", <=" << latest << "): ";
    regionOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  // Forward the constraint through a single terminator's operand.
  auto constrainTermOperand = [&](Operation *term) -> LogicalResult {
    // Get the value and type operands from the terminator.
    auto getOperands = [&](Operation *t) -> std::pair<ValueRange, ValueRange> {
      if (auto yieldOp = dyn_cast<YieldOp>(t))
        return {yieldOp.getValues(), yieldOp.getTypeOfValues()};
      if (auto breakOp = dyn_cast<BreakOp>(t))
        return {breakOp.getValues(), breakOp.getTypeOfValues()};
      return {{}, {}};
    };

    auto [values, typeOfValues] = getOperands(term);

    // Constrain the value operand.
    if (resultIdx < values.size()) {
      auto phase = constrainValue(values[resultIdx], latest);
      if (failed(phase)) {
        emitRemark(term->getLoc()) << "required by yield operand";
        return failure();
      }
      // Propagate actual phase to parent result.
      actualPhase[regionOp->getResult(resultIdx)] = *phase;
    }

    // Constrain the type operand (type operand -1 rule).
    if (resultIdx < typeOfValues.size()) {
      if (failed(constrainValue(typeOfValues[resultIdx], latest - 1))) {
        emitRemark(term->getLoc()) << "required by yield type operand";
        return failure();
      }
    }

    return success();
  };

  // Dispatch to all terminators that produce this result.
  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    return constrainTermOperand(exprOp.getBody().front().getTerminator());
  } else if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    if (failed(
            constrainTermOperand(ifOp.getThenRegion().front().getTerminator())))
      return failure();
    if (!ifOp.getElseRegion().empty())
      if (failed(constrainTermOperand(
              ifOp.getElseRegion().front().getTerminator())))
        return failure();
    return success();
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end()) {
      for (auto breakOp : it->second)
        if (failed(constrainTermOperand(breakOp)))
          return failure();
    }
    return success();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// processBlock — Push Block Phase Down to Ops
//
// Walks all ops in a block (including the terminator), calling processOp on
// each. processOp handles phase assignment and validation for all op types.
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::processBlock(Block &block, int16_t blockPhase) {
  SaveAndRestore guard(depth, depth + 2);
  LLVM_DEBUG(dbgs() << INDENT << "processBlock(phase " << blockPhase << ")\n");

  for (auto &op : block)
    if (failed(processOp(&op, blockPhase)))
      return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// processOp — Push Block Phase to a Single Op
//
// Called by processBlock for each non-terminator. Sets the op's phase and
// pushes constraints to operands and regions. Skips floating exprs, pure ops,
// and constants (demand-driven only).
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::processOp(Operation *op, int16_t blockPhase) {
  // Floating uir.expr: demand-driven only. Skip.
  if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && !exprOp.getPin())
    return success();

  // Pure ops and constants are demand-driven only — unless they are zero-use
  // expression statements, which are anchored at the block phase.
  if (!isa<ExprOp>(op) && op->getNumRegions() == 0 &&
      (hir::isEffectivelyPure(op) || op->hasTrait<OpTrait::ConstantLike>()) &&
      !op->use_empty())
    return success();

  SaveAndRestore guard(depth, depth + 2);

  // Pinned expr: phase = blockPhase + offset.
  if (auto exprOp = dyn_cast<ExprOp>(op)) {
    assert(exprOp.getPin() && "floating expr should have been skipped");
    int16_t exprPhase = blockPhase + exprOp.getPhaseShift();

    // Tightening check.
    auto it = opPhases.find(op);
    if (it != opPhases.end() && it->second <= exprPhase)
      return success();

    LLVM_DEBUG(dbgs() << INDENT << "processOp(pinnedExpr, phase " << exprPhase
                      << ")\n");
    opPhases[op] = exprPhase;

    // Push result type constraints.
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
    for (auto result : op->getResults())
      actualPhase[result] = pinPhase;

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

  // LoopOp: anchored at blockPhase.
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

    // Update result phases with per-result offsets.
    auto resultPhases = callOp.getResultPhases();
    for (auto [i, result] : llvm::enumerate(callOp.getResults()))
      actualPhase[result] = blockPhase + static_cast<int16_t>(resultPhases[i]);

    // Constrain arguments.
    auto argPhases = callOp.getArgPhases();
    for (auto [i, arg] : llvm::enumerate(callOp.getArguments())) {
      int16_t argLatest = blockPhase + static_cast<int16_t>(argPhases[i]);
      if (failed(constrainValue(arg, argLatest))) {
        emitRemark(callOp.getLoc())
            << "required by call argument " << i << " at phase " << argLatest;
        return failure();
      }
    }

    // Constrain argument types.
    for (auto [i, typeVal] : llvm::enumerate(callOp.getTypeOfArgs())) {
      int16_t typeLatest = blockPhase + static_cast<int16_t>(argPhases[i]) - 1;
      if (failed(constrainValue(typeVal, typeLatest))) {
        emitRemark(callOp.getLoc())
            << "required by type of call argument " << i;
        return failure();
      }
    }

    // Constrain result types.
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
  // Yield/signature operand constraints are handled elsewhere (constrainRegion-
  // Result and run() step 4, respectively).
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
    if (blockPhase != 0) {
      emitError(op->getLoc())
          << "signature in a phase-shifted block is not allowed";
      return failure();
    }
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
    return success();
  }

  // Zero-use expression statements and other side-effecting ops.
  auto it = opPhases.find(op);
  if (it != opPhases.end() && it->second <= blockPhase)
    return success();

  LLVM_DEBUG({
    dbgs() << INDENT << "processOp(generic, phase " << blockPhase << "): ";
    op->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });
  opPhases[op] = blockPhase;
  for (auto result : op->getResults())
    actualPhase[result] = blockPhase;

  // Generic ops: push to operands with type operand -1 shift.
  for (auto &operand : op->getOpOperands()) {
    int16_t opLatest = isTypeOperand(operand) ? (blockPhase - 1) : blockPhase;
    if (failed(constrainValue(operand.get(), opLatest))) {
      emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
      return failure();
    }
  }
  return success();
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
