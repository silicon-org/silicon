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
#define INDENT llvm::indent(depth * 2)

/// Check whether an operand is a "type operand" — a value that describes a
/// type rather than carrying data. Type operands must be available one phase
/// before the value they describe (the "-1 rule").
///
/// Uses tablegen-generated Mutable accessors to identify type operands without
/// hardcoding indices.
///
/// NOTE: When adding a new HIR op with type operands, add a case here.
/// See also the corresponding op definitions in HIR/Ops.td.
static bool isTypeOperand(OpOperand &operand) {
  return TypeSwitch<Operation *, bool>(operand.getOwner())
      .Case<hir::AddOp, hir::SubOp, hir::MulOp, hir::DivOp, hir::ModOp,
            hir::AndOp, hir::OrOp, hir::XorOp, hir::ShlOp, hir::ShrOp,
            hir::EqOp, hir::NeqOp, hir::LtOp, hir::GtOp, hir::GeqOp,
            hir::LeqOp>(
          [&](auto op) { return &operand == &op.getResultTypeMutable(); })
      .Case<hir::ConstantIntOp, hir::CoerceTypeOp>(
          [&](auto op) { return &operand == &op.getTypeOperandMutable(); })
      .Case<hir::StoreOp>(
          [&](auto op) { return &operand == &op.getValueTypeMutable(); })
      .Default([](auto) { return false; });
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

/// Run the phase analysis. The algorithm has four steps:
/// 1. Pre-walk: collect terminators.
/// 2. Constrain blocks containing function terminators to phase ≤ 0.
/// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
/// 4. Demand-driven constraints: push from declared arg/result phases.
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

  // Step 3b: Result phase floor for function returns. When multiple return ops
  // provide distinct SSA values for a result, the result phase is floored at
  // the body phase (0). If the declared result phase is < 0, this is an error.
  {
    auto resultPhases = funcOp.getResultPhases();
    for (unsigned i = 0; i < resultPhases.size(); ++i) {
      int16_t rp = static_cast<int16_t>(resultPhases[i]);
      if (rp >= 0)
        continue;

      SmallVector<Value, 4> returnValues;
      for (auto *term : funcTerminators)
        if (auto returnOp = dyn_cast<ReturnOp>(term))
          if (i < returnOp.getValues().size())
            returnValues.push_back(returnOp.getValues()[i]);

      if (returnValues.size() <= 1)
        continue;
      bool allSame = llvm::all_of(
          returnValues, [&](Value v) { return v == returnValues[0]; });
      if (allSame)
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

  // Step 5: Reschedule pure ops to their earliest possible phase. During the
  // DFS, pure ops were assigned their demanded phase (`latest`). Now that all
  // phases are final, recompute each pure op's phase as max(operand phases).
  // Walk in pre-order so operands are visited before their consumers.
  LLVM_DEBUG(dbgs() << "  step 5: reschedule pure ops\n");
  funcOp->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!hir::isEffectivelyPure(op) || op->hasTrait<OpTrait::IsTerminator>() ||
        op->getNumRegions() > 0)
      return;
    auto phaseIt = opPhases.find(op);
    if (phaseIt == opPhases.end())
      return;

    int16_t earliest = INT16_MIN;
    if (isa<hir::TypeOfOp>(op)) {
      earliest = actualPhase.at(op->getOperand(0));
      if (earliest != INT16_MIN)
        earliest -= 1;
    } else {
      for (auto &operand : op->getOpOperands()) {
        auto valIt = actualPhase.find(operand.get());
        if (valIt == actualPhase.end())
          continue;
        int16_t opActual = valIt->second;
        if (isTypeOperand(operand) && opActual != INT16_MIN)
          opActual += 1;
        earliest = std::max(earliest, opActual);
      }
    }

    if (earliest < phaseIt->second) {
      LLVM_DEBUG({
        dbgs() << "    " << phaseIt->second << " => " << earliest << ": ";
        op->print(dbgs(), OpPrintingFlags().skipRegions());
        dbgs() << "\n";
      });
      phaseIt->second = earliest;
      for (auto r : op->getResults())
        actualPhase[r] = earliest;
    }
  });

  return success();
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
    opPhases[op] = INT16_MIN;
    for (auto r : op->getResults())
      actualPhase[r] = INT16_MIN;
    return INT16_MIN;
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
    int16_t resolved = (*inputPhase == INT16_MIN) ? INT16_MIN : *inputPhase - 1;
    LLVM_DEBUG(dbgs() << INDENT << "  => phase " << resolved << "\n");
    opPhases[op] = resolved;
    actualPhase[typeOfOp.getResult()] = resolved;
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
/// Yields and breaks are transparent conduits — they just forward
/// constrainValue to the corresponding operand and propagate the resolved phase
/// to the region result.
///
/// For merging CF ops (IfOp, LoopOp), a result phase floor applies: when
/// multiple terminators provide distinct SSA values for a result, the result
/// phase must be >= the block phase (the CF op performs a selection at its
/// block phase, so the result can't be known earlier). When all terminators
/// provide the same SSA value, no floor applies — no selection occurs.
LogicalResult PhaseAnalysis::constrainRegionResult(Operation *regionOp,
                                                   unsigned resultIdx,
                                                   int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Skip if this result already satisfies the demand.
  auto resultIt = actualPhase.find(regionOp->getResult(resultIdx));
  if (resultIt != actualPhase.end() && resultIt->second <= latest)
    return success();

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainRegionResult(result " << resultIdx
           << ", <=" << latest << "): ";
    regionOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  auto constrainTermOperand = [&](Operation *term) -> LogicalResult {
    auto getOperands = [&](Operation *t) -> std::pair<ValueRange, ValueRange> {
      if (auto yieldOp = dyn_cast<YieldOp>(t))
        return {yieldOp.getValues(), yieldOp.getTypeOfValues()};
      if (auto breakOp = dyn_cast<BreakOp>(t))
        return {breakOp.getValues(), breakOp.getTypeOfValues()};
      return {{}, {}};
    };

    auto [values, typeOfValues] = getOperands(term);

    if (resultIdx < values.size()) {
      auto phase = constrainValue(values[resultIdx], latest);
      if (failed(phase)) {
        emitRemark(term->getLoc()) << "required by yield operand";
        return failure();
      }
      actualPhase[regionOp->getResult(resultIdx)] = *phase;
    }

    if (resultIdx < typeOfValues.size()) {
      if (failed(constrainValue(typeOfValues[resultIdx], latest - 1))) {
        emitRemark(term->getLoc()) << "required by yield type operand";
        return failure();
      }
    }

    return success();
  };

  // Get the value operand for `resultIdx` from a yield or break terminator.
  // Returns a null Value for other terminators (return, unreachable).
  auto getTermValue = [&](Operation *term) -> Value {
    if (auto yieldOp = dyn_cast<YieldOp>(term)) {
      auto values = yieldOp.getValues();
      if (resultIdx < values.size())
        return values[resultIdx];
    } else if (auto breakOp = dyn_cast<BreakOp>(term)) {
      auto values = breakOp.getValues();
      if (resultIdx < values.size())
        return values[resultIdx];
    }
    return {};
  };

  // Check the result phase floor for merging CF ops. When multiple terminators
  // provide distinct SSA values for a result, the result phase is floored at
  // the block phase — the CF op performs a selection that can't happen earlier.
  // Each entry is a (value, terminator) pair for diagnostic purposes.
  auto checkFloor =
      [&](SmallVectorImpl<std::pair<Value, Operation *>> &termEntries)
      -> LogicalResult {
    if (termEntries.size() <= 1)
      return success();
    bool allSame = llvm::all_of(termEntries, [&](auto &entry) {
      return entry.first == termEntries[0].first;
    });
    if (allSame)
      return success();

    auto phaseIt = opPhases.find(regionOp);
    if (phaseIt == opPhases.end())
      return success(); // Not yet processed; floor check deferred.
    int16_t blockPhase = phaseIt->second;
    if (latest < blockPhase) {
      auto diag = emitError(regionOp->getLoc())
                  << "result at phase " << latest
                  << " is selected by control flow at phase " << blockPhase
                  << "; result phase must be >= " << blockPhase;
      for (auto &[val, term] : termEntries)
        diag.attachNote(term->getLoc()) << "value provided here";
      return failure();
    }
    return success();
  };

  if (auto exprOp = dyn_cast<ExprOp>(regionOp))
    return constrainTermOperand(exprOp.getBody().front().getTerminator());

  if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    // Collect values for the floor check before constraining.
    SmallVector<std::pair<Value, Operation *>, 2> termEntries;
    auto *thenTerm = ifOp.getThenRegion().front().getTerminator();
    if (auto v = getTermValue(thenTerm))
      termEntries.push_back({v, thenTerm});
    if (!ifOp.getElseRegion().empty()) {
      auto *elseTerm = ifOp.getElseRegion().front().getTerminator();
      if (auto v = getTermValue(elseTerm))
        termEntries.push_back({v, elseTerm});
    }
    if (failed(checkFloor(termEntries)))
      return failure();

    if (failed(
            constrainTermOperand(ifOp.getThenRegion().front().getTerminator())))
      return failure();
    if (!ifOp.getElseRegion().empty())
      if (failed(constrainTermOperand(
              ifOp.getElseRegion().front().getTerminator())))
        return failure();
    return success();
  }

  if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    // Collect values for the floor check before constraining.
    SmallVector<std::pair<Value, Operation *>, 2> termEntries;
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end())
      for (auto breakOp : it->second)
        if (auto v = getTermValue(breakOp))
          termEntries.push_back({v, breakOp});
    if (failed(checkFloor(termEntries)))
      return failure();

    if (it != loopBreaks.end())
      for (auto breakOp : it->second)
        if (failed(constrainTermOperand(breakOp)))
          return failure();
    return success();
  }

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
