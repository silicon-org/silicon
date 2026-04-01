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
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

void PhaseAnalysis::collectTerminators() {
  // Walk a block's ops, passing the current enclosing loop context.
  std::function<void(Block &, LoopOp)> walkBlock = [&](Block &block,
                                                       LoopOp currentLoop) {
    for (auto &op : block) {
      // Collect function-interface terminators.
      if (isa<ReturnOp, SignatureOp>(&op)) {
        // REVIEW: let's call this "funcTerminators"
        funcInterfaceTerminators.push_back(&op);
        continue;
      }

      // Collect break/continue and map to enclosing loop.
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

      // Recurse into regions. LoopOp updates the currentLoop context.
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
//
// The algorithm has four steps:
// 1. Pre-walk: collect terminators.
// 2. Constrain blocks containing function-interface terminators to phase ≤ 0.
// 3. Structural setup: processBlock(sig, 0) and processBlock(body, 0).
// 4. Demand-driven constraints: push from declared arg/result phases.
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::run() {
  LLVM_DEBUG(dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
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
    LLVM_DEBUG(if (argPhase != 0) dbgs()
               << "  arg " << idx << " has phase " << argPhase << "\n");
  }

  // Step 1: Pre-walk — collect terminators.
  collectTerminators();

  // Step 2: Constrain blocks containing function-interface terminators.
  // This ensures any floating expr containing a return or signature gets
  // constrained to phase ≤ 0.
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

      // Constrain argument types.
      for (auto [i, typeVal] : llvm::enumerate(sigOp.getTypeOfArgs())) {
        int16_t latest = static_cast<int16_t>(argPhases[i]) - 1;
        if (failed(constrainValue(typeVal, latest))) {
          emitRemark(sigOp.getLoc())
              << "required by signature type of arg " << i;
          return failure();
        }
      }

      // Constrain result types.
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

      // Constrain result values.
      for (auto [i, value] : llvm::enumerate(returnOp.getValues())) {
        int16_t latest = static_cast<int16_t>(resultPhases[i]);
        if (failed(constrainValue(value, latest))) {
          emitRemark(returnOp.getLoc())
              << "required by return value at phase " << latest;
          return failure();
        }
      }

      // Constrain result types.
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
// - Call result: constrainBlock with adjusted demand.
// - Pin result: constrainBlock with adjusted demand.
// - Other side-effecting result: constrainBlock(parentBlock, latest).
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

// REVIEW: Add comments to disabled lint warning about recursion
FailureOr<int16_t> PhaseAnalysis::constrainValue(Value value, int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Block arguments have fixed phases.
  if (isa<BlockArgument>(value)) {
    // REVIEW: use at(...)
    auto it = actualPhase.find(value);
    assert(it != actualPhase.end() && "block arg phase not seeded");
    int16_t phase = it->second;
    LLVM_DEBUG(dbgs() << INDENT << "constrainValue(blockArg, <=" << latest
                      << "): actual " << phase << "\n");
    // REVIEW: Isn't the != INT16_MIN always true implicitly by phase > latest?
    if (phase != INT16_MIN && phase > latest) {
      // REVIEW: say something like "argument `xyz` at phase ..."
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
    // REVIEW: Just print the result as an SSA operand
    op->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  // hir.type_of: result is at p(operand) - 1. Push latest + 1 to input.
  // REVIEW: use dyn_cast and to get access to convenience operand accessors
  if (isa<hir::TypeOfOp>(op)) {
    assert(op->getNumOperands() == 1);
    if (failed(constrainValue(op->getOperand(0), latest + 1)))
      return failure();
    // REVIEW: instead of looking up the actual phase everywhere, couldn't you
    // just use the result of `constrainValue` which already returns that phase?
    int16_t inputPhase = actualPhase.lookup(op->getOperand(0));
    int16_t resolved = (inputPhase == INT16_MIN) ? INT16_MIN : inputPhase - 1;
    // REVIEW: print the result as ssa value again
    LLVM_DEBUG(dbgs() << INDENT << "  => phase " << resolved << "\n");
    opPhases[op] = resolved;
    // REVIEW: there's only one result if you use dyn_cast
    for (auto result : op->getResults())
      actualPhase[result] = resolved;
    return resolved;
  }

  // Region ops: push per-result constraint through the yield.
  if (isa<ExprOp, IfOp, LoopOp>(op)) {
    // REVIEW: Should we iterate over the regions of the op, and make
    // constrainRegionResult take a `Region &` arg instead of `op`?
    if (failed(constrainRegionResult(op, resultIdx, latest)))
      return failure();
    // REVIEW: Again, can't we make constrainRegionResult return the final value
    // assigned to the result passed into the function, such that we can save
    // the lookup here?
    int16_t actual = actualPhase.lookup(value);
    // REVIEW: INT16_MIN check really needed?
    if (actual != INT16_MIN && actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // Call results: anchored at blockPhase + resultOffset. Calls are fixed at
  // their enclosing block's phase, same as pins. constrainBlock propagates
  // demand for floating exprs (tightening their block phase).
  if (auto callOp = dyn_cast<CallOp>(op)) {
    auto resultPhases = callOp.getResultPhases();
    int16_t resultOffset = static_cast<int16_t>(resultPhases[resultIdx]);
    int16_t demandedBlockPhase = latest - resultOffset;

    if (failed(constrainBlock(*op->getBlock(), demandedBlockPhase))) {
      emitRemark(op->getLoc())
          << "required by call result " << resultIdx << " at phase " << latest;
      return failure();
    }

    int16_t actual = actualPhase.lookup(value);
    if (actual > latest) {
      emitError(op->getLoc())
          << "call result " << resultIdx << " at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // PinOp: anchored at blockPhase + offset. Pins fix a value's phase and
  // cannot be tightened by consumer demand. constrainBlock propagates demand
  // for floating exprs.
  if (auto pinOp = dyn_cast<PinOp>(op)) {
    int16_t offset = pinOp.getPhaseOffset();
    int16_t demandedBlockPhase = latest - offset;

    if (failed(constrainBlock(*op->getBlock(), demandedBlockPhase))) {
      emitRemark(op->getLoc()) << "required by pin at phase " << latest;
      return failure();
    }

    int16_t actual = actualPhase.lookup(value);
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
    // REVIEW: Let's have a local SmallVector of the operand phases that we get
    // back from `constrainValue`, and then let's use them instead of
    // `actualPhase.lookup` further down.

    // REVIEW: instead of eagerly pushing constraints to operands, let's first
    // check if latest < phase(op). If it isn't, we can just return the phase we
    // already have before ever propagating to operands.

    // Push constraints to operands.
    for (auto &operand : op->getOpOperands()) {
      int16_t opLatest = isTypeOperand(operand) ? (latest - 1) : latest;
      if (failed(constrainValue(operand.get(), opLatest))) {
        emitRemark(op->getLoc()) << "required by operand at phase " << opLatest;
        return failure();
      }
    }

    // Post-order: schedule at max(operand actuals), accounting for the type
    // operand -1 rule.
    int16_t earliest = INT16_MIN;
    for (auto &operand : op->getOpOperands()) {
      // REVIEW: use phase vector
      int16_t opActual = actualPhase.lookup(operand.get());
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

    int16_t actual = actualPhase.lookup(value);
    // REVIEW: We should just assert actual <= latest here. The constrainValue
    // calls above will either return an error if they can't at least satisfy
    // "latest", or otherwise they return a phase and we'll end up down here. In
    // any case, we only get here if the operands could satisfy the latest
    // phase. If this breaks somewhere, we need to discuss what's going on. Can
    // we also take a note in the `constrain*` function doc comments that the
    // function must return either with a phase that is <= latest, or it must
    // return with a failure. The exception might be `constrainBlock`, which may
    // return a concrete phase in case the block is fixed. We somehow have to
    // make sure that we don't report block phase issues for blocks that
    // obviously cannot be moved (e.g. ones anchored to the func directly or
    // transitively). In those cases we want the constrainValue of the op
    // anchored to the block to emit the first leaf error.
    if (actual != INT16_MIN && actual > latest) {
      emitError(op->getLoc())
          << "value at phase " << actual
          << " cannot satisfy requirement for phase " << latest;
      return failure();
    }
    return actual;
  }

  // Other side-effecting ops: anchored at block phase, same as calls/pins.
  int16_t actual = actualPhase.lookup(value);
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
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::constrainBlock(Block &block,
                                            int16_t demandedPhase) {
  SaveAndRestore guard(depth, depth + 2);
  auto *parentOp = block.getParentOp();
  // REVIEW: assert here -- parent op should never be null
  if (!parentOp)
    return success();

  // Function body/signature: phase is fixed at 0. Nothing to constrain.
  if (isa<FuncOp>(parentOp)) {
    LLVM_DEBUG(dbgs() << INDENT << "constrainBlock(<=" << demandedPhase
                      << "): func body, fixed at 0\n");
    return success();
  }

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainBlock(<=" << demandedPhase << "): ";
    parentOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  if (auto exprOp = dyn_cast<ExprOp>(parentOp)) {
    if (exprOp.getPin()) {
      // Pinned expr at offset N: need p(parentBlock) + N ≤ demanded,
      // so constrainBlock(parentBlock, demanded - N).
      return constrainBlock(*parentOp->getBlock(),
                            demandedPhase - exprOp.getPhaseShift());
    } else {
      // Floating expr: tighten block phase if needed, then re-process.
      auto it = opPhases.find(parentOp);
      if (it != opPhases.end() && it->second <= demandedPhase)
        return success(); // already tight enough

      // Tighten: set the new phase and re-process the block.
      LLVM_DEBUG(dbgs() << INDENT << "  floating expr => phase "
                        << demandedPhase << "\n");
      opPhases[parentOp] = demandedPhase;
      for (auto typeVal : exprOp.getResultTypes()) {
        if (failed(constrainValue(typeVal, demandedPhase - 1))) {
          emitRemark(exprOp.getLoc())
              << "required by expr result type at phase " << demandedPhase - 1;
          return failure();
        }
      }
      return processBlock(exprOp.getBody().front(), demandedPhase);
    }
    return success();
  }

  // Anchored CF (if/loop): demand propagates to parent block.
  if (isa<IfOp, LoopOp>(parentOp))
    // REVIEW: if this returns failure, we may want to add a remark similar to
    // what we do in constrainValue
    return constrainBlock(*parentOp->getBlock(), demandedPhase);

  return success();
}

//===----------------------------------------------------------------------===//
// constrainRegionResult — Per-Result Constraint Through Yield/Break
//
// Pushes a phase constraint onto a specific result of a region-bearing op.
// The constraint propagates through the yield (for expr/if) or break ops
// (for loop) to the corresponding operand. Enables sparse, per-result
// constraints.
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::constrainRegionResult(Operation *regionOp,
                                                   unsigned resultIdx,
                                                   int16_t latest) {
  SaveAndRestore guard(depth, depth + 2);

  // Initialize result constraints if needed.
  auto &constraints = resultConstraints[regionOp];
  if (constraints.empty())
    constraints.assign(regionOp->getNumResults(), kUnconstrained);
  assert(resultIdx < constraints.size());

  // Only tighten: skip if already at a tighter-or-equal constraint.
  if (constraints[resultIdx] != kUnconstrained &&
      constraints[resultIdx] <= latest) {
    LLVM_DEBUG(dbgs() << INDENT << "constrainRegionResult(result " << resultIdx
                      << ", <=" << latest
                      << "): already <=" << constraints[resultIdx] << "\n");
    return success();
  }

  LLVM_DEBUG({
    dbgs() << INDENT << "constrainRegionResult(result " << resultIdx
           << ", <=" << latest << "): ";
    regionOp->print(dbgs(), OpPrintingFlags().skipRegions());
    dbgs() << "\n";
  });

  constraints[resultIdx] = latest;

  // REVIEW: This feels waaaay too complicated. If a result of an expr/if/loop
  // is constrained with `constrainValue` and then `constrainRegionResult`,
  // shouldn't we just look up the relevant terminators for this op and simply
  // call `constrainValue` on their operands and bail out on the first sign of
  // trouble? The fact that this handles floating exprs separately feels very
  // wrong. Any interaction with the parent op being anchored to a block or not
  // should not be relevant for the yield/continue/break terminators, which are
  // just conduits. They just forward to the operands, that's it. And since they
  // just forward, they also shouldn't have to create their own leaf errors --
  // they should just forward whatever `constrainValue` on the corresponding
  // operand returned.

  // Push constraint through yields/breaks.
  auto pushToYield = [&](Region &region) -> LogicalResult {
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
            LLVM_DEBUG(dbgs() << INDENT << "  floating expr => phase " << latest
                              << "\n");
            opPhases[regionOp] = latest;
            // Don't set actualPhase for results here — they are transparent
            // conduits through the yield. actualPhase will be set when
            // constrainRegionResult resolves each result's yield operand.
            for (auto typeVal : exprOp.getResultTypes()) {
              if (failed(constrainValue(typeVal, latest - 1))) {
                emitRemark(exprOp.getLoc())
                    << "required by expr result type at phase " << latest - 1;
                return failure();
              }
            }
            if (failed(processBlock(exprOp.getBody().front(), latest)))
              return failure();
          }
        }

        auto valIt = actualPhase.find(value);
        if (valIt == actualPhase.end()) {
          if (failed(constrainValue(value, latest))) {
            emitRemark(yieldOp.getLoc()) << "required by yield operand";
            return failure();
          }
          valIt = actualPhase.find(value);
        } else if (valIt->second != INT16_MIN && valIt->second > latest) {
          emitError(value.getLoc())
              << "value at phase " << valIt->second
              << " cannot satisfy requirement for phase " << latest;
          emitRemark(yieldOp.getLoc()) << "required by yield operand";
          return failure();
        }
        // Propagate actual phase to parent result.
        if (valIt != actualPhase.end())
          actualPhase[regionOp->getResult(resultIdx)] = valIt->second;
      }
      // Type operand for this result.
      auto typeOfValues = yieldOp.getTypeOfValues();
      if (resultIdx < typeOfValues.size()) {
        if (failed(constrainValue(typeOfValues[resultIdx], latest - 1))) {
          emitRemark(yieldOp.getLoc()) << "required by yield type operand";
          return failure();
        }
      }
    }
    return success();
  };

  if (auto exprOp = dyn_cast<ExprOp>(regionOp)) {
    if (failed(pushToYield(exprOp.getBody())))
      return failure();
  } else if (auto ifOp = dyn_cast<IfOp>(regionOp)) {
    if (failed(pushToYield(ifOp.getThenRegion())))
      return failure();
    if (!ifOp.getElseRegion().empty())
      if (failed(pushToYield(ifOp.getElseRegion())))
        return failure();
  } else if (auto loopOp = dyn_cast<LoopOp>(regionOp)) {
    // REVIEW: Same comment as above: this is all way too complicated.
    // Constraints on an op result should just propagate to the corresponding
    // terminator operands. This means we look up all the yields (or breaks for
    // loops), iterate over them, and constrain _only the operand that
    // corresponds to the constrained result_. That's it. Nothing else. No error
    // printing, no pahse inspection, no weird stuff. If not doing this leads to
    // issues in the algorithm or test failures, let's discuss. That's an
    // indicator that we're missing something in the algorithm's design.

    // Loop results come from break ops, not yields.
    auto it = loopBreaks.find(loopOp);
    if (it != loopBreaks.end()) {
      for (auto breakOp : it->second) {
        auto values = breakOp.getValues();
        if (resultIdx < values.size()) {
          Value value = values[resultIdx];
          auto valIt = actualPhase.find(value);
          if (valIt == actualPhase.end()) {
            if (failed(constrainValue(value, latest))) {
              emitRemark(breakOp.getLoc()) << "required by break operand";
              return failure();
            }
            valIt = actualPhase.find(value);
          } else if (valIt->second != INT16_MIN && valIt->second > latest) {
            emitError(value.getLoc())
                << "value at phase " << valIt->second
                << " cannot satisfy requirement for phase " << latest;
            emitRemark(breakOp.getLoc()) << "required by break operand";
            return failure();
          }
          // Propagate actual phase to loop result.
          if (valIt != actualPhase.end())
            actualPhase[regionOp->getResult(resultIdx)] = valIt->second;
        }
        // Type operands for break.
        auto typeOfValues = breakOp.getTypeOfValues();
        if (resultIdx < typeOfValues.size()) {
          if (failed(constrainValue(typeOfValues[resultIdx], latest - 1))) {
            emitRemark(breakOp.getLoc()) << "required by break type operand";
            return failure();
          }
        }
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// processBlock — Push Block Phase Down to Ops
//
// Walks ops in block order, calling processOp for non-terminators. Then sets
// the terminator phase and validates phase equalities (return at 0, break/
// continue at loop phase).
// REVIEW: make this a doc comment on the function
//===----------------------------------------------------------------------===//

LogicalResult PhaseAnalysis::processBlock(Block &block, int16_t blockPhase) {
  SaveAndRestore guard(depth, depth + 2);
  LLVM_DEBUG(dbgs() << INDENT << "processBlock(phase " << blockPhase << ")\n");

  // Process non-terminator ops.
  for (auto &op : block.without_terminator())
    if (failed(processOp(&op, blockPhase)))
      return failure();

  // Process the terminator.
  auto *terminator = block.getTerminator();
  opPhases[terminator] = blockPhase;

  // REVIEW: Again, why are regions so complicated? Yields and breaks are only
  // conduits for values and simply forward constraints (and return resolved
  // actual phases). This `processBlock` function should not have to have to do
  // any kind of `constrainValue` calls (that is done in constrainRegionResult).
  // That's also where any of these remarks should be printed. Processing here
  // should not do anything. Maybe just call `processOp` also on the terminator,
  // and have the terminators inherit the parent block phase in `processOp`.
  // That `processOp` is a great location to check that the terminators are in
  // the correct phases, e.g. the "return from phase shifted bla" or "break from
  // phase shifted bla" should be emitted there. At the end of the day, this
  // function should just iterate over the ops and call `processOp` with the
  // block phase.

  if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
    auto *parent = yieldOp->getParentOp();
    auto it = resultConstraints.find(parent);
    if (it != resultConstraints.end()) {
      auto &constraints = it->second;
      // Push constraints for results that have already been constrained.
      for (auto [i, value] : llvm::enumerate(yieldOp.getValues())) {
        if (i >= constraints.size() || constraints[i] == kUnconstrained)
          continue;
        if (failed(constrainValue(value, constraints[i]))) {
          emitRemark(yieldOp.getLoc()) << "required by yield operand";
          return failure();
        }
        // Propagate actual phase to parent result.
        auto valIt = actualPhase.find(value);
        if (valIt != actualPhase.end() && parent->getNumResults() > i)
          actualPhase[parent->getResult(i)] = valIt->second;
      }
      for (auto [i, typeVal] : llvm::enumerate(yieldOp.getTypeOfValues())) {
        if (i >= constraints.size() || constraints[i] == kUnconstrained)
          continue;
        if (failed(constrainValue(typeVal, constraints[i] - 1))) {
          emitRemark(yieldOp.getLoc()) << "required by yield type operand";
          return failure();
        }
      }
    }

  } else if (isa<SignatureOp>(terminator)) {
    // Phase validation and operand constraints handled in Step 4 of run().

    // REVIEW: `uir.signature` terminators must undergo the exact same phase
    // shift check as returns. These are identical to returns in all but their
    // operands.
  } else if (auto returnOp = dyn_cast<ReturnOp>(terminator)) {
    int16_t funcBodyPhase = 0;
    if (blockPhase != funcBodyPhase) {
      emitError(returnOp.getLoc())
          << "return from a phase-shifted block is not allowed";
      return failure();
    }
    // Operand constraints handled in Step 4 of run().

  } else if (auto breakOp = dyn_cast<BreakOp>(terminator)) {
    // REVIEW: move this check to processOp
    auto loopIt = breakToLoop.find(breakOp);
    assert(loopIt != breakToLoop.end() && "break not in pre-collected map");
    int16_t loopPhase = getPhase(loopIt->second);
    if (blockPhase != loopPhase) {
      emitError(breakOp.getLoc())
          << "break from a phase-shifted block is not allowed";
      return failure();
    }
    // Break operand constraints are handled via constrainRegionResult.

  } else if (auto continueOp = dyn_cast<ContinueOp>(terminator)) {
    // REVIEW: move this check to processOp
    auto loopIt = continueToLoop.find(continueOp);
    assert(loopIt != continueToLoop.end() &&
           "continue not in pre-collected map");
    int16_t loopPhase = getPhase(loopIt->second);
    if (blockPhase != loopPhase) {
      emitError(continueOp.getLoc())
          << "continue from a phase-shifted block is not allowed";
      return failure();
    }

  } else if (isa<UnreachableOp>(terminator)) {
    // Nothing to check.
  } else {
    llvm_unreachable("unexpected terminator in UIR block");
  }

  return success();
}

//===----------------------------------------------------------------------===//
// processOp — Push Block Phase to a Single Op
//
// Called by processBlock for each non-terminator. Sets the op's phase and
// pushes constraints to operands and regions. Skips floating exprs, pure ops,
// and constants (demand-driven only).
// REVIEW: make this a doc comment on the function
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
    // Don't set actualPhase for results — they are transparent conduits
    // through the yield. constrainRegionResult sets them.

    // Push result type constraints.
    for (auto typeVal : exprOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, exprPhase - 1))) {
        emitRemark(exprOp.getLoc())
            << "required by expr result type at phase " << exprPhase - 1;
        return failure();
      }
    }

    // Result constraints start unconstrained, same as if/loop. Results are
    // transparent conduits through the yield — consumers pull via
    // constrainRegionResult.
    //
    // REVIEW: Why do we even need to bother with these result constraints? If
    // region results are truly just passed through via terminators, Why do we
    // need a resultConstraints map at all? The constraints are actively pushed
    // down by the DFS in constrainValue and constrainRegionResult; no need to
    // keep that stuff around. If we need the exact phase assigned to a region
    // result for the sake of tightening checks, we can always just check
    // `actualPhase.at(regionOp.getResult(...))`. One thing I'm missing here is
    // an `actualPhase` assignment after the processBlock call.
    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(exprOp.getNumResults(), kUnconstrained);

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
    // Don't set actualPhase for results — they are transparent conduits
    // through the yield. constrainRegionResult sets them.

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

    // REVIEW: Same as above. Why resultConstraints? Why no actualPhase
    // assignment for the results after processBlock? We can just traverse all
    // the terminators using the data structures we've built and collect their
    // operand actualPhases and find the latest here. After the processBlock
    // we're on the upward path in the DFS, which is precisely where we want to
    // go and collect that information.
    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(ifOp.getNumResults(), kUnconstrained);

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
    // Don't set actualPhase for results — they are transparent conduits
    // through the break. constrainRegionResult sets them.

    for (auto typeVal : loopOp.getResultTypes()) {
      if (failed(constrainValue(typeVal, blockPhase - 1))) {
        emitRemark(loopOp.getLoc())
            << "required by loop result type at phase " << blockPhase - 1;
        return failure();
      }
    }

    // REVIEW: Same as above.

    auto &constraints = resultConstraints[op];
    if (constraints.empty())
      constraints.assign(loopOp.getNumResults(), kUnconstrained);

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

    // Constrain arguments..
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
