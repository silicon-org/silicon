//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Analysis/PhaseAnalysis.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "phase-analysis"

void PhaseAnalysis::analyze() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                          << "\n");

  // Pre-populate valuePhases for all body block arguments.
  auto bodyArgs = funcOp.getBody().getArguments();
  for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
    int16_t argPhase = static_cast<int16_t>(phase);
    valuePhases[bodyArgs[idx]] = argPhase;
    LLVM_DEBUG(if (argPhase != 0) llvm::dbgs()
               << "- Arg " << idx << " has phase " << argPhase << "\n");
  }

  // Seed the function itself at phase 0.
  opPhases.insert({funcOp, 0});

  // Walk the body in PreOrder, computing earliest available phases.
  funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto *parentOp = op->getParentOp();
    int16_t parentPhase = opPhases.at(parentOp);

    // Top-level ops (parent is the unified func) can float to any phase;
    // nested ops are floored at their parent's phase.
    bool isTopLevel = isa<UnifiedFuncOp>(parentOp);
    int16_t floor = isTopLevel ? INT16_MIN : parentPhase;

    // ExprOps with a non-zero phaseShift shift relative to their parent.
    int16_t phase;
    if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && exprOp.getPhaseShift()) {
      phase = parentPhase + exprOp.getPhaseShift();
    } else if (!isa<ExprOp, IfOp>(op) && mlir::isMemoryEffectFree(op)) {
      // Pure ops: phase is max of floor and all operand phases.
      phase = floor;
      for (auto operand : op->getOperands())
        phase = std::max(phase, getValuePhase(operand));
    } else {
      // Side-effecting ops (and ExprOps without phaseShift): inherit parent
      // phase.
      phase = parentPhase;
    }

    LLVM_DEBUG({
      llvm::dbgs() << "- Computed phase " << phase << " for: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases.insert({op, phase});
    for (auto result : op->getResults())
      valuePhases[result] = phase;
  });
}

int16_t PhaseAnalysis::getValuePhase(Value value) const {
  auto it = valuePhases.find(value);
  assert(it != valuePhases.end());
  return it->second;
}

LogicalResult PhaseAnalysis::checkCallArgPhases() {
  bool anyErrors = false;
  funcOp.getBody().walk([&](UnifiedCallOp callOp) {
    int16_t callOpPhase = opPhases.at(callOp);
    for (auto [arg, phase] :
         llvm::zip(callOp.getArguments(), callOp.getArgPhases())) {
      int16_t required = callOpPhase + static_cast<int16_t>(phase);
      int16_t available = getValuePhase(arg);
      if (available > required) {
        emitError(callOp.getLoc())
            << "call argument requires phase " << required
            << " but value is only available at phase " << available;
        anyErrors = true;
      }
    }
  });
  return anyErrors ? failure() : success();
}

//===----------------------------------------------------------------------===//
// Phase Back-Propagation
//
// After the forward analysis, some values may be assigned a later phase than
// necessary. When a call requires an argument at an earlier phase, we attempt
// to "pull" the value (and its transitive dependencies) to the required phase.
// This succeeds when all operands and captured values are already available at
// the target phase (e.g., they are floating constants). Block arguments cannot
// be pulled since their phase is fixed by the function signature.
//===----------------------------------------------------------------------===//

void PhaseAnalysis::pullPhases() {
  funcOp.getBody().walk([&](UnifiedCallOp callOp) {
    int16_t callOpPhase = opPhases.at(callOp);
    for (auto [arg, phase] :
         llvm::zip(callOp.getArguments(), callOp.getArgPhases())) {
      int16_t required = callOpPhase + static_cast<int16_t>(phase);
      int16_t available = getValuePhase(arg);
      if (available > required)
        pullValueToPhase(arg, required);
    }
  });
}

bool PhaseAnalysis::pullValueToPhase(Value value, int16_t targetPhase) {
  int16_t current = getValuePhase(value);
  if (current <= targetPhase)
    return true;

  // Block arguments have fixed phases; cannot pull.
  if (isa<BlockArgument>(value))
    return false;

  auto *defOp = value.getDefiningOp();
  assert(defOp);

  // Check feasibility: all direct operands must be pullable.
  for (auto operand : defOp->getOperands()) {
    if (getValuePhase(operand) > targetPhase &&
        !pullValueToPhase(operand, targetPhase))
      return false;
  }

  // For region-bearing ops (ExprOp, IfOp), check that all values captured
  // from the enclosing scope are available at the target phase.
  for (auto &region : defOp->getRegions()) {
    for (auto &block : region) {
      for (auto &nestedOp : block) {
        for (auto operand : nestedOp.getOperands()) {
          // Only check operands defined outside this op's regions.
          auto *operandParent = operand.getParentRegion();
          bool isCapture = true;
          for (auto &r : defOp->getRegions()) {
            if (r.isAncestor(operandParent)) {
              isCapture = false;
              break;
            }
          }
          if (isCapture) {
            if (getValuePhase(operand) > targetPhase &&
                !pullValueToPhase(operand, targetPhase))
              return false;
          }
        }
      }
    }
  }

  // All checks passed — update phases.
  LLVM_DEBUG({
    llvm::dbgs() << "- Pulling to phase " << targetPhase << ": ";
    defOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
  });
  opPhases[defOp] = targetPhase;
  for (auto result : defOp->getResults())
    valuePhases[result] = targetPhase;

  // Re-compute phases for nested regions with the new floor. This updates
  // nested op phases (including unified_calls) to reflect the pull, so
  // opPhase + attrPhase correctly gives absolute caller-frame phases.
  for (auto &region : defOp->getRegions())
    recomputeRegionPhases(region, targetPhase);

  return true;
}

void PhaseAnalysis::refreshPhases() {
  funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto *parentOp = op->getParentOp();
    int16_t parentPhase = opPhases.at(parentOp);
    bool isTopLevel = isa<UnifiedFuncOp>(parentOp);
    int16_t floor = isTopLevel ? INT16_MIN : parentPhase;

    int16_t phase;
    if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && exprOp.getPhaseShift()) {
      phase = parentPhase + exprOp.getPhaseShift();
    } else if (!isa<ExprOp, IfOp>(op) && mlir::isMemoryEffectFree(op)) {
      phase = floor;
      for (auto operand : op->getOperands())
        phase = std::max(phase, getValuePhase(operand));
    } else {
      phase = parentPhase;
    }

    // Keep the earlier of existing (possibly pulled) and recomputed phase.
    auto &existing = opPhases[op];
    phase = std::min(existing, phase);
    existing = phase;
    for (auto result : op->getResults())
      valuePhases[result] = phase;
  });
}

void PhaseAnalysis::recomputeRegionPhases(Region &region, int16_t floor) {
  region.walk<WalkOrder::PreOrder>([&](Operation *op) {
    int16_t parentPhase = opPhases.at(op->getParentOp());

    // ExprOps with a non-zero phaseShift shift relative to their parent.
    int16_t phase;
    if (auto exprOp = dyn_cast<ExprOp>(op); exprOp && exprOp.getPhaseShift()) {
      phase = parentPhase + exprOp.getPhaseShift();
    } else if (!isa<ExprOp, IfOp>(op) && mlir::isMemoryEffectFree(op)) {
      phase = floor;
      for (auto operand : op->getOperands())
        phase = std::max(phase, getValuePhase(operand));
    } else {
      phase = parentPhase;
    }

    opPhases[op] = phase;
    for (auto result : op->getResults())
      valuePhases[result] = phase;
  });
}
