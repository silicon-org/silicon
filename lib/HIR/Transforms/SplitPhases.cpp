//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

#define DEBUG_TYPE "split-phases"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_SPLITPHASESPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

namespace {
/// Compute the earliest available phase for each op and value in a unified
/// function. This uses a forward PreOrder walk: each op's phase is determined
/// by its parent's phase and its operands' phases (for pure ops), or just the
/// parent's phase (for side-effecting ops). ExprOps with a `const` attribute
/// shift relative to their parent.
///
/// Constants and other pure ops with no operands get INT16_MIN, meaning they
/// float to whatever phase needs them. This is clipped to minPhase during
/// splitting.
struct PhaseAnalysis {
  PhaseAnalysis(UnifiedFuncOp funcOp) : funcOp(funcOp) {}
  void analyze();
  void pullPhases();
  void refreshPhases();
  int16_t getValuePhase(Value value) const;
  LogicalResult checkCallArgPhases();

  UnifiedFuncOp funcOp;
  DenseMap<Operation *, int16_t> opPhases;

  /// Phases for all values: body block args and op results.
  DenseMap<Value, int16_t> valuePhases;

private:
  /// Try to pull a value (and its transitive dependencies) to an earlier phase.
  /// Returns true if the pull succeeded.
  bool pullValueToPhase(Value value, int16_t targetPhase);

  /// Re-compute phases for all ops inside a region, using `floor` as the
  /// minimum phase. This mirrors the forward analysis logic but with an updated
  /// floor.
  void recomputeRegionPhases(Region &region, int16_t floor);
};
} // namespace

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

    // ExprOps with a `const` attribute shift relative to their parent.
    IntegerAttr constAttr;
    int16_t phase;
    if (isa<ExprOp>(op) &&
        (constAttr = op->getAttrOfType<IntegerAttr>("const"))) {
      phase = parentPhase + constAttr.getInt();
    } else if (!isa<ExprOp, IfOp>(op) && mlir::isMemoryEffectFree(op)) {
      // Pure ops: phase is max of floor and all operand phases.
      phase = floor;
      for (auto operand : op->getOperands())
        phase = std::max(phase, getValuePhase(operand));
    } else {
      // Side-effecting ops (and ExprOps without const): inherit parent phase.
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

/// Look up the earliest phase at which a value is available.
int16_t PhaseAnalysis::getValuePhase(Value value) const {
  auto it = valuePhases.find(value);
  assert(it != valuePhases.end());
  return it->second;
}

/// Check that unified_call arguments are available at their required phases.
LogicalResult PhaseAnalysis::checkCallArgPhases() {
  bool anyErrors = false;
  funcOp.getBody().walk([&](UnifiedCallOp callOp) {
    for (auto [arg, phase] :
         llvm::zip(callOp.getArguments(), callOp.getArgPhases())) {
      int16_t required = static_cast<int16_t>(phase);
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
    for (auto [arg, phase] :
         llvm::zip(callOp.getArguments(), callOp.getArgPhases())) {
      int16_t required = static_cast<int16_t>(phase);
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

  // Re-compute phases for nested regions with the new floor.
  for (auto &region : defOp->getRegions())
    recomputeRegionPhases(region, targetPhase);

  // Shift arg/result phase attributes on any unified_calls inside the pulled
  // op's regions. The decomposition code uses these attributes to place
  // per-phase calls; without shifting, the decomposed calls would be placed at
  // the original absolute phases, which no longer match the containing op's
  // phase.
  int16_t delta = targetPhase - current;
  if (delta != 0) {
    for (auto &region : defOp->getRegions()) {
      region.walk([&](UnifiedCallOp callOp) {
        auto shiftPhases = [&](ArrayRef<int32_t> phases) {
          SmallVector<int32_t> shifted;
          for (auto p : phases)
            shifted.push_back(p + delta);
          return shifted;
        };
        callOp.setArgPhasesAttr(DenseI32ArrayAttr::get(
            callOp.getContext(), shiftPhases(callOp.getArgPhases())));
        callOp.setResultPhasesAttr(DenseI32ArrayAttr::get(
            callOp.getContext(), shiftPhases(callOp.getResultPhases())));
      });
    }
  }

  return true;
}

/// Re-run forward phase computation to propagate pulled phases to users. After
/// pulling a value to an earlier phase, pure ops that depend on it (like
/// `type_of` or `unify`) may also need earlier phases. This pass takes the min
/// of the existing and recomputed phase, preserving any pull-induced phases
/// while propagating their effects forward.
void PhaseAnalysis::refreshPhases() {
  funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    auto *parentOp = op->getParentOp();
    int16_t parentPhase = opPhases.at(parentOp);
    bool isTopLevel = isa<UnifiedFuncOp>(parentOp);
    int16_t floor = isTopLevel ? INT16_MIN : parentPhase;

    IntegerAttr constAttr;
    int16_t phase;
    if (isa<ExprOp>(op) &&
        (constAttr = op->getAttrOfType<IntegerAttr>("const"))) {
      phase = parentPhase + constAttr.getInt();
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

    // ExprOps with a `const` attribute shift relative to their parent.
    IntegerAttr constAttr;
    int16_t phase;
    if (isa<ExprOp>(op) &&
        (constAttr = op->getAttrOfType<IntegerAttr>("const"))) {
      phase = parentPhase + constAttr.getInt();
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

namespace {
struct PhaseSplit {
  FuncOp funcOp;
  int16_t phase;
  SmallVector<Value> returnValues;
  SmallVector<Value> returnTypeOperands;
};

/// A group of consecutive phases. A group ends when an externally visible
/// phase is reached. Trailing internal phases after the last external phase
/// form their own group.
struct PhaseGroup {
  SmallVector<int16_t> phases;
  bool hasExternalPhase = false;
};

struct PhaseSplitter {
  PhaseSplitter(PhaseAnalysis &analysis, SymbolTable &symbolTable)
      : analysis(analysis), symbolTable(symbolTable), funcOp(analysis.funcOp) {}
  void run();

  PhaseAnalysis &analysis;
  SymbolTable &symbolTable;
  UnifiedFuncOp funcOp;
};
} // namespace

void PhaseSplitter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Splitting " << funcOp.getSymNameAttr() << "\n");

  // Determine the phase range, skipping INT16_MIN (floating constants).
  int16_t minPhase = 0, maxPhase = 0;
  for (auto &[op, phase] : analysis.opPhases) {
    if (phase == INT16_MIN)
      continue;
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  }
  for (auto &[value, phase] : analysis.valuePhases) {
    if (phase == INT16_MIN)
      continue;
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  }

  // Extend the phase range based on the function's declared arg/result phases.
  // These may lie outside the range of computed op/value phases (e.g., a `dyn`
  // return type shifts the result to a later phase).
  for (auto p : funcOp.getArgPhases()) {
    minPhase = std::min(minPhase, static_cast<int16_t>(p));
    maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
  }
  for (auto p : funcOp.getResultPhases()) {
    minPhase = std::min(minPhase, static_cast<int16_t>(p));
    maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
  }

  // Extend the phase range based on UnifiedCallOps in the body. The call's
  // arg/result phases are absolute in the caller's frame, so the caller needs
  // split functions covering those phases.
  funcOp.getBody().walk([&](UnifiedCallOp callOp) {
    for (auto p : callOp.getArgPhases()) {
      minPhase = std::min(minPhase, static_cast<int16_t>(p));
      maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
    }
    for (auto p : callOp.getResultPhases()) {
      minPhase = std::min(minPhase, static_cast<int16_t>(p));
      maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "- Phases range [" << minPhase << ", " << maxPhase
                          << "]\n");

  //===--------------------------------------------------------------------===//
  // Compute External Phases and Group Phases
  //
  // External phases are those with caller-provided arguments or caller-visible
  // results. We walk minPhase..maxPhase, accumulating phases into groups. A
  // group ends when an externally visible phase is reached. Trailing internal
  // phases form their own group.
  //===--------------------------------------------------------------------===//

  DenseSet<int16_t> externalPhases;
  for (auto phase : funcOp.getArgPhases())
    externalPhases.insert(static_cast<int16_t>(phase));
  for (auto phase : funcOp.getResultPhases())
    externalPhases.insert(static_cast<int16_t>(phase));

  SmallVector<PhaseGroup> groups;
  {
    PhaseGroup current;
    for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
      current.phases.push_back(phase);
      if (externalPhases.contains(phase)) {
        current.hasExternalPhase = true;
        groups.push_back(std::move(current));
        current = {};
      }
    }
    if (!current.phases.empty())
      groups.push_back(std::move(current));
  }

  LLVM_DEBUG({
    for (auto [idx, group] : llvm::enumerate(groups)) {
      llvm::dbgs() << "- Group " << idx << ": [";
      llvm::interleaveComma(group.phases, llvm::dbgs());
      llvm::dbgs() << "]" << (group.hasExternalPhase ? " (external)" : "")
                   << "\n";
    }
  });

  //===--------------------------------------------------------------------===//
  // Create Per-Phase Functions
  //
  // Name per-phase functions using group-based convention:
  // - Standalone group (1 phase): @name.<groupIdx>
  // - Multi-phase group: @name.<groupIdx><letter> (a, b, c, ...)
  // We iterate in reverse so earlier phases appear first in the module.
  //===--------------------------------------------------------------------===//

  OpBuilder builder(funcOp);
  auto privateAttr = builder.getStringAttr("private");
  SmallVector<PhaseSplit> splits(maxPhase - minPhase + 1);
  for (int groupIdx = groups.size() - 1; groupIdx >= 0; --groupIdx) {
    auto &group = groups[groupIdx];
    bool isMulti = group.phases.size() > 1;
    for (int posIdx = group.phases.size() - 1; posIdx >= 0; --posIdx) {
      int16_t phase = group.phases[posIdx];
      StringAttr name;
      if (isMulti) {
        char letter = 'a' + posIdx;
        name = builder.getStringAttr(funcOp.getSymName() + "." +
                                     Twine(groupIdx) + Twine(letter));
      } else {
        name =
            builder.getStringAttr(funcOp.getSymName() + "." + Twine(groupIdx));
      }
      auto emptyArray = builder.getArrayAttr({});
      // Only the last phase of a module gets the isModule marker. Earlier
      // phases are plain compile-time evaluation; the final phase is the
      // actual hardware description.
      auto isModuleAttr =
          (phase == maxPhase) ? funcOp.getIsModuleAttr() : mlir::UnitAttr{};
      auto phaseFuncOp =
          FuncOp::create(builder, funcOp.getLoc(), name, privateAttr,
                         emptyArray, emptyArray, isModuleAttr);
      if (phase != 0)
        phaseFuncOp.getBody().emplaceBlock();
      symbolTable.insert(phaseFuncOp);
      builder.setInsertionPoint(phaseFuncOp);
      splits[phase - minPhase].funcOp = phaseFuncOp;
      splits[phase - minPhase].phase = phase;
    }
  }

  // Handle the return operation. Each return value and its type operand are
  // placed in the split matching its declared result phase. If a value is
  // defined in an earlier phase, cross-phase threading will create context
  // returns and block args to forward it. The return op itself is erased since
  // each phase will get its own.
  auto returnOp = funcOp.getReturnOp();
  auto resultPhases = funcOp.getResultPhases();
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t resultPhase = static_cast<int16_t>(resultPhases[idx]);
    splits[resultPhase - minPhase].returnValues.push_back(value);
  }
  for (auto [idx, type] : llvm::enumerate(returnOp.getTypeOfValues())) {
    int16_t resultPhase = static_cast<int16_t>(resultPhases[idx]);
    splits[resultPhase - minPhase].returnTypeOperands.push_back(type);
  }
  returnOp.erase();

  // Move all operations to phase 0 initially.
  splits[0 - minPhase].funcOp.getBody().takeBody(funcOp.getBody());

  // Track where each unified func arg ended up after shifted arg movement.
  // Shifted args (phase != 0) map to their new block arg in the target phase
  // function; phase-0 args map to the remaining block args in the phase-0
  // function.
  SmallVector<Value> unifiedArgValues(funcOp.getArgPhases().size());

  // Move shifted-phase body block arguments to their respective phase
  // functions. We create block args in each arg's target phase, replace all
  // uses, and let cross-phase value threading handle multi-hop forwarding.
  // This handles both const args (phase < 0) and dyn args (phase > 0).
  {
    auto &phase0Block = splits[0 - minPhase].funcOp.getBody().front();

    // Collect shifted args and their phases first to avoid iterator
    // invalidation.
    SmallVector<std::pair<unsigned, int16_t>> shiftedArgs;
    for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
      int16_t argPhase = static_cast<int16_t>(phase);
      if (argPhase != 0)
        shiftedArgs.push_back({idx, argPhase});
    }
    llvm::sort(shiftedArgs);

    // Step 1: Create a block arg in each shifted arg's target phase function.
    SmallVector<std::pair<unsigned, Value>> replacements;
    SmallVector<unsigned> argsToErase;
    for (auto [idx, argPhase] : shiftedArgs) {
      auto bodyArg = phase0Block.getArgument(idx);
      LLVM_DEBUG(llvm::dbgs() << "- Moving body arg " << idx << " to phase "
                              << argPhase << "\n");
      auto &targetBlock = splits[argPhase - minPhase].funcOp.getBody().front();
      Value ownArg =
          targetBlock.addArgument(bodyArg.getType(), bodyArg.getLoc());
      unifiedArgValues[idx] = ownArg;
      replacements.push_back({idx, ownArg});
      argsToErase.push_back(idx);
    }

    // Step 2: Replace all uses of old body block args with the target-phase
    // values. Cross-phase value threading will handle forwarding to other
    // phases as needed. Also update returnTypeOperands, which hold Value
    // handles that replaceAllUsesWith won't touch.
    for (auto &[idx, ownArg] : replacements) {
      auto bodyArg = phase0Block.getArgument(idx);
      bodyArg.replaceAllUsesWith(ownArg);
      for (auto &split : splits)
        for (auto &typeOp : split.returnTypeOperands)
          if (typeOp == bodyArg)
            typeOp = ownArg;
    }

    // Erase old body block args in reverse order.
    for (auto idx : llvm::reverse(argsToErase))
      phase0Block.eraseArgument(idx);

    // Fill in phase-0 args (those that remained in the phase-0 block).
    unsigned phase0ArgIdx = 0;
    for (unsigned i = 0; i < unifiedArgValues.size(); ++i) {
      if (unifiedArgValues[i]) // already filled (shifted arg)
        continue;
      unifiedArgValues[i] = phase0Block.getArgument(phase0ArgIdx++);
    }
  }

  // Decompose UnifiedCallOps into per-phase CallOps. Each per-phase call is
  // registered in the analysis at the appropriate phase, so the distribution
  // worklist moves it to the correct split function.
  {
    auto &phase0Body = splits[0 - minPhase].funcOp.getBody();
    SmallVector<UnifiedCallOp> unifiedCalls;
    phase0Body.walk([&](UnifiedCallOp op) { unifiedCalls.push_back(op); });

    for (auto callOp : unifiedCalls) {
      auto calleeName = callOp.getCallee();
      auto argPhases = callOp.getArgPhases();
      auto resultPhases = callOp.getResultPhases();

      // Compute the callee's phase range.
      int16_t calleeMinPhase = 0, calleeMaxPhase = 0;
      for (auto p : argPhases) {
        calleeMinPhase = std::min(calleeMinPhase, static_cast<int16_t>(p));
        calleeMaxPhase = std::max(calleeMaxPhase, static_cast<int16_t>(p));
      }
      for (auto p : resultPhases) {
        calleeMinPhase = std::min(calleeMinPhase, static_cast<int16_t>(p));
        calleeMaxPhase = std::max(calleeMaxPhase, static_cast<int16_t>(p));
      }

      // Look up the callee's split_func and build a phase-to-function mapping
      // by walking its entries. MultiphaseFuncOps expand to their
      // sub-functions.
      auto calleeSplitFunc = symbolTable.lookup<SplitFuncOp>(calleeName);
      if (!calleeSplitFunc)
        continue;

      SmallVector<std::pair<int16_t, FuncOp>> splitFuncs;
      {
        auto phaseNums = calleeSplitFunc.getPhaseNumbers();
        auto phaseFuncRefs = calleeSplitFunc.getPhaseFuncs();

        // Reconstruct the callee's full phase range from its split_func
        // entries. Each entry is either a standalone FuncOp (1 phase) or a
        // MultiphaseFuncOp (multiple phases). We walk them in order.
        int16_t phase = calleeMinPhase;
        for (auto [entryPhase, funcRef] : llvm::zip(phaseNums, phaseFuncRefs)) {
          auto name = cast<FlatSymbolRefAttr>(funcRef).getValue();
          if (auto mpFunc = symbolTable.lookup<MultiphaseFuncOp>(name)) {
            for (auto subRef : mpFunc.getPhaseFuncs()) {
              auto subName = cast<FlatSymbolRefAttr>(subRef).getValue();
              splitFuncs.push_back(
                  {phase++, symbolTable.lookup<FuncOp>(subName)});
            }
          } else {
            splitFuncs.push_back({phase++, symbolTable.lookup<FuncOp>(name)});
          }
        }
      }
      if (splitFuncs.empty())
        continue;

      OpBuilder callBuilder(callOp);
      auto loc = callOp.getLoc();

      // Update phases of existing type operands so they land in the correct
      // split function during distribution. Use std::min since a single value
      // may be shared across multiple phases (e.g., the same int_type used for
      // both a const and a runtime argument); the earliest phase ensures the
      // value is available everywhere it's needed, with value threading
      // handling forwarding to later phases.
      for (auto [type, phase] : llvm::zip(callOp.getTypeOfArgs(), argPhases)) {
        int16_t p = static_cast<int16_t>(phase);
        if (auto *defOp = type.getDefiningOp()) {
          auto &opPhase = analysis.opPhases[defOp];
          opPhase = std::min(opPhase, p);
          auto &valPhase = analysis.valuePhases[type];
          valPhase = std::min(valPhase, p);
        }
      }
      for (auto [type, phase] :
           llvm::zip(callOp.getTypeOfResults(), resultPhases)) {
        int16_t p = static_cast<int16_t>(phase);
        if (auto *defOp = type.getDefiningOp()) {
          auto &opPhase = analysis.opPhases[defOp];
          opPhase = std::min(opPhase, p);
          auto &valPhase = analysis.valuePhases[type];
          valPhase = std::min(valPhase, p);
        }
      }

      // Partition arguments and their type operands by phase.
      DenseMap<int16_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
      for (auto [arg, type, phase] : llvm::zip(
               callOp.getArguments(), callOp.getTypeOfArgs(), argPhases)) {
        phaseArgs[static_cast<int16_t>(phase)].push_back(arg);
        phaseTypeOfArgs[static_cast<int16_t>(phase)].push_back(type);
      }

      // Determine the phase that produces the unified_call's actual results.
      // This is the maximum result phase, which may differ from
      // calleeMaxPhase when dyn args extend the callee beyond its results.
      int16_t maxResultPhase = INT16_MIN;
      for (auto p : resultPhases)
        maxResultPhase = std::max(maxResultPhase, static_cast<int16_t>(p));

      // Collect which result indices belong to which phase.
      DenseSet<int16_t> resultPhaseSet;
      for (auto p : resultPhases)
        resultPhaseSet.insert(static_cast<int16_t>(p));

      // Chain calls from earliest phase to latest. Each call gets its own
      // phase's arguments plus all results from the previous phase's call.
      // The call at the result phase produces the unified_call's original
      // result types; all other phases use opaque types derived from the
      // split function's return op.
      SmallVector<Value> prevResults;
      SmallVector<Value> unifiedCallReplacements;
      for (auto &[phase, splitFunc] : splitFuncs) {
        SmallVector<Value> callArgs(phaseArgs[phase]);
        SmallVector<Value> callTypeOfArgs(phaseTypeOfArgs[phase]);

        // Thread results from the previous phase.
        for (auto result : prevResults) {
          callArgs.push_back(result);
          auto opaqueType = OpaqueTypeOp::create(callBuilder, loc);
          analysis.opPhases[opaqueType] = phase;
          analysis.valuePhases[opaqueType.getResult()] = phase;
          callTypeOfArgs.push_back(opaqueType.getResult());
        }

        // The result phase uses the original unified_call's result types.
        // All other phases determine their result count from the split
        // function's return op.
        bool isResultPhase = (phase == maxResultPhase);
        SmallVector<Value> callTypeOfResults;
        SmallVector<Type> resultTypes;
        if (isResultPhase) {
          callTypeOfResults.append(callOp.getTypeOfResults().begin(),
                                   callOp.getTypeOfResults().end());
          resultTypes.append(callOp.getResultTypes().begin(),
                             callOp.getResultTypes().end());
        }

        // For non-result phases (and result phases that also forward context
        // to later phases), add opaque result types for the split function's
        // return values beyond the unified_call's own results.
        if (!isResultPhase || phase != calleeMaxPhase) {
          auto retOp =
              cast<ReturnOp>(splitFunc.getBody().front().getTerminator());
          unsigned startIdx = isResultPhase ? callOp.getNumResults() : 0;
          for (unsigned i = startIdx; i < retOp.getValues().size(); ++i) {
            auto opaqueType = OpaqueTypeOp::create(callBuilder, loc);
            analysis.opPhases[opaqueType] = phase;
            analysis.valuePhases[opaqueType.getResult()] = phase;
            callTypeOfResults.push_back(opaqueType.getResult());
            resultTypes.push_back(retOp.getValues()[i].getType());
          }
        }

        auto call =
            CallOp::create(callBuilder, loc, resultTypes,
                           callBuilder.getStringAttr(splitFunc.getSymName()),
                           callArgs, callTypeOfArgs, callTypeOfResults);
        analysis.opPhases[call] = phase;
        for (auto result : call.getResults())
          analysis.valuePhases[result] = phase;
        prevResults.assign(call.getResults().begin(), call.getResults().end());

        // Capture the results from the result phase as replacements for the
        // original unified_call.
        if (isResultPhase) {
          for (unsigned i = 0; i < callOp.getNumResults(); ++i)
            unifiedCallReplacements.push_back(call.getResult(i));
        }
      }

      // Update any return values that reference the old unified call's results.
      // These are not IR uses (the unified_return was already erased), so
      // replaceAllUsesWith won't touch them. Return values may be in any split
      // (depending on result phases), so check all of them.
      for (auto &split : splits)
        for (auto &rv : split.returnValues)
          for (auto [oldResult, newResult] :
               llvm::zip(callOp.getResults(), unifiedCallReplacements))
            if (rv == oldResult)
              rv = newResult;

      callOp.replaceAllUsesWith(unifiedCallReplacements);
      callOp.erase();
    }
  }

  // Set up worklist to move operations into their respective phase functions.
  // This is done after unified call decomposition so the worklist sees the new
  // per-phase call ops.
  SmallVector<std::tuple<int16_t, Block::iterator, Block::iterator>> worklist;
  for (auto &block : llvm::reverse(splits[0 - minPhase].funcOp.getBody()))
    worklist.push_back({0, block.begin(), block.end()});

  // Move operations into their respective phase functions.
  while (!worklist.empty()) {
    auto &[phase, opIt, opEnd] = worklist.back();
    if (opIt == opEnd) {
      worklist.pop_back();
      continue;
    }

    // If the current operation has been assigned to a different phase, move it
    // to the appropriate phase function.
    Operation *op = &*opIt;
    ++opIt;
    int16_t opPhase = analysis.opPhases.at(op);
    if (opPhase == INT16_MIN)
      opPhase = minPhase;
    if (opPhase != phase) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Moving to phase " << opPhase << ": "
                     << op->getName();
        if (op->getNumResults() > 0)
          llvm::dbgs() << " (" << op->getNumResults() << " results)";
        llvm::dbgs() << "\n";
      });
      auto &split = splits[opPhase - minPhase];
      auto *block = &split.funcOp.getBody().back();
      op->moveBefore(block, block->end());
    }

    // If this is an `ExprOp` or `IfOp`, push its nested operations onto the
    // worklist, since those might move to a different phase as well.
    if (auto exprOp = dyn_cast<ExprOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "- Descending into " << op->getName() << "\n");
      for (auto &block : llvm::reverse(exprOp.getBody()))
        worklist.push_back({opPhase, block.begin(), block.end()});
    } else if (auto ifOp = dyn_cast<IfOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "- Descending into " << op->getName() << "\n");
      for (auto &block : llvm::reverse(ifOp.getThenRegion()))
        worklist.push_back({opPhase, block.begin(), block.end()});
      for (auto &block : llvm::reverse(ifOp.getElseRegion()))
        worklist.push_back({opPhase, block.begin(), block.end()});
    } else {
      // TODO: Otherwise ensure that any nested operations have been assigned
      // the same phase.
    }
  }

  // Record the number of "own" block args and return values for each phase
  // function before cross-phase threading adds context values. These counts
  // are used afterward to bundle context values into opaque packs.
  SmallVector<unsigned> numOwnArgs(maxPhase - minPhase + 1);
  SmallVector<unsigned> numOwnReturns(maxPhase - minPhase + 1);
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    numOwnArgs[phase - minPhase] =
        splits[phase - minPhase].funcOp.getBody().front().getNumArguments();
    numOwnReturns[phase - minPhase] =
        splits[phase - minPhase].returnValues.size();
  }

  //===--------------------------------------------------------------------===//
  // Insert coerce_type Ops on Own Block Args
  //
  // For each phase function, insert an `hir.coerce_type` on every own block
  // arg, annotating it with its type from the unified func's signature. This
  // gives downstream passes (HIR-to-MIR) the type information needed to
  // materialize concrete MLIR types. Cross-phase type operands are resolved
  // later by the value threading loop below.
  //===--------------------------------------------------------------------===//

  {
    auto sigOp = funcOp.getSignatureOp();
    auto typeOfArgs = sigOp.getTypeOfArgs();
    auto &sigBlock = funcOp.getSignature().front();

    // Build per-phase list of unified arg indices for own args.
    SmallVector<SmallVector<unsigned>> phaseOwnArgIndices(maxPhase - minPhase +
                                                          1);
    for (unsigned i = 0; i < funcOp.getArgPhases().size(); ++i) {
      int16_t phase = static_cast<int16_t>(funcOp.getArgPhases()[i]);
      phaseOwnArgIndices[phase - minPhase].push_back(i);
    }

    for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
      auto &split = splits[phase - minPhase];
      auto &block = split.funcOp.getBody().front();
      auto &ownIndices = phaseOwnArgIndices[phase - minPhase];
      if (ownIndices.empty())
        continue;

      // Build a per-phase mapping from signature values to body/split values.
      // Signature block args map to unifiedArgValues; cloned op results are
      // added as we go. A fresh mapping per phase ensures cloned ops are local.
      IRMapping sigToBody;
      for (auto [sigArg, bodyVal] :
           llvm::zip(sigBlock.getArguments(), unifiedArgValues))
        sigToBody.map(sigArg, bodyVal);

      OpBuilder insertBuilder(&block, block.begin());
      for (auto [localIdx, unifiedIdx] : llvm::enumerate(ownIndices)) {
        Value ownArg = block.getArgument(localIdx);
        Value sigTypeVal = typeOfArgs[unifiedIdx];

        // Resolve the signature type value to a body/split value.
        Value resolvedType;
        if (auto mapped = sigToBody.lookupOrNull(sigTypeVal)) {
          resolvedType = mapped;
        } else {
          auto *defOp = sigTypeVal.getDefiningOp();
          auto *cloned = insertBuilder.clone(*defOp, sigToBody);
          resolvedType =
              cloned->getResult(cast<OpResult>(sigTypeVal).getResultNumber());
        }

        auto coerceOp = CoerceTypeOp::create(insertBuilder, ownArg.getLoc(),
                                             ownArg, resolvedType);

        // Replace uses of the own arg within this function with the coerced
        // value. Exclude the coerce_type itself and scope to this function
        // (cross-phase refs to this block arg exist from const arg movement).
        ownArg.replaceUsesWithIf(coerceOp.getResult(), [&](OpOperand &use) {
          if (use.getOwner() == coerceOp)
            return false;
          Operation *parent = use.getOwner();
          while (parent && !isa<FuncOp>(parent))
            parent = parent->getParentOp();
          return parent == split.funcOp;
        });

        // Update returnValues entries that reference the own arg, since
        // replaceUsesWithIf only modifies OpOperands, not standalone vectors.
        for (auto &rv : split.returnValues)
          if (rv == ownArg)
            rv = coerceOp.getResult();
      }
    }
  }

  // Add return operations to all phase functions and plumb values from earlier
  // phases to later phases. We iterate in reverse execution order (most-runtime
  // first, most-const last) so that thread-through adds values to later splits
  // before their return ops are emitted.
  DenseMap<Value, Value> mapping;
  SmallDenseSet<Operation *, 4> closedFuncs;
  for (int16_t phase = maxPhase; phase >= minPhase; --phase) {
    auto &split = splits[phase - minPhase];
    closedFuncs.insert(split.funcOp);

    // Add a return operation to this split. Use preserved type operands for
    // "own" return values (from the original unified return), and fall back to
    // getOrCreateTypeOf for context values added later by cross-phase
    // threading.
    builder.setInsertionPointToEnd(&split.funcOp.getBody().back());
    SmallVector<Value> returnTypes;
    for (auto [idx, val] : llvm::enumerate(split.returnValues)) {
      if (idx < split.returnTypeOperands.size())
        returnTypes.push_back(split.returnTypeOperands[idx]);
      else
        returnTypes.push_back(getOrCreateTypeOf(builder, funcOp.getLoc(), val));
    }
    ReturnOp::create(builder, funcOp.getLoc(), split.returnValues, returnTypes);

    // Replace all uses of values from earlier phases with additional block
    // arguments. This will be replaced with constants through function
    // specialization.
    split.funcOp.walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        auto &value = mapping[operand.get()];
        if (!value) {
          // Get the `FuncOp` within which this value is defined.
          auto *parentOp = operand.get().getParentBlock()->getParentOp();
          while (!isa<FuncOp>(parentOp))
            parentOp = parentOp->getParentOp();

          // If the definition is inside the current split, simply use the value
          // as-is. Otherwise add a block argument.
          if (parentOp == split.funcOp) {
            value = operand.get();
          } else {
            // Do a quick sanity check that we are not using a value from a
            // later phase. This should be caught earlier and reported properly
            // to the user.
            if (closedFuncs.contains(parentOp)) {
              emitBug(op->getLoc()) << "op uses value from later phase";
              return;
            }

            // Pure ops with no operands (constants, type constructors) can be
            // cloned into the current split instead of threading through block
            // args. This keeps type constants like `hir.int_type` local to
            // each phase, which is needed by the HIR-to-MIR lowering.
            auto *defOp = operand.get().getDefiningOp();
            if (defOp && mlir::isMemoryEffectFree(defOp) &&
                defOp->getNumOperands() == 0) {
              LLVM_DEBUG({
                llvm::dbgs() << "- Cloning into phase " << phase << ": "
                             << defOp->getName() << "\n";
              });
              auto &block = split.funcOp.getBody().front();
              OpBuilder cloneBuilder(&block, block.begin());
              auto *cloned = cloneBuilder.clone(*defOp);
              value = cloned->getResult(
                  cast<OpResult>(operand.get()).getResultNumber());
            } else {
              // Add an argument to the current split function.
              LLVM_DEBUG({
                llvm::dbgs()
                    << "- Creating arg in phase " << phase << " for value from "
                    << operand.get().getDefiningOp()->getName() << "\n";
              });
              value = split.funcOp.getBody().front().addArgument(
                  operand.get().getType(), operand.get().getLoc());

              // Add the value as a result to the one-earlier phase function so
              // it flows into the current phase as a block argument.
              assert(phase > minPhase);
              splits[phase - 1 - minPhase].returnValues.push_back(
                  operand.get());
            }
          }
        }
        operand.set(value);
      }
    });
    mapping.clear();
  }

  //===--------------------------------------------------------------------===//
  // Bundle Context Values into Opaque Packs
  //
  // For each phase boundary, bundle the context values (added during
  // cross-phase threading) into a single `hir.opaque_pack` at the source
  // phase's return, and unpack them with `hir.opaque_unpack` at the target
  // phase's entry. This ensures each phase boundary passes a single opaque
  // bundle instead of individual values.
  //===--------------------------------------------------------------------===//

  auto anyType = AnyType::get(builder.getContext());
  for (int16_t phase = minPhase + 1; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &block = split.funcOp.getBody().front();
    unsigned ownArgs = numOwnArgs[phase - minPhase];
    unsigned totalArgs = block.getNumArguments();
    unsigned contextArgs = totalArgs - ownArgs;
    if (contextArgs == 0)
      continue;

    // In this phase: replace trailing context block args with a single opaque
    // arg followed by an opaque_unpack. The opaque context arg feeds directly
    // into the unpack without an intermediate coerce_type.
    auto opaqueArg = block.addArgument(anyType, funcOp.getLoc());
    OpBuilder unpackBuilder(&block, block.begin());
    auto unpackOp = OpaqueUnpackOp::create(
        unpackBuilder, funcOp.getLoc(), SmallVector<Type>(contextArgs, anyType),
        opaqueArg);
    for (unsigned i = 0; i < contextArgs; ++i)
      block.getArgument(ownArgs + i).replaceAllUsesWith(unpackOp.getResult(i));
    block.eraseArguments(ownArgs, contextArgs);

    // In the previous phase: replace trailing context return values with a
    // single opaque_pack.
    auto &prevSplit = splits[phase - 1 - minPhase];
    auto prevReturnOp =
        cast<ReturnOp>(prevSplit.funcOp.getBody().back().getTerminator());
    unsigned prevOwnReturns = numOwnReturns[phase - 1 - minPhase];
    unsigned prevTotalReturns = prevReturnOp.getValues().size();
    unsigned contextReturns = prevTotalReturns - prevOwnReturns;
    if (contextReturns == 0)
      continue;

    OpBuilder packBuilder(prevReturnOp);
    SmallVector<Value> contextValues(
        prevReturnOp.getValues().drop_front(prevOwnReturns));
    auto packOp =
        OpaquePackOp::create(packBuilder, funcOp.getLoc(), contextValues);

    SmallVector<Value> newReturnValues(
        prevReturnOp.getValues().take_front(prevOwnReturns));
    newReturnValues.push_back(packOp);
    SmallVector<Value> newReturnTypes(
        prevReturnOp.getTypeOfValues().take_front(prevOwnReturns));
    newReturnTypes.push_back(
        OpaqueTypeOp::create(packBuilder, funcOp.getLoc()).getResult());
    ReturnOp::create(packBuilder, funcOp.getLoc(), newReturnValues,
                     newReturnTypes);
    prevReturnOp.erase();
  }

  //===--------------------------------------------------------------------===//
  // Re-apply coerce_type Replacements
  //
  // The coerce_type insertion step above ran replaceUsesWithIf before the
  // opaque pack bundling step. Opaque packs created later reference the raw
  // block arg, leaving coerce_type dead. Re-apply the replacement so that
  // opaque_pack operands (and any other remaining raw-arg uses) go through
  // the coerced value.
  //===--------------------------------------------------------------------===//

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &block = split.funcOp.getBody().front();
    unsigned ownArgs = numOwnArgs[phase - minPhase];
    for (unsigned i = 0; i < ownArgs; ++i) {
      Value ownArg = block.getArgument(i);
      CoerceTypeOp coerceOp;
      for (auto *user : ownArg.getUsers()) {
        if (auto c = dyn_cast<CoerceTypeOp>(user)) {
          coerceOp = c;
          break;
        }
      }
      if (!coerceOp)
        continue;
      ownArg.replaceUsesWithIf(coerceOp.getResult(), [&](OpOperand &use) {
        return use.getOwner() != coerceOp;
      });
    }
  }

  //===--------------------------------------------------------------------===//
  // Set Argument and Result Names on Phase Functions
  //
  // Now that each phase function has its final block args and return values,
  // assign meaningful names. Own args get their original names from the
  // unified func; the opaque context arg (if any) gets named "ctx". The
  // external phase matching the result phases gets the unified func's result
  // names; context returns get "ctx".
  //===--------------------------------------------------------------------===//

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &block = split.funcOp.getBody().front();

    // Build argNames: original names for own args, "ctx" for the opaque arg.
    SmallVector<Attribute> phaseArgNames;
    for (auto [name, argPhase] :
         llvm::zip(funcOp.getArgNames(), funcOp.getArgPhases())) {
      if (static_cast<int16_t>(argPhase) == phase)
        phaseArgNames.push_back(name);
    }
    if (block.getNumArguments() > phaseArgNames.size())
      phaseArgNames.push_back(builder.getStringAttr("ctx"));
    split.funcOp.setArgNamesAttr(builder.getArrayAttr(phaseArgNames));

    // Build resultNames: the phase matching the result phases gets the unified
    // func's result names; context returns get "ctx".
    SmallVector<Attribute> phaseResultNames;
    if (auto retOp = split.funcOp.getReturnOp()) {
      unsigned ownReturns = numOwnReturns[phase - minPhase];
      // Check if this phase has user-visible results.
      for (auto [name, resultPhase] :
           llvm::zip(funcOp.getResultNames(), funcOp.getResultPhases())) {
        if (static_cast<int16_t>(resultPhase) == phase)
          phaseResultNames.push_back(name);
      }
      if (retOp.getValues().size() > ownReturns)
        phaseResultNames.push_back(builder.getStringAttr("ctx"));
    }
    split.funcOp.setResultNamesAttr(builder.getArrayAttr(phaseResultNames));
  }

  //===--------------------------------------------------------------------===//
  // Emit Structural Ops
  //
  // Build a split_func with one entry per group. Standalone groups (1 phase)
  // map directly to a FuncOp. Multi-phase groups get a multiphase_func
  // wrapping their sub-functions. The representative phase for each group in
  // the split_func is the last phase (the externally visible one, or the last
  // internal phase for trailing groups).
  //===--------------------------------------------------------------------===//

  auto sfLoc = funcOp.getLoc();
  auto sfName = funcOp.getSymNameAttr();
  auto sfArgNames = funcOp.getArgNames();
  auto sfResultNames = funcOp.getResultNames();
  SmallVector<int32_t> sfArgPhases(funcOp.getArgPhases());
  SmallVector<int32_t> sfResultPhases(funcOp.getResultPhases());

  // Create builder for structural ops, inserting after the last per-phase func.
  auto *lastPhaseFunc = splits[maxPhase - minPhase].funcOp.getOperation();
  OpBuilder sfBuilder(lastPhaseFunc->getBlock(),
                      std::next(lastPhaseFunc->getIterator()));
  auto *ctx = sfBuilder.getContext();

  // Build phase-to-function mapping arrays and collect multiphase_func info.
  SmallVector<int32_t> phaseNumbers;
  SmallVector<Attribute> phaseFuncAttrs;

  struct MultiphaseInfo {
    StringAttr name;
    SmallVector<Attribute> argNames;
    SmallVector<bool> argIsFirst;
    SmallVector<Attribute> resultNames;
    SmallVector<Attribute> subFuncAttrs;
  };
  SmallVector<MultiphaseInfo, 2> multiphaseInfos;

  for (auto [groupIdx, group] : llvm::enumerate(groups)) {
    int16_t repPhase = group.phases.back();
    phaseNumbers.push_back(repPhase);

    if (group.phases.size() == 1) {
      // Standalone group: split_func entry points directly to the func.
      phaseFuncAttrs.push_back(FlatSymbolRefAttr::get(
          ctx, splits[repPhase - minPhase].funcOp.getSymName()));
    } else {
      // Multi-phase group: create a multiphase_func.
      auto mpName =
          sfBuilder.getStringAttr(funcOp.getSymName() + "." + Twine(groupIdx));
      phaseFuncAttrs.push_back(FlatSymbolRefAttr::get(ctx, mpName));

      MultiphaseInfo mp;
      mp.name = mpName;

      // Collect sub-function references.
      for (auto phase : group.phases)
        mp.subFuncAttrs.push_back(FlatSymbolRefAttr::get(
            ctx, splits[phase - minPhase].funcOp.getSymName()));

      // "first" args: only opaque context from a prior group. The first phase
      // of the group gets a ctx arg only if it's not the very first phase.
      int16_t firstPhase = group.phases.front();
      if (firstPhase > minPhase) {
        mp.argNames.push_back(sfBuilder.getStringAttr("ctx"));
        mp.argIsFirst.push_back(true);
      }

      // "last" args: user-visible args at the group's external phase (if any).
      if (group.hasExternalPhase) {
        for (auto [name, argPhase] : llvm::zip(sfArgNames, sfArgPhases)) {
          if (static_cast<int16_t>(argPhase) == repPhase) {
            mp.argNames.push_back(name);
            mp.argIsFirst.push_back(false);
          }
        }
      }

      // Results: user results at the external phase + ctx if not last group.
      if (group.hasExternalPhase) {
        for (auto [name, resultPhase] :
             llvm::zip(sfResultNames, sfResultPhases)) {
          if (static_cast<int16_t>(resultPhase) == repPhase)
            mp.resultNames.push_back(name);
        }
      }
      if (static_cast<unsigned>(groupIdx) + 1 < groups.size())
        mp.resultNames.push_back(sfBuilder.getStringAttr("ctx"));

      multiphaseInfos.push_back(std::move(mp));
    }
  }

  // Create the split_func.
  auto splitFuncOp = SplitFuncOp::create(
      sfBuilder, sfLoc, sfName,
      /*sym_visibility=*/funcOp.getSymVisibilityAttr(), sfArgNames,
      sfBuilder.getDenseI32ArrayAttr(sfArgPhases), sfResultNames,
      sfBuilder.getDenseI32ArrayAttr(sfResultPhases),
      sfBuilder.getDenseI32ArrayAttr(phaseNumbers),
      sfBuilder.getArrayAttr(phaseFuncAttrs));

  // Move the unified func's signature region into the split_func and replace
  // the unified_signature terminator with a signature terminator.
  splitFuncOp.getSignature().takeBody(funcOp.getSignature());
  {
    auto &sigBlock = splitFuncOp.getSignature().back();
    auto unifiedSigOp = cast<UnifiedSignatureOp>(sigBlock.getTerminator());
    OpBuilder sigBuilder(unifiedSigOp);
    SignatureOp::create(sigBuilder, unifiedSigOp.getLoc(),
                        unifiedSigOp.getTypeOfArgs(),
                        unifiedSigOp.getTypeOfResults());
    unifiedSigOp.erase();
  }

  // Emit multiphase_funcs for multi-phase groups.
  for (auto &mp : multiphaseInfos) {
    auto mpOp = MultiphaseFuncOp::create(
        sfBuilder, sfLoc, mp.name,
        /*sym_visibility=*/StringAttr{}, sfBuilder.getArrayAttr(mp.argNames),
        sfBuilder.getDenseBoolArrayAttr(mp.argIsFirst),
        sfBuilder.getArrayAttr(mp.resultNames),
        sfBuilder.getArrayAttr(mp.subFuncAttrs));
    symbolTable.insert(mpOp);
  }

  // Erase the unified func and register the split_func in the symbol table
  // so that callers processed later can look it up.
  symbolTable.erase(funcOp);
  symbolTable.insert(splitFuncOp);
}

namespace {
struct SplitPhasesPass
    : public hir::impl::SplitPhasesPassBase<SplitPhasesPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Topological Sort and Pass Entry Point
//
// Process unified functions in dependency order (callees before callers). This
// ensures callee split functions exist when we decompose a caller's
// unified_call during splitting.
//===----------------------------------------------------------------------===//

void SplitPhasesPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Run analysis and check call arg phases.
  SmallVector<std::pair<UnifiedFuncOp, PhaseAnalysis>> analyses;
  bool anyErrors = false;
  for (auto op : getOperation().getOps<UnifiedFuncOp>()) {
    auto &[funcOp, analysis] = analyses.emplace_back(op, op);
    analysis.analyze();
    analysis.pullPhases();
    analysis.refreshPhases();
    if (failed(analysis.checkCallArgPhases()))
      anyErrors = true;
  }
  if (anyErrors)
    return signalPassFailure();

  // Build a call graph to topologically sort functions. For each function,
  // record which other unified functions it calls.
  DenseMap<StringRef, unsigned> nameToIndex;
  for (auto [idx, entry] : llvm::enumerate(analyses))
    nameToIndex[entry.first.getSymName()] = idx;

  SmallVector<SmallVector<unsigned>> deps(analyses.size());
  for (unsigned i = 0; i < analyses.size(); ++i) {
    analyses[i].first.getBody().walk([&](UnifiedCallOp callOp) {
      auto it = nameToIndex.find(callOp.getCallee());
      if (it != nameToIndex.end())
        deps[i].push_back(it->second);
    });
  }

  // Post-order DFS for topological sort. Callees appear before callers.
  SmallVector<unsigned> topoOrder;
  SmallVector<uint8_t> visited(analyses.size(), 0);
  bool hasCycle = false;
  std::function<void(unsigned)> visit = [&](unsigned idx) {
    if (visited[idx] == 2)
      return;
    if (visited[idx] == 1) {
      emitError(analyses[idx].first.getLoc())
          << "recursive call cycle detected";
      hasCycle = true;
      return;
    }
    visited[idx] = 1;
    for (unsigned dep : deps[idx])
      visit(dep);
    visited[idx] = 2;
    topoOrder.push_back(idx);
  };
  for (unsigned i = 0; i < analyses.size(); ++i)
    visit(i);
  if (hasCycle)
    return signalPassFailure();

  // Split each function in topological order (callees first).
  for (unsigned idx : topoOrder) {
    auto &[funcOp, analysis] = analyses[idx];
    PhaseSplitter splitter(analysis, symbolTable);
    splitter.run();
    // Unified func is erased inside run().
  }
}
