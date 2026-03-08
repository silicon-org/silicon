//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Analysis/PhaseAnalysis.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
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

/// Resolve a value from one region into another by recursively cloning its
/// defining op and all transitive operands. Values already in the mapping are
/// returned directly (e.g., block arguments that have been pre-mapped).
static Value resolveTypeIntoRegion(OpBuilder &builder, Value val,
                                   IRMapping &mapping) {
  if (auto mapped = mapping.lookupOrNull(val))
    return mapped;
  auto *defOp = val.getDefiningOp();
  for (Value operand : defOp->getOperands())
    resolveTypeIntoRegion(builder, operand, mapping);
  auto *cloned = builder.clone(*defOp, mapping);
  return cloned->getResult(cast<OpResult>(val).getResultNumber());
}

/// Check whether an op and all its transitive operands are pure (side-effect
/// free). This is used to decide whether an op can be cloned into a split
/// phase function instead of being threaded through as a block argument.
static bool isPurelyLocal(Operation *op) {
  if (!mlir::isMemoryEffectFree(op))
    return false;
  for (Value operand : op->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    if (!defOp || !isPurelyLocal(defOp))
      return false;
  }
  return true;
}

/// Clone a pure op and its transitive operand tree into the target location.
/// The mapping is populated with the original-to-clone correspondence.
static void clonePureOp(Operation *op, OpBuilder &builder, IRMapping &mapping) {
  if (mapping.contains(op->getResult(0)))
    return;
  for (Value operand : op->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    clonePureOp(defOp, builder, mapping);
  }
  builder.clone(*op, mapping);
}

namespace {
struct PhaseSplit {
  FuncOp funcOp;
  int16_t phase;
  SmallVector<Value> returnValues;
  SmallVector<Value> returnTypeOperands;
  SmallVector<Value> argTypeOperands;
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
  LogicalResult run();

  PhaseAnalysis &analysis;
  SymbolTable &symbolTable;
  UnifiedFuncOp funcOp;
};
} // namespace

LogicalResult PhaseSplitter::run() {
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

  // Compute effective result phases. The declared result phase is
  // authoritative: if the body produces a value at a later phase than declared,
  // that is a user error (the function signature promises an earlier phase than
  // the body can deliver).
  auto returnOp = funcOp.getReturnOp();
  auto declaredResultPhases = funcOp.getResultPhases();
  SmallVector<int16_t> effectiveResultPhases;
  bool hasReturnPhaseMismatch = false;
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t declared = static_cast<int16_t>(declaredResultPhases[idx]);
    int16_t valuePhase = analysis.getValuePhase(value);
    if (valuePhase == INT16_MIN)
      valuePhase = declared;
    if (valuePhase > declared) {
      emitError(returnOp.getLoc())
          << "return value is available at phase " << valuePhase
          << " but function declares phase " << declared << " return";
      hasReturnPhaseMismatch = true;
    }
    effectiveResultPhases.push_back(std::max(declared, valuePhase));
  }
  if (hasReturnPhaseMismatch)
    return failure();

  // Extend the phase range based on the function's declared arg/result phases.
  // These may lie outside the range of computed op/value phases (e.g., a `dyn`
  // return type shifts the result to a later phase).
  for (auto p : funcOp.getArgPhases()) {
    minPhase = std::min(minPhase, static_cast<int16_t>(p));
    maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
  }
  for (auto p : effectiveResultPhases) {
    minPhase = std::min(minPhase, p);
    maxPhase = std::max(maxPhase, p);
  }

  // Extend the phase range based on UnifiedCallOps in the body. The call's
  // The call's arg/result phase attributes are callee-relative; the absolute
  // caller-frame phase is the call op's phase plus the attribute value.
  funcOp.getBody().walk([&](UnifiedCallOp callOp) {
    int16_t callOpPhase = analysis.opPhases.at(callOp);
    for (auto p : callOp.getArgPhases()) {
      int16_t abs = callOpPhase + static_cast<int16_t>(p);
      minPhase = std::min(minPhase, abs);
      maxPhase = std::max(maxPhase, abs);
    }
    for (auto p : callOp.getResultPhases()) {
      int16_t abs = callOpPhase + static_cast<int16_t>(p);
      minPhase = std::min(minPhase, abs);
      maxPhase = std::max(maxPhase, abs);
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
  for (auto phase : effectiveResultPhases)
    externalPhases.insert(phase);

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
  // placed in the split matching its effective result phase. The effective
  // phase accounts for cases where the return value is only available at a
  // later phase than declared (e.g., a dyn arg forces the result later). If a
  // value is defined in an earlier phase, cross-phase threading will create
  // context returns and block args to forward it. The return op itself is
  // erased since each phase will get its own.
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t resultPhase = effectiveResultPhases[idx];
    splits[resultPhase - minPhase].returnValues.push_back(value);
  }
  for (auto [idx, type] : llvm::enumerate(returnOp.getTypeOfValues())) {
    int16_t resultPhase = effectiveResultPhases[idx];
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

      // The call's arg/result phase attributes are callee-relative; the
      // absolute caller-frame phase is the call op's phase plus the attribute.
      int16_t callOpPhase = analysis.opPhases.at(callOp);

      // Compute the caller-frame phase range for this call.
      int16_t calleeMinPhase = INT16_MAX, calleeMaxPhase = INT16_MIN;
      for (auto p : argPhases) {
        int16_t abs = callOpPhase + static_cast<int16_t>(p);
        calleeMinPhase = std::min(calleeMinPhase, abs);
        calleeMaxPhase = std::max(calleeMaxPhase, abs);
      }
      for (auto p : resultPhases) {
        int16_t abs = callOpPhase + static_cast<int16_t>(p);
        calleeMinPhase = std::min(calleeMinPhase, abs);
        calleeMaxPhase = std::max(calleeMaxPhase, abs);
      }
      if (calleeMinPhase == INT16_MAX) {
        calleeMinPhase = calleeMaxPhase = 0;
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
        int16_t p = callOpPhase + static_cast<int16_t>(phase);
        if (auto *defOp = type.getDefiningOp()) {
          auto &opPhase = analysis.opPhases[defOp];
          opPhase = std::min(opPhase, p);
          auto &valPhase = analysis.valuePhases[type];
          valPhase = std::min(valPhase, p);
        }
      }
      for (auto [type, phase] :
           llvm::zip(callOp.getTypeOfResults(), resultPhases)) {
        int16_t p = callOpPhase + static_cast<int16_t>(phase);
        if (auto *defOp = type.getDefiningOp()) {
          auto &opPhase = analysis.opPhases[defOp];
          opPhase = std::min(opPhase, p);
          auto &valPhase = analysis.valuePhases[type];
          valPhase = std::min(valPhase, p);
        }
      }

      // Partition arguments and their type operands by absolute phase.
      DenseMap<int16_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
      for (auto [arg, type, phase] : llvm::zip(
               callOp.getArguments(), callOp.getTypeOfArgs(), argPhases)) {
        int16_t abs = callOpPhase + static_cast<int16_t>(phase);
        phaseArgs[abs].push_back(arg);
        phaseTypeOfArgs[abs].push_back(type);
      }

      // Determine the phase that produces the unified_call's actual results.
      // This is the maximum result phase, which may differ from
      // calleeMaxPhase when dyn args extend the callee beyond its results.
      int16_t maxResultPhase = INT16_MIN;
      for (auto p : resultPhases)
        maxResultPhase = std::max(
            maxResultPhase,
            static_cast<int16_t>(callOpPhase + static_cast<int16_t>(p)));

      // Collect which result indices belong to which phase.
      DenseSet<int16_t> resultPhaseSet;
      for (auto p : resultPhases)
        resultPhaseSet.insert(callOpPhase + static_cast<int16_t>(p));

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

  //===--------------------------------------------------------------------===//
  // Dissolve ExprOps
  //
  // After phase splitting, ExprOps have served their purpose as phase
  // annotation boundaries. Inline their bodies into the parent block and
  // replace results with the yielded values.
  //===--------------------------------------------------------------------===//

  for (auto &split : splits) {
    SmallVector<ExprOp> exprOps;
    split.funcOp.walk<WalkOrder::PostOrder>(
        [&](ExprOp op) { exprOps.push_back(op); });
    for (auto exprOp : exprOps) {
      auto &body = exprOp.getBody().front();
      auto yieldOp = cast<YieldOp>(body.getTerminator());
      for (auto [result, operand] :
           llvm::zip(exprOp.getResults(), yieldOp.getOperands())) {
        result.replaceAllUsesWith(operand);
        // Update returnValues across ALL splits, not just the current one.
        // An ExprOp result may have been stashed in a different split's
        // returnValues (e.g., when the function's result phase differs from
        // the ExprOp's phase due to a phaseShift).
        for (auto &s : splits)
          for (auto &rv : s.returnValues)
            if (rv == result)
              rv = operand;
      }
      yieldOp.erase();
      auto *parentBlock = exprOp->getBlock();
      parentBlock->getOperations().splice(exprOp->getIterator(),
                                          body.getOperations());
      exprOp.erase();
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

        // Resolve the signature type value to a body/split value. This
        // recursively clones the defining op and its transitive operands.
        Value resolvedType =
            resolveTypeIntoRegion(insertBuilder, sigTypeVal, sigToBody);

        auto coerceOp = CoerceTypeOp::create(insertBuilder, ownArg.getLoc(),
                                             ownArg, resolvedType);
        split.argTypeOperands.push_back(resolvedType);

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
    ReturnOp::create(builder, funcOp.getLoc(), split.returnValues, returnTypes,
                     split.argTypeOperands);

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

            // Pure ops (constants, type constructors, etc.) can be cloned
            // into the current split instead of threading through block args.
            // This keeps type constants like `hir.int_type` and
            // `hir.uint_type` local to each phase, which is needed by the
            // HIR-to-MIR lowering. The clone is recursive to handle ops like
            // `uint_type(constant_int 42)` where the type op depends on
            // other pure ops.
            auto *defOp = operand.get().getDefiningOp();
            if (defOp && isPurelyLocal(defOp)) {
              LLVM_DEBUG({
                llvm::dbgs() << "- Cloning into phase " << phase << ": "
                             << defOp->getName() << "\n";
              });
              auto &block = split.funcOp.getBody().front();
              OpBuilder cloneBuilder(&block, block.begin());
              IRMapping cloneMapping;
              clonePureOp(defOp, cloneBuilder, cloneMapping);
              value = cloneMapping.lookup(operand.get());
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

    // Update the return op's typeOfArgs: replace any entries for the old
    // context args with a single opaque type for the bundled context arg.
    auto retOp = cast<ReturnOp>(block.getTerminator());
    SmallVector<Value> newArgTypes(retOp.getTypeOfArgs().take_front(ownArgs));
    newArgTypes.push_back(
        OpaqueTypeOp::create(unpackBuilder, funcOp.getLoc()).getResult());
    retOp.getTypeOfArgsMutable().assign(newArgTypes);

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
    SmallVector<Value> argTypes(prevReturnOp.getTypeOfArgs());
    ReturnOp::create(packBuilder, funcOp.getLoc(), newReturnValues,
                     newReturnTypes, argTypes);
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
           llvm::zip(funcOp.getResultNames(), effectiveResultPhases)) {
        if (resultPhase == phase)
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
  SmallVector<int32_t> sfResultPhases(effectiveResultPhases.begin(),
                                      effectiveResultPhases.end());

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
  return success();
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
    if (failed(splitter.run()))
      return signalPassFailure();
    // Unified func is erased inside run().
  }
}
