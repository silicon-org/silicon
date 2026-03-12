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

/// Check whether an op and all its transitive operands are pure (side-effect
/// free). This is used to decide whether an op can be cloned into a split
/// phase function instead of being threaded through as a block argument.
static bool isPurelyLocal(Operation *op) {
  if (!hir::isEffectivelyPure(op))
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
  if (op->getNumResults() == 0) {
    emitBug(op->getLoc()) << "clonePureOp called on zero-result op `"
                          << op->getName() << "`";
    return;
  }
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

  /// Per-phase split functions and their return values.
  SmallVector<PhaseSplit> splits;

  /// Effective result phases (max of declared and computed).
  SmallVector<int16_t> effectiveResultPhases;

  /// Phase range of the function.
  int16_t minPhase = 0, maxPhase = 0;

  /// Number of "own" block args and return values per phase, before
  /// cross-phase threading adds context values.
  SmallVector<unsigned> numOwnArgs;
  SmallVector<unsigned> numOwnReturns;

private:
  /// Split the body region by distributing ops into per-phase functions,
  /// threading cross-phase values, and bundling context into opaque packs.
  LogicalResult splitBodyByPhase();

  /// Populate per-phase signature regions. For each phase function, resolve
  /// arg and result types from the unified signature by cloning pure ops or
  /// falling back to OpaqueTypeOp for cross-phase dependencies.
  void splitSignatureByPhase();
};
} // namespace

LogicalResult PhaseSplitter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Splitting " << funcOp.getSymNameAttr() << "\n");

  // Determine the phase range, skipping INT16_MIN (floating constants).
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
  // the body can deliver). For return values that come from a unified_call, we
  // use the call's result phase attribute to determine the actual phase, since
  // the phase analysis assigns all unified_call results to the call op's own
  // phase without considering callee-relative result shifts (e.g., `dyn fn`).
  SmallVector<ReturnOp> allReturns;
  funcOp.getBody().walk([&](ReturnOp r) { allReturns.push_back(r); });
  assert(!allReturns.empty() && "unified_func must have at least one return");
  auto returnOp = allReturns.front();
  auto declaredResultPhases = funcOp.getResultPhases();
  bool hasReturnPhaseMismatch = false;
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t declared = static_cast<int16_t>(declaredResultPhases[idx]);
    int16_t valuePhase = analysis.getValuePhase(value);
    if (valuePhase == INT16_MIN)
      valuePhase = declared;

    // If this return value comes from a unified_call, compute the actual
    // result phase from the call's result phase attributes.
    if (auto callOp = value.getDefiningOp<UnifiedCallOp>()) {
      auto resultIdx = cast<OpResult>(value).getResultNumber();
      int16_t callOpPhase = analysis.opPhases.at(callOp.getOperation());
      int16_t relResultPhase =
          static_cast<int16_t>(callOp.getResultPhases()[resultIdx]);
      valuePhase = callOpPhase + relResultPhase;
    }
    if (valuePhase > declared) {
      auto diag = emitError(returnOp.getLoc())
                  << "return value is available at phase " << valuePhase
                  << " but function declares phase " << declared << " return";
      if (valuePhase > 0)
        diag.attachNote(funcOp.getLoc())
            << "consider adding `dyn` to the return type to shift it to a "
               "later phase";
      else
        diag.attachNote(funcOp.getLoc())
            << "consider adding `const` to the arguments or removing `const` "
               "from the return type";
      hasReturnPhaseMismatch = true;
    }
    effectiveResultPhases.push_back(std::max(declared, valuePhase));
  }
  if (hasReturnPhaseMismatch)
    return failure();

  // Verify that earlier-phase results are identical across all returns.
  // Earlier-phase results get factored out before the split point, so they
  // must be the same SSA value in every return. (Trivially satisfied for
  // single-return functions.)
  if (allReturns.size() > 1) {
    for (auto [idx, phase] : llvm::enumerate(effectiveResultPhases)) {
      if (phase >= 0)
        continue;
      Value firstVal = allReturns[0].getValues()[idx];
      for (unsigned i = 1; i < allReturns.size(); ++i) {
        if (allReturns[i].getValues()[idx] != firstVal) {
          emitError(allReturns[i].getLoc())
              << "earlier-phase result at position " << idx
              << " differs across returns; all returns must produce the "
                 "same value for const results";
          return failure();
        }
      }
    }
  }

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
  splits.resize(maxPhase - minPhase + 1);
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
      // Create an empty signature region with a placeholder terminator.
      // Block args and actual type operands are added later once the per-phase
      // functions have their final arg/result shapes.
      {
        auto &sigBlock = phaseFuncOp.getSignature().emplaceBlock();
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToEnd(&sigBlock);
        SignatureOp::create(builder, funcOp.getLoc(), ValueRange{},
                            ValueRange{});
      }
      if (phase != 0)
        phaseFuncOp.getBody().emplaceBlock();
      symbolTable.insert(phaseFuncOp);
      builder.setInsertionPoint(phaseFuncOp);
      splits[phase - minPhase].funcOp = phaseFuncOp;
      splits[phase - minPhase].phase = phase;
    }
  }

  if (failed(splitBodyByPhase()))
    return failure();

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
    ReturnOp firstRetOp;
    split.funcOp.getBody().walk([&](ReturnOp r) {
      if (!firstRetOp)
        firstRetOp = r;
    });
    if (firstRetOp) {
      unsigned ownReturns = numOwnReturns[phase - minPhase];
      // Check if this phase has user-visible results.
      for (auto [name, resultPhase] :
           llvm::zip(funcOp.getResultNames(), effectiveResultPhases)) {
        if (resultPhase == phase)
          phaseResultNames.push_back(name);
      }
      if (firstRetOp.getValues().size() > ownReturns)
        phaseResultNames.push_back(builder.getStringAttr("ctx"));
    }
    split.funcOp.setResultNamesAttr(builder.getArrayAttr(phaseResultNames));
  }

  // Populate per-phase signature regions.
  splitSignatureByPhase();

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

  // Move the unified func's signature region into the split_func.
  splitFuncOp.getSignature().takeBody(funcOp.getSignature());

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

//===----------------------------------------------------------------------===//
// splitBodyByPhase
//
// Handle the return operation, move the body to phase 0, shift block args,
// decompose unified calls, distribute ops by phase, dissolve ExprOps, thread
// cross-phase values, and bundle context into opaque packs.
//===----------------------------------------------------------------------===//

LogicalResult PhaseSplitter::splitBodyByPhase() {
  // Handle the return operation. Each return value is placed in the split
  // matching its effective result phase. The return op itself is erased since
  // each phase will get its own.
  SmallVector<ReturnOp> allReturns;
  funcOp.getBody().walk([&](ReturnOp r) { allReturns.push_back(r); });
  assert(!allReturns.empty() && "unified_func must have at least one return");
  auto returnOp = allReturns.front();
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t resultPhase = effectiveResultPhases[idx];
    splits[resultPhase - minPhase].returnValues.push_back(value);
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
    // phases as needed.
    for (auto &[idx, ownArg] : replacements) {
      auto bodyArg = phase0Block.getArgument(idx);
      bodyArg.replaceAllUsesWith(ownArg);
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

      // Look up the callee's split_func and build a phase-to-function mapping
      // by walking its entries. MultiphaseFuncOps expand to their
      // sub-functions.
      auto calleeSplitFunc = symbolTable.lookup<SplitFuncOp>(calleeName);
      if (!calleeSplitFunc)
        continue;

      // Build a list of call entries from the callee's split_func. Each
      // entry produces exactly one call, whether it's a standalone FuncOp
      // or a MultiphaseFuncOp. We do NOT expand MultiphaseFuncOps into
      // their sub-functions — callers call the multiphase_func itself, and
      // SpecializeFuncs handles the internal decomposition later.
      struct CallEntry {
        int16_t phase;
        StringRef symbolName;
        unsigned numResults;
      };
      SmallVector<CallEntry> splitEntries;
      {
        auto phaseNums = calleeSplitFunc.getPhaseNumbers();
        auto phaseFuncRefs = calleeSplitFunc.getPhaseFuncs();

        for (auto [entryPhase, funcRef] : llvm::zip(phaseNums, phaseFuncRefs)) {
          auto name = cast<FlatSymbolRefAttr>(funcRef).getValue();
          int16_t callerPhase = callOpPhase + entryPhase;
          unsigned numResults;
          if (auto mpFunc = symbolTable.lookup<MultiphaseFuncOp>(name)) {
            numResults = mpFunc.getResultNames().size();
          } else if (auto func = symbolTable.lookup<FuncOp>(name)) {
            numResults = func.getResultNames().size();
          } else {
            continue;
          }
          splitEntries.push_back({callerPhase, name, numResults});
        }
      }
      if (splitEntries.empty())
        continue;

      // Determine the caller-frame phase range from the split entries.
      int16_t calleeMaxPhase = INT16_MIN;
      for (auto &entry : splitEntries)
        calleeMaxPhase = std::max(calleeMaxPhase, entry.phase);

      OpBuilder callBuilder(callOp);
      auto loc = callOp.getLoc();

      // Update phases of existing type operands so they land in the correct
      // split function during distribution.
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
      int16_t maxResultPhase = INT16_MIN;
      for (auto p : resultPhases)
        maxResultPhase = std::max(
            maxResultPhase,
            static_cast<int16_t>(callOpPhase + static_cast<int16_t>(p)));

      // Collect which result indices belong to which phase.
      DenseSet<int16_t> resultPhaseSet;
      for (auto p : resultPhases)
        resultPhaseSet.insert(callOpPhase + static_cast<int16_t>(p));

      // Chain calls from earliest phase to latest.
      SmallVector<Value> prevResults;
      SmallVector<Value> unifiedCallReplacements;
      for (auto &entry : splitEntries) {
        int16_t phase = entry.phase;
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
        // to later phases), add opaque result types.
        if (!isResultPhase || phase != calleeMaxPhase) {
          unsigned startIdx = isResultPhase ? callOp.getNumResults() : 0;
          for (unsigned i = startIdx; i < entry.numResults; ++i) {
            auto opaqueType = OpaqueTypeOp::create(callBuilder, loc);
            analysis.opPhases[opaqueType] = phase;
            analysis.valuePhases[opaqueType.getResult()] = phase;
            callTypeOfResults.push_back(opaqueType.getResult());
            resultTypes.push_back(AnyType::get(callBuilder.getContext()));
          }
        }

        auto call = CallOp::create(callBuilder, loc, resultTypes,
                                   callBuilder.getStringAttr(entry.symbolName),
                                   callArgs, callTypeOfArgs, callTypeOfResults);
        analysis.opPhases[call] = phase;
        for (auto result : call.getResults())
          analysis.valuePhases[result] = phase;

        if (isResultPhase) {
          for (unsigned i = 0; i < callOp.getNumResults(); ++i)
            unifiedCallReplacements.push_back(call.getResult(i));
          prevResults.assign(call.getResults().begin() + callOp.getNumResults(),
                             call.getResults().end());
        } else {
          prevResults.assign(call.getResults().begin(),
                             call.getResults().end());
        }
      }

      // Update any return values that reference the old unified call's results.
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

    if (auto exprOp = dyn_cast<ExprOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "- Descending into " << op->getName() << "\n");
      for (auto &block : llvm::reverse(exprOp.getBody()))
        worklist.push_back({opPhase, block.begin(), block.end()});
    } else {
      // TODO: Otherwise ensure that any nested operations have been assigned
      // the same phase.
    }
  }

  //===--------------------------------------------------------------------===//
  // Dissolve ExprOps
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

  // Record the number of "own" block args and return values per phase.
  numOwnArgs.assign(maxPhase - minPhase + 1, 0);
  numOwnReturns.assign(maxPhase - minPhase + 1, 0);
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    numOwnArgs[phase - minPhase] =
        splits[phase - minPhase].funcOp.getBody().front().getNumArguments();
    numOwnReturns[phase - minPhase] =
        splits[phase - minPhase].returnValues.size();
  }

  //===--------------------------------------------------------------------===//
  // Cross-Phase Value Threading
  //===--------------------------------------------------------------------===//

  OpBuilder builder(funcOp);
  DenseMap<Value, Value> mapping;
  SmallDenseSet<Operation *, 4> closedFuncs;
  for (int16_t phase = maxPhase; phase >= minPhase; --phase) {
    auto &split = splits[phase - minPhase];
    closedFuncs.insert(split.funcOp);

    builder.setInsertionPointToEnd(&split.funcOp.getBody().back());
    ReturnOp::create(builder, funcOp.getLoc(), split.returnValues,
                     /*typeOfValues=*/ValueRange{});

    split.funcOp.walk([&](Operation *op) {
      for (auto &operand : op->getOpOperands()) {
        auto &value = mapping[operand.get()];
        if (!value) {
          auto *parentOp = operand.get().getParentBlock()->getParentOp();
          while (!isa<FuncOp>(parentOp))
            parentOp = parentOp->getParentOp();

          if (parentOp == split.funcOp) {
            value = operand.get();
          } else {
            if (closedFuncs.contains(parentOp)) {
              emitBug(op->getLoc()) << "op uses value from later phase";
              return;
            }

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
              LLVM_DEBUG({
                llvm::dbgs()
                    << "- Creating arg in phase " << phase << " for value from "
                    << operand.get().getDefiningOp()->getName() << "\n";
              });
              value = split.funcOp.getBody().front().addArgument(
                  operand.get().getType(), operand.get().getLoc());

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
  //===--------------------------------------------------------------------===//

  auto anyType = AnyType::get(builder.getContext());
  for (int16_t phase = minPhase + 1; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &block = split.funcOp.getBody().front();
    unsigned ownArgs = numOwnArgs[phase - minPhase];
    unsigned totalArgs = block.getNumArguments();
    unsigned contextArgs = totalArgs - ownArgs;

    auto opaqueArg = block.addArgument(anyType, funcOp.getLoc());
    OpBuilder unpackBuilder(&block, block.begin());
    auto unpackOp = OpaqueUnpackOp::create(
        unpackBuilder, funcOp.getLoc(), SmallVector<Type>(contextArgs, anyType),
        opaqueArg);
    for (unsigned i = 0; i < contextArgs; ++i)
      block.getArgument(ownArgs + i).replaceAllUsesWith(unpackOp.getResult(i));
    block.eraseArguments(ownArgs, contextArgs);

    auto &prevSplit = splits[phase - 1 - minPhase];
    unsigned prevOwnReturns = numOwnReturns[phase - 1 - minPhase];

    SmallVector<ReturnOp> prevReturns;
    prevSplit.funcOp.getBody().walk(
        [&](ReturnOp r) { prevReturns.push_back(r); });
    for (auto prevReturnOp : prevReturns) {
      OpBuilder packBuilder(prevReturnOp);
      SmallVector<Value> contextValues(
          prevReturnOp.getValues().drop_front(prevOwnReturns));
      auto packOp =
          OpaquePackOp::create(packBuilder, funcOp.getLoc(), contextValues);

      SmallVector<Value> newReturnValues(
          prevReturnOp.getValues().take_front(prevOwnReturns));
      newReturnValues.push_back(packOp);
      ReturnOp::create(packBuilder, funcOp.getLoc(), newReturnValues,
                       /*typeOfValues=*/ValueRange{});
      prevReturnOp.erase();
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// splitSignatureByPhase
//
// Populate per-phase signature regions from the unified signature. For each
// phase function, we add sig block args matching the body, map the unified
// sig block args to their per-phase counterparts, then resolve each type
// operand (arg and result types) by cloning pure ops or falling back to
// OpaqueTypeOp for cross-phase dependencies.
//===----------------------------------------------------------------------===//

void PhaseSplitter::splitSignatureByPhase() {
  hir::consolidateSignatureTerminators(funcOp.getSignature());
  auto unifiedSigOp =
      cast<SignatureOp>(funcOp.getSignature().back().getTerminator());
  auto unifiedArgPhases = funcOp.getArgPhases();
  auto &unifiedSigBlock = funcOp.getSignature().front();

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &sigBlock = split.funcOp.getSignature().front();
    auto &bodyBlock = split.funcOp.getBody().front();

    // Add block args matching the body's entry block.
    for (auto bodyArg : bodyBlock.getArguments())
      sigBlock.addArgument(bodyArg.getType(), bodyArg.getLoc());

    OpBuilder sigBuilder(&sigBlock, sigBlock.begin());
    IRMapping sigMapping;

    // Map unified sig block args to per-phase sig block args for own args.
    unsigned localArgIdx = 0;
    for (unsigned i = 0; i < unifiedArgPhases.size(); ++i) {
      if (static_cast<int16_t>(unifiedArgPhases[i]) == phase)
        sigMapping.map(unifiedSigBlock.getArgument(i),
                       sigBlock.getArgument(localArgIdx++));
    }

    // Resolve a type value from the unified signature into this phase's
    // signature. Pure ops are cloned transitively; cross-phase block arg
    // dependencies fall back to OpaqueTypeOp.
    auto resolveType = [&](Value val) -> Value {
      if (auto mapped = sigMapping.lookupOrNull(val))
        return mapped;
      auto *defOp = val.getDefiningOp();
      if (!defOp)
        return OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult();
      if (isPurelyLocal(defOp)) {
        clonePureOp(defOp, sigBuilder, sigMapping);
        return sigMapping.lookup(val);
      }
      return OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult();
    };

    // Resolve arg types for this phase's own args.
    SmallVector<Value> sigArgTypes;
    for (unsigned i = 0; i < unifiedArgPhases.size(); ++i) {
      if (static_cast<int16_t>(unifiedArgPhases[i]) == phase)
        sigArgTypes.push_back(resolveType(unifiedSigOp.getTypeOfArgs()[i]));
    }

    // Context arg gets opaque type.
    if (sigBlock.getNumArguments() > sigArgTypes.size())
      sigArgTypes.push_back(
          OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult());

    // Resolve result types for this phase.
    SmallVector<Value> sigResultTypes;
    for (auto [idx, resultPhase] : llvm::enumerate(effectiveResultPhases)) {
      if (resultPhase == phase)
        sigResultTypes.push_back(
            resolveType(unifiedSigOp.getTypeOfResults()[idx]));
    }

    // Context result gets opaque type.
    ReturnOp splitRetOp;
    split.funcOp.getBody().walk([&](ReturnOp r) {
      if (!splitRetOp)
        splitRetOp = r;
    });
    if (splitRetOp)
      if (splitRetOp.getValues().size() > sigResultTypes.size())
        sigResultTypes.push_back(
            OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult());

    // Replace the placeholder SignatureOp terminator.
    sigBlock.getTerminator()->erase();
    sigBuilder.setInsertionPointToEnd(&sigBlock);
    SignatureOp::create(sigBuilder, funcOp.getLoc(), sigArgTypes,
                        sigResultTypes);
  }
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

  // Post-order DFS for topological sort. Callees appear before callers. If a
  // cycle is detected, we report an error with the full call cycle listed,
  // since recursive functions cannot be synthesized to hardware.
  SmallVector<unsigned> topoOrder;
  SmallVector<uint8_t> visited(analyses.size(), 0);
  SmallVector<unsigned> dfsStack;
  bool hasCycle = false;
  std::function<void(unsigned)> visit = [&](unsigned idx) {
    if (visited[idx] == 2)
      return;
    if (visited[idx] == 1) {
      // Find the cycle in the DFS stack and list the involved functions.
      auto *cycleStart = llvm::find(dfsStack, idx);
      assert(cycleStart != dfsStack.end());
      SmallVector<unsigned> cycle(cycleStart, dfsStack.end());
      cycle.push_back(idx);
      auto diag = emitError(analyses[idx].first.getLoc())
                  << "recursive call cycle detected; recursive functions "
                     "cannot be synthesized to hardware because they require "
                     "unbounded inlining";
      for (unsigned i = 0; i + 1 < cycle.size(); ++i)
        diag.attachNote(analyses[cycle[i]].first.getLoc())
            << "'" << analyses[cycle[i]].first.getSymName() << "' calls '"
            << analyses[cycle[i + 1]].first.getSymName() << "'";
      diag.attachNote() << "consider restructuring your code to avoid "
                           "recursion, e.g., by using loops or unrolled logic";
      hasCycle = true;
      return;
    }
    visited[idx] = 1;
    dfsStack.push_back(idx);
    for (unsigned dep : deps[idx])
      visit(dep);
    dfsStack.pop_back();
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
