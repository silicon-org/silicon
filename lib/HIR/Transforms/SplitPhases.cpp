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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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

/// Resolve a type value for use in a reconstructed signature. Traces through
/// phi block arguments by inspecting predecessor branches. If all predecessors
/// pass the same value, resolves that common value instead. Falls back to
/// OpaqueTypeOp for cycles, divergent values, or unresolvable ops.
static Value resolveTypeValue(Value val, OpBuilder &builder, IRMapping &mapping,
                              Location loc, SmallPtrSetImpl<Value> &visited) {
  if (auto mapped = mapping.lookupOrNull(val))
    return mapped;

  // Has a defining op — try to clone it.
  if (auto *defOp = val.getDefiningOp()) {
    if (isPurelyLocal(defOp)) {
      clonePureOp(defOp, builder, mapping);
      return mapping.lookup(val);
    }
    return OpaqueTypeOp::create(builder, loc).getResult();
  }

  // Phi block arg — trace through predecessors.
  auto blockArg = cast<BlockArgument>(val);
  if (!visited.insert(val).second)
    return OpaqueTypeOp::create(builder, loc).getResult();

  auto *block = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();

  Value commonValue;
  bool divergent = false;
  for (auto *pred : block->getPredecessors()) {
    auto *term = pred->getTerminator();
    Value incoming;
    if (auto brOp = dyn_cast<cf::BranchOp>(term)) {
      incoming = brOp.getDestOperands()[argIdx];
    } else if (auto condBr = dyn_cast<cf::CondBranchOp>(term)) {
      if (condBr.getTrueDest() == block)
        incoming = condBr.getTrueDestOperands()[argIdx];
      else
        incoming = condBr.getFalseDestOperands()[argIdx];
    } else {
      divergent = true;
      break;
    }
    if (!commonValue)
      commonValue = incoming;
    else if (commonValue != incoming) {
      divergent = true;
      break;
    }
  }

  if (divergent || !commonValue)
    return OpaqueTypeOp::create(builder, loc).getResult();

  // All predecessors agree — resolve the common value recursively.
  return resolveTypeValue(commonValue, builder, mapping, loc, visited);
}

namespace {
struct PhaseSplit {
  FuncOp funcOp;
  int16_t phase;
  /// Per-phase return values from the body's ReturnOp.
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

  /// Build per-phase SignatureOps by resolving type values from the unified
  /// func's original signature. Pure ops are cloned transitively; unresolvable
  /// values fall back to OpaqueTypeOp.
  void reconstructSignatures();

  /// Walk each phase function's returns and populate typeOfValues from the
  /// signature's typeOfResults, cloning type ops into the body region.
  void populateReturnTypeOfValues();
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
      // Create empty body and signature blocks. Phase 0 already gets its body
      // from takeBody; other phases need an empty block. The signature is
      // populated later by reconstructSignatures.
      phaseFuncOp.getSignature().emplaceBlock();
      if (phase != 0)
        phaseFuncOp.getBody().emplaceBlock();
      symbolTable.insert(phaseFuncOp);
      builder.setInsertionPoint(phaseFuncOp);
      splits[phase - minPhase].funcOp = phaseFuncOp;
      splits[phase - minPhase].phase = phase;
    }
  }

  // Clone the unified signature before splitting consumes it. The clone is
  // moved into the split_func later to preserve the original signature.
  Region clonedUnifiedSig;
  IRMapping sigCloneMapping;
  funcOp.getSignature().cloneInto(&clonedUnifiedSig, sigCloneMapping);

  if (failed(splitBodyByPhase()))
    return failure();
  reconstructSignatures();
  populateReturnTypeOfValues();

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

  // Move the cloned unified signature into the split_func.
  splitFuncOp.getSignature().takeBody(clonedUnifiedSig);

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
// Split the body region by distributing ops into per-phase functions, threading
// cross-phase values via block args, and bundling context into opaque packs.
//===----------------------------------------------------------------------===//

LogicalResult PhaseSplitter::splitBodyByPhase() {
  Region &sourceRegion = funcOp.getBody();

  for (auto &split : splits)
    split.returnValues.clear();

  //===--------------------------------------------------------------------===//
  // Exit Block Normalization
  //
  // If the body has multiple returns across different blocks, normalize to a
  // single exit block. Each original return is replaced with a branch to the
  // exit block, which holds the single canonical return.
  //===--------------------------------------------------------------------===//

  SmallVector<ReturnOp> allReturns;
  sourceRegion.walk([&](ReturnOp op) { allReturns.push_back(op); });
  assert(!allReturns.empty() && "body must have at least one return");

  if (allReturns.size() > 1) {
    auto *exitBlock = new Block();
    sourceRegion.push_back(exitBlock);

    SmallVector<Type> argTypes;
    SmallVector<Location> argLocs;
    for (auto value : allReturns.front().getValues()) {
      argTypes.push_back(value.getType());
      argLocs.push_back(value.getLoc());
    }
    exitBlock->addArguments(argTypes, argLocs);

    for (auto returnOp : allReturns) {
      int16_t termPhase = analysis.opPhases.at(returnOp.getOperation());
      OpBuilder termBuilder(returnOp);
      auto brOp =
          mlir::cf::BranchOp::create(termBuilder, returnOp.getLoc(), exitBlock,
                                     SmallVector<Value>(returnOp.getValues()));
      analysis.opPhases[brOp] = termPhase;
      returnOp.erase();
    }

    OpBuilder exitBuilder(exitBlock, exitBlock->end());
    ReturnOp::create(exitBuilder, funcOp.getLoc(), exitBlock->getArguments(),
                     /*typeOfValues=*/ValueRange{});

    for (auto arg : exitBlock->getArguments())
      analysis.valuePhases[arg] = 0;
    for (auto &op : *exitBlock) {
      analysis.opPhases[&op] = 0;
      for (auto result : op.getResults())
        analysis.valuePhases[result] = 0;
    }

    allReturns.clear();
    sourceRegion.walk([&](ReturnOp op) { allReturns.push_back(op); });
  }

  //===--------------------------------------------------------------------===//
  // Partition Return Values by Phase
  //===--------------------------------------------------------------------===//

  auto returnOp = allReturns.front();
  for (auto [idx, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t resultPhase = effectiveResultPhases[idx];
    splits[resultPhase - minPhase].returnValues.push_back(value);
  }
  returnOp.erase();

  // Move all operations to phase 0 initially.
  splits[0 - minPhase].funcOp.getBody().takeBody(sourceRegion);

  // Move shifted-phase block arguments to their respective phase functions.
  {
    auto &phase0Block = splits[0 - minPhase].funcOp.getBody().front();

    SmallVector<std::pair<unsigned, int16_t>> shiftedArgs;
    for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
      int16_t argPhase = static_cast<int16_t>(phase);
      if (argPhase != 0)
        shiftedArgs.push_back({idx, argPhase});
    }
    llvm::sort(shiftedArgs);

    SmallVector<std::pair<unsigned, Value>> replacements;
    SmallVector<unsigned> argsToErase;
    for (auto [idx, argPhase] : shiftedArgs) {
      auto blockArg = phase0Block.getArgument(idx);
      LLVM_DEBUG(llvm::dbgs() << "- Moving body arg " << idx << " to phase "
                              << argPhase << "\n");
      auto &targetBlock = splits[argPhase - minPhase].funcOp.getBody().front();
      Value ownArg =
          targetBlock.addArgument(blockArg.getType(), blockArg.getLoc());
      replacements.push_back({idx, ownArg});
      argsToErase.push_back(idx);
    }

    for (auto &[idx, ownArg] : replacements) {
      auto blockArg = phase0Block.getArgument(idx);
      for (auto &split : splits)
        for (auto &val : split.returnValues)
          if (val == blockArg)
            val = ownArg;
      blockArg.replaceAllUsesWith(ownArg);
    }

    for (auto idx : llvm::reverse(argsToErase))
      phase0Block.eraseArgument(idx);
  }

  // Track where each unified func arg ended up after shifted arg movement.
  SmallVector<Value> unifiedArgValues;
  unifiedArgValues.resize(funcOp.getArgPhases().size());
  {
    auto &phase0Block = splits[0 - minPhase].funcOp.getBody().front();
    unsigned phase0ArgIdx = 0;
    for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
      int16_t argPhase = static_cast<int16_t>(phase);
      if (argPhase != 0) {
        auto &targetBlock =
            splits[argPhase - minPhase].funcOp.getBody().front();
        unsigned numArgsAtPhase = 0;
        for (auto [j, p] : llvm::enumerate(funcOp.getArgPhases()))
          if (static_cast<int16_t>(p) == argPhase && j <= idx)
            ++numArgsAtPhase;
        unifiedArgValues[idx] = targetBlock.getArgument(numArgsAtPhase - 1);
      } else {
        unifiedArgValues[idx] = phase0Block.getArgument(phase0ArgIdx++);
      }
    }
  }

  //===--------------------------------------------------------------------===//
  // Decompose Unified Calls
  //===--------------------------------------------------------------------===//

  {
    auto &phase0Body = splits[0 - minPhase].funcOp.getBody();
    SmallVector<UnifiedCallOp> unifiedCalls;
    phase0Body.walk([&](UnifiedCallOp op) { unifiedCalls.push_back(op); });

    for (auto callOp : unifiedCalls) {
      auto calleeName = callOp.getCallee();
      auto argPhases = callOp.getArgPhases();
      auto resultPhases = callOp.getResultPhases();

      int16_t callOpPhase = analysis.opPhases.at(callOp);

      auto calleeSplitFunc = symbolTable.lookup<SplitFuncOp>(calleeName);
      if (!calleeSplitFunc)
        continue;

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

      int16_t calleeMaxPhase = INT16_MIN;
      for (auto &entry : splitEntries)
        calleeMaxPhase = std::max(calleeMaxPhase, entry.phase);

      OpBuilder callBuilder(callOp);
      auto loc = callOp.getLoc();

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

      DenseMap<int16_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
      for (auto [arg, type, phase] : llvm::zip(
               callOp.getArguments(), callOp.getTypeOfArgs(), argPhases)) {
        int16_t abs = callOpPhase + static_cast<int16_t>(phase);
        phaseArgs[abs].push_back(arg);
        phaseTypeOfArgs[abs].push_back(type);
      }

      int16_t maxResultPhase = INT16_MIN;
      for (auto p : resultPhases)
        maxResultPhase = std::max(
            maxResultPhase,
            static_cast<int16_t>(callOpPhase + static_cast<int16_t>(p)));

      DenseSet<int16_t> resultPhaseSet;
      for (auto p : resultPhases)
        resultPhaseSet.insert(callOpPhase + static_cast<int16_t>(p));

      SmallVector<Value> prevResults;
      SmallVector<Value> unifiedCallReplacements;
      for (auto &entry : splitEntries) {
        int16_t phase = entry.phase;
        SmallVector<Value> callArgs(phaseArgs[phase]);
        SmallVector<Value> callTypeOfArgs(phaseTypeOfArgs[phase]);

        for (auto result : prevResults) {
          callArgs.push_back(result);
          auto opaqueType = OpaqueTypeOp::create(callBuilder, loc);
          analysis.opPhases[opaqueType] = phase;
          analysis.valuePhases[opaqueType.getResult()] = phase;
          callTypeOfArgs.push_back(opaqueType.getResult());
        }

        bool isResultPhase = (phase == maxResultPhase);
        SmallVector<Value> callTypeOfResults;
        SmallVector<Type> resultTypes;
        if (isResultPhase) {
          callTypeOfResults.append(callOp.getTypeOfResults().begin(),
                                   callOp.getTypeOfResults().end());
          resultTypes.append(callOp.getResultTypes().begin(),
                             callOp.getResultTypes().end());
        }

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

  //===--------------------------------------------------------------------===//
  // Distribute Ops to Phase Functions
  //===--------------------------------------------------------------------===//

  SmallVector<std::tuple<int16_t, Block::iterator, Block::iterator>> worklist;
  auto &phase0Body = splits[0 - minPhase].funcOp.getBody();
  LLVM_DEBUG({
    llvm::dbgs() << "- Distribution: " << phase0Body.getBlocks().size()
                 << " blocks in phase 0 body\n";
    for (auto &block : phase0Body)
      for (auto &op : block)
        llvm::dbgs() << "  op: " << op << "\n";
  });
  for (auto &block : llvm::reverse(phase0Body))
    worklist.push_back({0, block.begin(), block.end()});

  while (!worklist.empty()) {
    auto &[phase, opIt, opEnd] = worklist.back();
    if (opIt == opEnd) {
      worklist.pop_back();
      continue;
    }

    Operation *op = &*opIt;
    ++opIt;
    auto phaseIt = analysis.opPhases.find(op);
    if (phaseIt == analysis.opPhases.end()) {
      LLVM_DEBUG(llvm::dbgs() << "! No phase for op: " << *op << "\n");
      continue;
    }
    int16_t opPhase = phaseIt->second;
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

  //===--------------------------------------------------------------------===//
  // Record Own Args/Returns Before Cross-Phase Threading
  //===--------------------------------------------------------------------===//

  SmallVector<unsigned> numOwnBodyArgs(maxPhase - minPhase + 1, 0);
  SmallVector<unsigned> numOwnReturnValues(maxPhase - minPhase + 1, 0);
  numOwnArgs.assign(maxPhase - minPhase + 1, 0);
  numOwnReturns.assign(maxPhase - minPhase + 1, 0);
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    numOwnBodyArgs[phase - minPhase] =
        splits[phase - minPhase].funcOp.getBody().front().getNumArguments();
    numOwnReturnValues[phase - minPhase] =
        splits[phase - minPhase].returnValues.size();
    numOwnArgs[phase - minPhase] = numOwnBodyArgs[phase - minPhase];
    numOwnReturns[phase - minPhase] = numOwnReturnValues[phase - minPhase];
  }

  //===--------------------------------------------------------------------===//
  // Cross-Phase Value Threading
  //
  // Walk each phase from latest to earliest. For each op that references a
  // value from an earlier phase, either clone it (if purely local) or create
  // a block arg and forward the value through the previous phase's return.
  //===--------------------------------------------------------------------===//

  OpBuilder builder(funcOp);
  DenseMap<Value, Value> mapping;
  SmallDenseSet<Operation *, 4> closedFuncs;
  for (int16_t phase = maxPhase; phase >= minPhase; --phase) {
    auto &split = splits[phase - minPhase];
    closedFuncs.insert(split.funcOp);

    auto &bodyRegion = split.funcOp.getBody();
    builder.setInsertionPointToEnd(&bodyRegion.back());
    ReturnOp::create(builder, funcOp.getLoc(), split.returnValues,
                     /*typeOfValues=*/ValueRange{});

    bodyRegion.walk([&](Operation *op) {
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
              auto &block = bodyRegion.front();
              OpBuilder cloneBuilder(&block, block.begin());
              IRMapping cloneMapping;
              clonePureOp(defOp, cloneBuilder, cloneMapping);
              value = cloneMapping.lookup(operand.get());
            } else {
              LLVM_DEBUG({
                llvm::dbgs() << "- Creating arg in phase " << phase << " for ";
                if (auto *defOp2 = operand.get().getDefiningOp())
                  llvm::dbgs() << "value from " << defOp2->getName();
                else
                  llvm::dbgs() << "block arg";
                llvm::dbgs() << "\n";
              });
              value = bodyRegion.front().addArgument(operand.get().getType(),
                                                     operand.get().getLoc());

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
    unsigned ownArgs = numOwnBodyArgs[phase - minPhase];
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
    unsigned prevOwnReturns = numOwnReturnValues[phase - 1 - minPhase];

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
// reconstructSignatures
//
// Build per-phase SignatureOps by resolving type values from the unified func's
// original signature. For each phase, own arg types are resolved from the
// signature's typeOfArgs (partitioned by argPhases), own result types from
// typeOfResults (partitioned by effectiveResultPhases). Pure ops are cloned
// transitively; unresolvable values (e.g., phi block args from multi-block
// signatures) fall back to OpaqueTypeOp.
//
// For phases with cross-phase context (phase > minPhase), type values that
// depend on earlier-phase values (e.g., `uint_type(%N)` where `%N` comes from
// a const arg) cannot be resolved from the unified signature alone. In these
// cases, we create a parallel `opaque_unpack` in the signature on the context
// block arg, and derive the type values from the body's `coerce_type` ops and
// return value type operands, cloning them into the signature with a mapping
// from body opaque_unpack results to signature opaque_unpack results.
//===----------------------------------------------------------------------===//

void PhaseSplitter::reconstructSignatures() {
  // Get the canonical SignatureOp from the unified func's (intact) signature.
  // For multi-block signatures, consolidate first so we have a single
  // terminator with all type values merged via block args.
  hir::consolidateSignatureTerminators(funcOp.getSignature());

  SignatureOp sigOp;
  funcOp.getSignature().walk([&](SignatureOp op) { sigOp = op; });
  assert(sigOp && "unified func must have a signature");
  auto sigTypeOfArgs = sigOp.getTypeOfArgs();
  auto sigTypeOfResults = sigOp.getTypeOfResults();

  // Map sig entry block args to indices in funcOp.getArgPhases().
  auto &sigEntry = funcOp.getSignature().front();

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];
    auto &sigRegion = split.funcOp.getSignature();
    auto &bodyBlock = split.funcOp.getBody().front();

    // Clear any existing sig blocks (from emplaceBlock during creation).
    while (!sigRegion.empty())
      sigRegion.front().erase();

    // Create a fresh sig block with args matching the body's entry block.
    auto *sigBlock = new Block();
    sigRegion.push_back(sigBlock);
    for (auto bodyArg : bodyBlock.getArguments())
      sigBlock->addArgument(bodyArg.getType(), bodyArg.getLoc());

    OpBuilder sigBuilder(sigBlock, sigBlock->begin());
    IRMapping sigMapping;

    // Map sig entry block args at this phase to the new sig block args.
    unsigned newArgIdx = 0;
    for (auto [idx, argPhase] : llvm::enumerate(funcOp.getArgPhases())) {
      if (static_cast<int16_t>(argPhase) == phase) {
        if (idx < sigEntry.getNumArguments())
          sigMapping.map(sigEntry.getArgument(idx),
                         sigBlock->getArgument(newArgIdx));
        ++newArgIdx;
      }
    }

    // Resolve a type value: try mapping → clone pure → trace phi → opaque.
    auto resolveType = [&](Value val) -> Value {
      SmallPtrSet<Value, 4> visited;
      return resolveTypeValue(val, sigBuilder, sigMapping, funcOp.getLoc(),
                              visited);
    };

    // # Cross-Phase Signature Type Derivation
    //
    // For phases with a cross-phase context (phase > minPhase), the body has
    // an `opaque_unpack` that extracts cross-phase values from the context
    // arg. We create a matching `opaque_unpack` in the signature and build a
    // body-to-sig mapping. When the unified signature's type resolution fails
    // (falls back to OpaqueTypeOp because a type depends on an earlier-phase
    // value), we instead derive the type from the body:
    //
    // - Arg types: from the body's `coerce_type %arg, %type_val` ops
    // - Result types: from the return values' defining ops' type operands
    //   (e.g., `add`'s resultType, `coerce_type`'s typeOperand, `call`'s
    //   typeOfResults)
    //
    // The cloned type ops in the sig then correctly reference sig
    // opaque_unpack results. When SpecializeFuncs later expands the opaque
    // context, these become concrete `mir_constant` values, producing correct
    // types in the final MIR function signature.

    OpaqueUnpackOp bodyUnpackOp;
    IRMapping bodyToSig;
    if (phase > minPhase) {
      // Find the body's opaque_unpack consuming the context arg.
      for (auto &op : bodyBlock) {
        if (auto u = dyn_cast<OpaqueUnpackOp>(&op)) {
          bodyUnpackOp = u;
          break;
        }
      }
      if (bodyUnpackOp && bodyUnpackOp.getNumResults() > 0) {
        // Create a matching opaque_unpack in the signature.
        auto ctxSigArg = sigBlock->getArgument(sigBlock->getNumArguments() - 1);
        auto sigUnpackOp = OpaqueUnpackOp::create(
            sigBuilder, funcOp.getLoc(),
            SmallVector<Type>(bodyUnpackOp.getNumResults(),
                              AnyType::get(sigBuilder.getContext())),
            ctxSigArg);

        // Map body opaque_unpack results to sig opaque_unpack results, and
        // body own args to sig own args.
        for (unsigned i = 0; i < bodyUnpackOp.getNumResults(); ++i)
          bodyToSig.map(bodyUnpackOp.getResult(i), sigUnpackOp.getResult(i));
        unsigned ownArgCount = numOwnArgs[phase - minPhase];
        for (unsigned i = 0; i < ownArgCount; ++i)
          bodyToSig.map(bodyBlock.getArgument(i), sigBlock->getArgument(i));
      }
    }

    // Clone a type op tree from the body into the signature, using the
    // body-to-sig mapping for cross-phase values (opaque_unpack results and
    // block args).
    auto cloneBodyTypeIntoSig = [&](Value bodyTypeVal) -> Value {
      if (auto mapped = bodyToSig.lookupOrNull(bodyTypeVal))
        return mapped;
      auto *typeDefOp = bodyTypeVal.getDefiningOp();
      if (!typeDefOp)
        return {};
      SmallVector<Operation *> toClone;
      SmallPtrSet<Operation *, 8> visited;
      std::function<void(Operation *)> collectOps = [&](Operation *op) {
        if (!visited.insert(op).second)
          return;
        for (auto operand : op->getOperands()) {
          if (auto *defOp = operand.getDefiningOp())
            if (!bodyToSig.contains(operand))
              collectOps(defOp);
        }
        toClone.push_back(op);
      };
      collectOps(typeDefOp);
      IRMapping cloneMapping(bodyToSig);
      for (auto *op : toClone) {
        if (!cloneMapping.contains(op->getResult(0)))
          sigBuilder.clone(*op, cloneMapping);
      }
      return cloneMapping.lookupOrDefault(bodyTypeVal);
    };

    // Extract the type-level SSA value describing a body value's HIR type.
    auto getBodyTypeOfValue = [](Value val) -> Value {
      auto *defOp = val.getDefiningOp();
      if (!defOp)
        return {};
      if (auto coerce = dyn_cast<CoerceTypeOp>(defOp))
        return coerce.getTypeOperand();
      // All HIRBinaryOp subclasses have (lhs, rhs, resultType) operands.
      if (isa<AddOp, SubOp, MulOp, DivOp, ModOp, AndOp, OrOp, XorOp, ShlOp,
              ShrOp, EqOp, NeqOp, LtOp, GtOp, GeqOp, LeqOp>(defOp))
        return defOp->getOperand(2);
      if (auto call = dyn_cast<CallOp>(defOp)) {
        auto idx = cast<OpResult>(val).getResultNumber();
        auto typeOfResults = call.getTypeOfResults();
        if (idx < typeOfResults.size())
          return typeOfResults[idx];
      }
      return {};
    };

    // Build typeOfArgs: resolve own arg types, opaque for context.
    SmallVector<Value> typeOfArgs;
    unsigned ownArgIdx = 0;
    for (auto [idx, argPhase] : llvm::enumerate(funcOp.getArgPhases())) {
      if (static_cast<int16_t>(argPhase) != phase)
        continue;
      Value resolved = resolveType(sigTypeOfArgs[idx]);

      // If resolution fell back to opaque_type and we have a body
      // opaque_unpack, derive the type from the body's coerce_type instead.
      if (bodyUnpackOp && resolved.getDefiningOp<OpaqueTypeOp>() &&
          ownArgIdx < numOwnArgs[phase - minPhase]) {
        auto bodyArg = bodyBlock.getArgument(ownArgIdx);
        for (auto *user : bodyArg.getUsers()) {
          auto coerce = dyn_cast<CoerceTypeOp>(user);
          if (coerce && coerce.getInput() == bodyArg) {
            if (auto sigType = cloneBodyTypeIntoSig(coerce.getTypeOperand())) {
              resolved.getDefiningOp()->erase();
              resolved = sigType;
            }
            break;
          }
        }
      }

      typeOfArgs.push_back(resolved);
      ++ownArgIdx;
    }
    if (bodyBlock.getNumArguments() > typeOfArgs.size())
      typeOfArgs.push_back(
          OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult());

    // Build typeOfResults: resolve own result types, opaque for context.
    SmallVector<Value> typeOfResults;
    ReturnOp bodyRetOp;
    split.funcOp.getBody().walk([&](ReturnOp r) {
      if (!bodyRetOp)
        bodyRetOp = r;
    });
    unsigned ownResultIdx = 0;
    for (auto [idx, resultPhase] : llvm::enumerate(effectiveResultPhases)) {
      if (resultPhase != phase)
        continue;
      Value resolved = resolveType(sigTypeOfResults[idx]);

      // If resolution fell back to opaque_type, derive from the body's
      // return value type operand.
      if (bodyUnpackOp && resolved.getDefiningOp<OpaqueTypeOp>() && bodyRetOp &&
          ownResultIdx < bodyRetOp.getValues().size()) {
        Value bodyType =
            getBodyTypeOfValue(bodyRetOp.getValues()[ownResultIdx]);
        if (bodyType) {
          if (auto sigType = cloneBodyTypeIntoSig(bodyType)) {
            resolved.getDefiningOp()->erase();
            resolved = sigType;
          }
        }
      }

      typeOfResults.push_back(resolved);
      ++ownResultIdx;
    }
    if (bodyRetOp && bodyRetOp.getValues().size() > typeOfResults.size())
      typeOfResults.push_back(
          OpaqueTypeOp::create(sigBuilder, funcOp.getLoc()).getResult());

    sigBuilder.setInsertionPointToEnd(sigBlock);
    SignatureOp::create(sigBuilder, funcOp.getLoc(), typeOfArgs, typeOfResults);
  }
}

//===----------------------------------------------------------------------===//
// populateReturnTypeOfValues
//
// After signatures are reconstructed, walk each phase function's returns and
// populate their typeOfValues from the signature's typeOfResults. This clones
// the signature's type ops into the body region so that each return value has
// a corresponding type-level value.
//===----------------------------------------------------------------------===//

void PhaseSplitter::populateReturnTypeOfValues() {
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splits[phase - minPhase];

    // Get the signature's typeOfResults.
    SignatureOp sigOp;
    split.funcOp.getSignature().walk([&](SignatureOp op) { sigOp = op; });
    if (!sigOp)
      continue;
    auto sigResultTypes = sigOp.getTypeOfResults();

    // Build a mapping from signature block args to body block args.
    auto &sigBlock = split.funcOp.getSignature().front();
    auto &bodyBlock = split.funcOp.getBody().front();
    IRMapping mapping;
    for (auto [sigArg, bodyArg] :
         llvm::zip(sigBlock.getArguments(), bodyBlock.getArguments()))
      mapping.map(sigArg, bodyArg);

    // Map sig opaque_unpack results to existing body opaque_unpack results.
    // Without this, cloning the sig's type ops (which may reference sig
    // opaque_unpack results) would create duplicate opaque_unpack ops in the
    // body.
    if (phase > minPhase) {
      OpaqueUnpackOp sigUnpack, bodyUnpack;
      for (auto &op : sigBlock)
        if (auto u = dyn_cast<OpaqueUnpackOp>(&op)) {
          sigUnpack = u;
          break;
        }
      for (auto &op : bodyBlock)
        if (auto u = dyn_cast<OpaqueUnpackOp>(&op)) {
          bodyUnpack = u;
          break;
        }
      if (sigUnpack && bodyUnpack)
        for (auto [sigRes, bodyRes] :
             llvm::zip(sigUnpack.getResults(), bodyUnpack.getResults()))
          mapping.map(sigRes, bodyRes);
    }

    split.funcOp.getBody().walk([&](ReturnOp retOp) {
      SmallVector<Value> typeOfValues;
      OpBuilder bodyBuilder(retOp);

      for (auto sigResultType : sigResultTypes) {
        // Clone the type op (and its transitive dependencies) into the body.
        auto *typeDefOp = sigResultType.getDefiningOp();
        if (!typeDefOp) {
          // Block arg — already mapped.
          typeOfValues.push_back(mapping.lookupOrDefault(sigResultType));
          continue;
        }

        // Post-order DFS clone of the defining op chain.
        SmallVector<Operation *> toClone;
        SmallPtrSet<Operation *, 8> visited;
        std::function<void(Operation *)> collectOps = [&](Operation *op) {
          if (!visited.insert(op).second)
            return;
          for (auto operand : op->getOperands()) {
            if (auto *defOp = operand.getDefiningOp())
              if (defOp->getParentRegion() == &split.funcOp.getSignature())
                collectOps(defOp);
          }
          toClone.push_back(op);
        };
        collectOps(typeDefOp);

        IRMapping cloneMapping(mapping);
        for (auto *op : toClone) {
          if (!cloneMapping.contains(op->getResult(0)))
            bodyBuilder.clone(*op, cloneMapping);
        }
        typeOfValues.push_back(cloneMapping.lookupOrDefault(sigResultType));
      }

      retOp.getTypeOfValuesMutable().assign(typeOfValues);
    });
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
