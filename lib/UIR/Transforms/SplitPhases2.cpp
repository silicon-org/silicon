//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # SplitPhases2: Split UIR Unified Functions into Per-Phase HIR Functions
//
// This pass runs phase analysis on each `uir.func`, then uses the resulting
// phase map to split each unified function into per-phase `hir.func` ops.
// It produces `uir.split_func` witnesses recording how each function was
// split, and `hir.multiphase_func` structural ops for groups of consecutive
// internal phases.
//
// The algorithm follows `docs/design/phase-splits.md`:
// 1. Determine phase range from the analysis + declared arg/result phases.
// 2. Group phases by external visibility (arg/result phases are boundaries).
// 3. Create per-phase `hir.func` ops.
// 4. Distribute ops into their phase functions (calls decomposed on-the-fly).
// 5. Dissolve `uir.expr` and `uir.pin` ops.
// 6. Fix up cross-phase value references via opaque context threading.
// 7. Reconstruct signatures for each phase function.
// 8. Emit structural ops (`uir.split_func`, `hir.multiphase_func`).
//
// Functions are processed in topological order (callees before callers) so
// that `uir.split_func` witnesses exist when decomposing caller `uir.call`
// ops.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "split-phases2"

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_SPLITPHASES2PASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

//===----------------------------------------------------------------------===//
// Data Structures
//===----------------------------------------------------------------------===//

namespace {

/// Per-phase split: the `hir.func` for this phase and tracked return values.
struct PhaseSplit {
  hir::FuncOp funcOp;
  int16_t phase = 0;
  SmallVector<Value> returnValues;
  SmallVector<Value> returnTypeOfValues;
};

/// A group of consecutive phases, ending at an externally visible phase.
struct PhaseGroup {
  SmallVector<int16_t> phases;
};

//===----------------------------------------------------------------------===//
// PhaseSplitter2
//===----------------------------------------------------------------------===//

/// Splits a single `uir.func` into per-phase `hir.func` ops.
struct PhaseSplitter2 {
  PhaseSplitter2(PhaseAnalysis &analysis, SymbolTable &symbolTable)
      : analysis(analysis), symbolTable(symbolTable), funcOp(analysis.funcOp) {}

  LogicalResult run();

private:
  PhaseAnalysis &analysis;
  SymbolTable &symbolTable;
  FuncOp funcOp;

  int16_t minPhase = 0;
  int16_t maxPhase = 0;
  SmallVector<PhaseSplit> splits;
  SmallVector<PhaseGroup> groups;
  DenseSet<int16_t> externalPhases;
  SmallVector<int16_t> effectiveResultPhases;

  /// Cloned unified signature, moved into the split_func witness.
  Region clonedUnifiedSig;

  /// Get the PhaseSplit for a given phase number.
  PhaseSplit &splitFor(int16_t phase) {
    assert(phase >= minPhase && phase <= maxPhase);
    return splits[phase - minPhase];
  }

  void determinePhaseRange();
  void computeExternalPhases();
  void groupPhases();
  void createPhaseFunctions();
  LogicalResult splitBodyByPhase();
  void distributeOps(Block &block);
  void dissolveExprsAndPins();
  int16_t findValuePhase(Value value);
  void fixupCrossPhaseRefs();
  void reconstructSignatures();
  void createReturnOps();
  void emitStructuralOps();
};

} // namespace

//===----------------------------------------------------------------------===//
// Phase Range and Grouping
//===----------------------------------------------------------------------===//

/// Scan the phase map for the range of phases present, extended by declared
/// arg/result phases and call arg/result absolute phases.
void PhaseSplitter2::determinePhaseRange() {
  minPhase = 0;
  maxPhase = 0;

  // Extend with declared arg and result phases.
  for (auto p : funcOp.getArgPhases()) {
    minPhase = std::min(minPhase, static_cast<int16_t>(p));
    maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
  }
  for (auto p : funcOp.getResultPhases()) {
    minPhase = std::min(minPhase, static_cast<int16_t>(p));
    maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
  }

  // Extend with phases from the analysis, skipping floating constants.
  for (auto [op, phase] : analysis.opPhases) {
    if (phase == INT16_MIN)
      continue;
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  }

  // Extend with absolute call phases: callOpPhase + per-arg/result offset.
  funcOp.getBody().walk([&](CallOp callOp) {
    int16_t callPhase = analysis.getPhase(callOp);
    if (callPhase == INT16_MIN)
      return;
    for (auto offset : callOp.getArgPhases()) {
      int16_t abs = callPhase + static_cast<int16_t>(offset);
      minPhase = std::min(minPhase, abs);
      maxPhase = std::max(maxPhase, abs);
    }
    for (auto offset : callOp.getResultPhases()) {
      int16_t abs = callPhase + static_cast<int16_t>(offset);
      minPhase = std::min(minPhase, abs);
      maxPhase = std::max(maxPhase, abs);
    }
  });

  LLVM_DEBUG(llvm::dbgs() << "Phase range for @" << funcOp.getSymName() << ": ["
                          << minPhase << ", " << maxPhase << "]\n");
}

/// External phases are the union of declared arg phases and effective result
/// phases.
void PhaseSplitter2::computeExternalPhases() {
  for (auto p : funcOp.getArgPhases())
    externalPhases.insert(static_cast<int16_t>(p));
  for (auto p : funcOp.getResultPhases())
    externalPhases.insert(static_cast<int16_t>(p));
  // Phase 0 is always external (the default body phase).
  externalPhases.insert(0);
}

/// Group consecutive phases. A group boundary occurs at each external phase.
void PhaseSplitter2::groupPhases() {
  PhaseGroup current;
  for (int16_t p = minPhase; p <= maxPhase; ++p) {
    current.phases.push_back(p);
    if (externalPhases.contains(p)) {
      groups.push_back(std::move(current));
      current = {};
    }
  }
  // Trailing internal phases form their own group.
  if (!current.phases.empty())
    groups.push_back(std::move(current));
}

//===----------------------------------------------------------------------===//
// Per-Phase Function Creation
//===----------------------------------------------------------------------===//

/// Create an `hir.func` for each phase. Names follow the convention:
/// `@name.G` for single-phase groups, `@name.Ga`/`@name.Gb` for multi-phase.
void PhaseSplitter2::createPhaseFunctions() {
  OpBuilder builder(funcOp);
  auto privateAttr = builder.getStringAttr("private");

  splits.resize(maxPhase - minPhase + 1);

  // Create in reverse order so that insertion before the unified func
  // puts them in the right order.
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
      auto isModuleAttr =
          (phase == maxPhase) ? funcOp.getIsModuleAttr() : UnitAttr{};
      auto phaseFuncOp =
          hir::FuncOp::create(builder, funcOp.getLoc(), name, privateAttr,
                              emptyArray, emptyArray, isModuleAttr);
      phaseFuncOp.getSignature().emplaceBlock();
      if (phase != 0)
        phaseFuncOp.getBody().emplaceBlock();
      symbolTable.insert(phaseFuncOp);
      builder.setInsertionPoint(phaseFuncOp);

      auto &split = splitFor(phase);
      split.funcOp = phaseFuncOp;
      split.phase = phase;
    }
  }
}

//===----------------------------------------------------------------------===//
// Body Splitting
//===----------------------------------------------------------------------===//

/// The core splitting algorithm. Moves the body into phase 0, distributes
/// ops, dissolves expr/pin, fixes cross-phase refs, creates returns.
LogicalResult PhaseSplitter2::splitBodyByPhase() {
  // Move the unified func body into phase 0's function.
  auto &phase0Body = splitFor(0).funcOp.getBody();
  phase0Body.takeBody(funcOp.getBody());

  auto &entryBlock = phase0Body.front();
  auto argPhases = funcOp.getArgPhases();
  auto anyTy = hir::AnyType::get(funcOp.getContext());

  // Move shifted block args to their phase functions first, before capturing
  // return values (so that RAUW updates the return op's operands).
  // Process in reverse order so that erasing block args doesn't invalidate
  // indices.
  for (int idx = argPhases.size() - 1; idx >= 0; --idx) {
    int16_t phase = static_cast<int16_t>(argPhases[idx]);
    if (phase == 0)
      continue;
    auto &targetBody = splitFor(phase).funcOp.getBody();
    auto *targetBlock = &targetBody.front();
    auto newArg = targetBlock->addArgument(anyTy, funcOp.getLoc());
    auto oldArg = entryBlock.getArgument(idx);
    oldArg.replaceAllUsesWith(newArg);
    entryBlock.eraseArgument(idx);
  }

  // Now capture return values (the return op's operands have been updated by
  // the RAUW above, so cross-phase references are already resolved to the
  // new block args in the target phase functions).
  auto returnOp = cast<ReturnOp>(entryBlock.getTerminator());
  auto resultPhases = funcOp.getResultPhases();
  effectiveResultPhases.clear();
  for (auto p : resultPhases)
    effectiveResultPhases.push_back(static_cast<int16_t>(p));

  for (auto [i, value] : llvm::enumerate(returnOp.getValues())) {
    int16_t phase = effectiveResultPhases[i];
    splitFor(phase).returnValues.push_back(value);
  }
  for (auto [i, typeVal] : llvm::enumerate(returnOp.getTypeOfValues())) {
    int16_t phase = effectiveResultPhases[i];
    splitFor(phase).returnTypeOfValues.push_back(typeVal);
  }

  // Erase the unified return op (we'll create hir.return ops later).
  returnOp.erase();

  // Distribute ops to their phase functions.
  distributeOps(entryBlock);

  // Dissolve uir.expr and uir.pin ops.
  dissolveExprsAndPins();

  // Fix up cross-phase value references through opaque context.
  fixupCrossPhaseRefs();

  // Create hir.return terminators for each phase function.
  createReturnOps();

  return success();
}

//===----------------------------------------------------------------------===//
// Op Distribution
//===----------------------------------------------------------------------===//

/// Distribute ops in a block to their assigned phase functions. Calls are
/// decomposed on-the-fly. Descends into expr/if/loop regions recursively.
void PhaseSplitter2::distributeOps(Block &block) {
  // Collect ops to process (avoid iterator invalidation).
  SmallVector<Operation *> ops;
  for (auto &op : block)
    ops.push_back(&op);

  for (auto *op : ops) {
    // Skip the terminator (already erased or will be handled separately).
    if (op->hasTrait<OpTrait::IsTerminator>())
      continue;

    auto it = analysis.opPhases.find(op);
    if (it == analysis.opPhases.end())
      continue;
    int16_t phase = it->second;

    // Floating constants (INT16_MIN) go to minPhase.
    if (phase == INT16_MIN)
      phase = minPhase;

    // Descend into expr regions to distribute their children first.
    if (auto exprOp = dyn_cast<ExprOp>(op)) {
      distributeOps(exprOp.getBody().front());
    }

    // Descend into structured CF regions to find nested expr/pin.
    if (auto ifOp = dyn_cast<IfOp>(op)) {
      distributeOps(ifOp.getThenRegion().front());
      if (!ifOp.getElseRegion().empty())
        distributeOps(ifOp.getElseRegion().front());
    }
    if (auto loopOp = dyn_cast<LoopOp>(op)) {
      distributeOps(loopOp.getBody().front());
    }

    // TODO: Decompose uir.call into per-phase hir.call ops on-the-fly.

    // Move the op to the target phase function if needed.
    if (phase < minPhase)
      phase = minPhase;
    if (phase > maxPhase)
      phase = maxPhase;

    auto &targetBody = splitFor(phase).funcOp.getBody();
    auto *targetBlock = &targetBody.front();
    if (op->getBlock() != targetBlock)
      op->moveBefore(targetBlock, targetBlock->end());
  }
}

//===----------------------------------------------------------------------===//
// Expr/Pin Dissolution
//===----------------------------------------------------------------------===//

/// Dissolve all `uir.expr` and `uir.pin` ops. Expr bodies are inlined into
/// the parent block, with results replaced by yield operands. Pin outputs
/// are replaced by their inputs.
void PhaseSplitter2::dissolveExprsAndPins() {
  // Process all phase functions.
  for (auto &split : splits) {
    if (!split.funcOp)
      continue;
    auto &body = split.funcOp.getBody();

    // Dissolve exprs (post-order: inner first).
    SmallVector<ExprOp> exprs;
    body.walk([&](ExprOp op) { exprs.push_back(op); });
    for (auto exprOp : exprs) {
      auto &exprBody = exprOp.getBody().front();
      auto yieldOp = cast<YieldOp>(exprBody.getTerminator());

      // Replace expr results with yield values.
      for (auto [result, yieldVal] :
           llvm::zip(exprOp.getResults(), yieldOp.getValues()))
        result.replaceAllUsesWith(yieldVal);

      // Erase the yield, then inline the body into the parent block.
      yieldOp.erase();
      auto *parentBlock = exprOp->getBlock();
      parentBlock->getOperations().splice(exprOp->getIterator(),
                                          exprBody.getOperations());
      exprOp.erase();
    }

    // Dissolve pins.
    SmallVector<PinOp> pins;
    body.walk([&](PinOp op) { pins.push_back(op); });
    for (auto pinOp : pins) {
      for (auto [output, input] :
           llvm::zip(pinOp.getOutputs(), pinOp.getInputs()))
        output.replaceAllUsesWith(input);
      pinOp.erase();
    }
  }
}

//===----------------------------------------------------------------------===//
// Cross-Phase Fixup
//===----------------------------------------------------------------------===//

/// Determine which phase function a value belongs to. Returns the phase
/// number, or INT16_MIN if the value is not in any phase function.
int16_t PhaseSplitter2::findValuePhase(Value value) {
  // For block args, check the parent region's parent op.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *parentOp = blockArg.getOwner()->getParentOp();
    if (auto func = dyn_cast<hir::FuncOp>(parentOp)) {
      for (int16_t p = minPhase; p <= maxPhase; ++p)
        if (splitFor(p).funcOp == func)
          return p;
    }
    return INT16_MIN;
  }
  // For op results, check the defining op's parent func.
  auto *defOp = value.getDefiningOp();
  if (!defOp)
    return INT16_MIN;
  auto parentFunc = defOp->getParentOfType<hir::FuncOp>();
  if (!parentFunc)
    return INT16_MIN;
  for (int16_t p = minPhase; p <= maxPhase; ++p)
    if (splitFor(p).funcOp == parentFunc)
      return p;
  return INT16_MIN;
}

/// After all ops are distributed, scan each phase function for values defined
/// in a different phase. Thread them through the opaque context chain.
void PhaseSplitter2::fixupCrossPhaseRefs() {
  if (minPhase == maxPhase)
    return; // Single phase, nothing to fix.

  auto anyTy = hir::AnyType::get(funcOp.getContext());
  auto loc = funcOp.getLoc();

  // Collect all cross-phase value references. A value is cross-phase if it is
  // defined in one phase function but used (or referenced in return values) in
  // a different phase function.
  // Map: value → set of target phases where it's needed.
  DenseMap<Value, DenseSet<int16_t>> valueTargetPhases;

  auto markIfCrossPhase = [&](Value val, int16_t usingPhase) {
    int16_t srcPhase = findValuePhase(val);
    if (srcPhase == INT16_MIN || srcPhase == usingPhase)
      return;
    valueTargetPhases[val].insert(usingPhase);
  };

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splitFor(phase);
    if (!split.funcOp)
      continue;

    // Check operands of all ops in this phase function.
    split.funcOp.getBody().walk([&](Operation *op) {
      for (auto operand : op->getOperands())
        markIfCrossPhase(operand, phase);
    });

    // Check return values/types.
    for (auto val : split.returnValues)
      markIfCrossPhase(val, phase);
    for (auto val : split.returnTypeOfValues)
      markIfCrossPhase(val, phase);
  }

  if (valueTargetPhases.empty())
    return;

  // For each phase boundary, collect values that need to cross it.
  DenseMap<int16_t, SmallVector<Value>> boundaryValues;

  // Separate trivially materializable ops (clone instead of threading).
  for (auto &[value, targets] : valueTargetPhases) {
    auto *defOp = value.getDefiningOp();
    if (defOp && defOp->hasTrait<OpTrait::ConstantLike>()) {
      for (auto targetPhase : targets) {
        auto &targetBlock = splitFor(targetPhase).funcOp.getBody().front();
        OpBuilder targetBuilder(&targetBlock, targetBlock.begin());
        auto *cloned = targetBuilder.clone(*defOp);
        auto targetFunc = splitFor(targetPhase).funcOp;
        value.replaceUsesWithIf(cloned->getResult(0), [targetFunc](
                                                          OpOperand &use) {
          return use.getOwner()->getParentOfType<hir::FuncOp>() == targetFunc;
        });
        for (auto &rv : splitFor(targetPhase).returnValues)
          if (rv == value)
            rv = cloned->getResult(0);
        for (auto &rv : splitFor(targetPhase).returnTypeOfValues)
          if (rv == value)
            rv = cloned->getResult(0);
      }
      continue;
    }

    int16_t sourcePhase = findValuePhase(value);
    int16_t maxTarget = *std::max_element(targets.begin(), targets.end());
    for (int16_t p = sourcePhase; p < maxTarget; ++p)
      boundaryValues[p].push_back(value);
  }

  // For each boundary, create opaque_pack at source and opaque_unpack at dest.
  // Track value → unpacked replacement at each phase.
  DenseMap<int16_t, DenseMap<Value, Value>> phaseValueMapping;

  for (int16_t p = minPhase; p < maxPhase; ++p) {
    auto &values = boundaryValues[p];
    if (values.empty())
      continue;

    auto &srcSplit = splitFor(p);
    auto &dstSplit = splitFor(p + 1);
    if (!srcSplit.funcOp || !dstSplit.funcOp)
      continue;

    // Resolve values: use the mapping if already unpacked from a prior hop.
    SmallVector<Value> resolvedValues;
    for (auto val : values) {
      auto it = phaseValueMapping[p].find(val);
      resolvedValues.push_back(it != phaseValueMapping[p].end() ? it->second
                                                                : val);
    }

    // Create opaque_pack at end of source phase body.
    auto &srcBlock = srcSplit.funcOp.getBody().front();
    OpBuilder srcBuilder(&srcBlock, srcBlock.end());
    auto packOp =
        hir::OpaquePackOp::create(srcBuilder, loc, anyTy, resolvedValues);
    srcSplit.returnValues.push_back(packOp.getResult());
    auto opaqueTypeOp = hir::OpaqueTypeOp::create(srcBuilder, loc);
    srcSplit.returnTypeOfValues.push_back(opaqueTypeOp.getResult());

    // Create block arg + opaque_unpack at start of destination phase body.
    auto &dstBlock = dstSplit.funcOp.getBody().front();
    auto ctxArg = dstBlock.addArgument(anyTy, loc);
    OpBuilder dstBuilder(&dstBlock, dstBlock.begin());
    auto unpackOp = hir::OpaqueUnpackOp::create(
        dstBuilder, loc, SmallVector<Type>(values.size(), anyTy), ctxArg);

    for (auto [i, val] : llvm::enumerate(values))
      phaseValueMapping[p + 1][val] = unpackOp.getResult(i);
  }

  // Replace all cross-phase uses with unpacked values.
  for (auto &[value, targets] : valueTargetPhases) {
    // Skip values already handled by cloning (ConstantLike).
    if (value.getDefiningOp() &&
        value.getDefiningOp()->hasTrait<OpTrait::ConstantLike>())
      continue;

    for (auto targetPhase : targets) {
      auto it = phaseValueMapping[targetPhase].find(value);
      if (it == phaseValueMapping[targetPhase].end())
        continue;
      Value replacement = it->second;
      auto targetFunc = splitFor(targetPhase).funcOp;
      value.replaceUsesWithIf(replacement, [targetFunc](OpOperand &use) {
        return use.getOwner()->getParentOfType<hir::FuncOp>() == targetFunc;
      });
      for (auto &rv : splitFor(targetPhase).returnValues)
        if (rv == value)
          rv = replacement;
      for (auto &rv : splitFor(targetPhase).returnTypeOfValues)
        if (rv == value)
          rv = replacement;
    }
  }
}

//===----------------------------------------------------------------------===//
// Signature Reconstruction
//===----------------------------------------------------------------------===//

/// Build `hir.signature` ops for each phase function.
void PhaseSplitter2::reconstructSignatures() {
  auto anyTy = hir::AnyType::get(funcOp.getContext());
  auto loc = funcOp.getLoc();

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splitFor(phase);
    if (!split.funcOp)
      continue;

    auto &sigBlock = split.funcOp.getSignature().front();
    OpBuilder sigBuilder(&sigBlock, sigBlock.end());

    auto &bodyBlock = split.funcOp.getBody().front();
    unsigned numBodyArgs = bodyBlock.getNumArguments();

    // Add block args to the signature block to match the body.
    for (unsigned i = 0; i < numBodyArgs; ++i)
      sigBlock.addArgument(anyTy, loc);

    // For now, use opaque_type for all arg/result types. This is a
    // simplification; proper signature reconstruction will derive type
    // values from the unified signature or the opaque context.
    // TODO: Derive type values from the unified signature for own args,
    // and from the opaque context for cross-phase args.
    SmallVector<Value> sigTypeOfArgs;
    for (unsigned i = 0; i < numBodyArgs; ++i) {
      auto opaqueType = hir::OpaqueTypeOp::create(sigBuilder, loc);
      sigTypeOfArgs.push_back(opaqueType.getResult());
    }

    SmallVector<Value> sigTypeOfResults;
    for (unsigned i = 0; i < split.returnValues.size(); ++i) {
      auto opaqueType = hir::OpaqueTypeOp::create(sigBuilder, loc);
      sigTypeOfResults.push_back(opaqueType.getResult());
    }

    hir::SignatureOp::create(sigBuilder, loc, sigTypeOfArgs, sigTypeOfResults);
  }
}

//===----------------------------------------------------------------------===//
// Return Op Creation
//===----------------------------------------------------------------------===//

/// Create `hir.return` terminators for each phase function.
void PhaseSplitter2::createReturnOps() {
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splitFor(phase);
    if (!split.funcOp)
      continue;

    auto &block = split.funcOp.getBody().front();
    OpBuilder builder(&block, block.end());
    hir::ReturnOp::create(builder, funcOp.getLoc(), split.returnValues,
                          split.returnTypeOfValues);
  }
}

//===----------------------------------------------------------------------===//
// Structural Op Emission
//===----------------------------------------------------------------------===//

/// Emit `uir.split_func` and `hir.multiphase_func` structural ops.
void PhaseSplitter2::emitStructuralOps() {
  OpBuilder builder(funcOp);
  auto loc = funcOp.getLoc();

  // Set arg and result names on phase functions.
  auto argPhases = funcOp.getArgPhases();
  auto argNames = funcOp.getArgNames();
  auto resultPhases = funcOp.getResultPhases();
  auto resultNames = funcOp.getResultNames();

  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    auto &split = splitFor(phase);
    if (!split.funcOp)
      continue;

    // Collect arg names for this phase.
    SmallVector<Attribute> phaseArgNames;
    for (auto [i, ap] : llvm::enumerate(argPhases)) {
      if (static_cast<int16_t>(ap) == phase)
        phaseArgNames.push_back(argNames[i]);
    }
    // Context args are named "ctx".
    unsigned numBodyArgs = split.funcOp.getBody().front().getNumArguments();
    while (phaseArgNames.size() < numBodyArgs)
      phaseArgNames.push_back(builder.getStringAttr("ctx"));
    split.funcOp.setArgNamesAttr(builder.getArrayAttr(phaseArgNames));

    // Collect result names.
    SmallVector<Attribute> phaseResultNames;
    for (auto [i, rp] : llvm::enumerate(resultPhases)) {
      if (static_cast<int16_t>(rp) == phase)
        phaseResultNames.push_back(resultNames[i]);
    }
    // Context result.
    if (split.returnValues.size() > phaseResultNames.size())
      phaseResultNames.push_back(builder.getStringAttr("ctx"));
    split.funcOp.setResultNamesAttr(builder.getArrayAttr(phaseResultNames));
  }

  // Build phase map entries for the split_func.
  SmallVector<int32_t> phaseNumbers;
  SmallVector<Attribute> phaseFuncs;

  for (auto &group : groups) {
    int16_t externalPhase =
        group.phases.back(); // External phase is the last in the group.
    phaseNumbers.push_back(externalPhase);

    if (group.phases.size() == 1) {
      // Single-phase group: reference the hir.func directly.
      phaseFuncs.push_back(
          FlatSymbolRefAttr::get(splitFor(externalPhase).funcOp));
    } else {
      // Multi-phase group: create a hir.multiphase_func.
      SmallVector<Attribute> subFuncRefs;
      SmallVector<Attribute> mpArgNames;
      SmallVector<bool> mpArgIsFirst;
      SmallVector<Attribute> mpResultNames;

      for (auto phase : group.phases)
        subFuncRefs.push_back(FlatSymbolRefAttr::get(splitFor(phase).funcOp));

      // Args for the first sub-phase are "first", args for the last are "last".
      int16_t firstPhase = group.phases.front();
      int16_t lastPhase = group.phases.back();

      // First-phase args (context from prior group).
      if (firstPhase != minPhase) {
        mpArgNames.push_back(builder.getStringAttr("ctx"));
        mpArgIsFirst.push_back(true);
      }

      // Last-phase args (user-visible).
      for (auto [i, ap] : llvm::enumerate(argPhases)) {
        if (static_cast<int16_t>(ap) == lastPhase) {
          mpArgNames.push_back(argNames[i]);
          mpArgIsFirst.push_back(false);
        }
      }

      // Results from the last sub-phase.
      mpResultNames = SmallVector<Attribute>(
          splitFor(lastPhase).funcOp.getResultNames().begin(),
          splitFor(lastPhase).funcOp.getResultNames().end());

      auto mpName = builder.getStringAttr(funcOp.getSymName() + "." +
                                          Twine(&group - &groups[0]));
      auto mpOp = hir::MultiphaseFuncOp::create(
          builder, loc, mpName, /*visibility=*/StringAttr{},
          builder.getArrayAttr(mpArgNames), mpArgIsFirst,
          builder.getArrayAttr(mpResultNames),
          builder.getArrayAttr(subFuncRefs));
      symbolTable.insert(mpOp);
      builder.setInsertionPoint(mpOp);

      phaseFuncs.push_back(FlatSymbolRefAttr::get(mpOp));
    }
  }

  // Create the uir.split_func witness.
  // Move the cloned unified signature into it.
  auto splitFuncOp = uir::SplitFuncOp::create(
      builder, loc, funcOp.getSymName(), funcOp.getSymVisibilityAttr(),
      funcOp.getArgNames(),
      DenseI32ArrayAttr::get(funcOp.getContext(),
                             SmallVector<int32_t>(funcOp.getArgPhases().begin(),
                                                  funcOp.getArgPhases().end())),
      funcOp.getResultNames(),
      DenseI32ArrayAttr::get(
          funcOp.getContext(),
          SmallVector<int32_t>(funcOp.getResultPhases().begin(),
                               funcOp.getResultPhases().end())),
      DenseI32ArrayAttr::get(funcOp.getContext(), phaseNumbers),
      builder.getArrayAttr(phaseFuncs));

  // Move cloned unified signature into the split_func.
  splitFuncOp.getSignature().takeBody(clonedUnifiedSig);

  // Erase the unified func from the symbol table before inserting the
  // split_func, since they share the same symbol name.
  symbolTable.erase(funcOp);
  symbolTable.insert(splitFuncOp);
}

//===----------------------------------------------------------------------===//
// PhaseSplitter2::run
//===----------------------------------------------------------------------===//

LogicalResult PhaseSplitter2::run() {
  LLVM_DEBUG(llvm::dbgs() << "Splitting @" << funcOp.getSymName() << "\n");

  determinePhaseRange();
  computeExternalPhases();
  groupPhases();
  createPhaseFunctions();

  // Clone the unified signature before splitting consumes it.
  IRMapping sigCloneMapping;
  funcOp.getSignature().cloneInto(&clonedUnifiedSig, sigCloneMapping);

  if (failed(splitBodyByPhase()))
    return failure();

  reconstructSignatures();
  emitStructuralOps(); // Also erases the original unified func.

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {

struct SplitPhases2Pass
    : public uir::impl::SplitPhases2PassBase<SplitPhases2Pass> {
  using SplitPhases2PassBase::SplitPhases2PassBase;

  void runOnOperation() override {
    auto module = getOperation();
    auto &symbolTable = getAnalysis<SymbolTable>();

    // Collect all uir.func ops and run phase analysis.
    SmallVector<std::pair<FuncOp, std::unique_ptr<PhaseAnalysis>>> funcs;
    for (auto funcOp : module.getOps<FuncOp>()) {
      auto analysis = std::make_unique<PhaseAnalysis>(funcOp);
      if (failed(analysis->run()))
        return signalPassFailure();
      funcs.push_back({funcOp, std::move(analysis)});
    }

    // Build call graph for topological sorting.
    unsigned numFuncs = funcs.size();
    DenseMap<StringRef, unsigned> nameToIdx;
    for (auto [i, pair] : llvm::enumerate(funcs))
      nameToIdx[pair.first.getSymName()] = i;

    // Adjacency list: edges[i] = set of indices that func i calls.
    SmallVector<SmallVector<unsigned>> edges(numFuncs);
    for (unsigned idx = 0; idx < numFuncs; ++idx) {
      funcs[idx].first.getBody().walk([&](CallOp callOp) {
        auto it = nameToIdx.find(callOp.getCallee());
        if (it != nameToIdx.end())
          edges[idx].push_back(it->second);
      });
    }

    // Topological sort via post-order DFS. Detect cycles.
    SmallVector<unsigned> order;
    SmallVector<uint8_t> state(numFuncs, 0); // 0=unvisited, 1=visiting, 2=done
    bool hasCycle = false;

    std::function<void(unsigned)> visit = [&](unsigned idx) {
      if (state[idx] == 2)
        return;
      if (state[idx] == 1) {
        hasCycle = true;
        return;
      }
      state[idx] = 1;
      for (auto calleeIdx : edges[idx])
        visit(calleeIdx);
      state[idx] = 2;
      order.push_back(idx);
    };

    for (unsigned i = 0; i < numFuncs; ++i)
      visit(i);

    if (hasCycle) {
      // Emit error on the first function involved in a cycle.
      for (unsigned i = 0; i < numFuncs; ++i) {
        if (state[i] == 1) {
          funcs[i].first.emitError("recursive call cycle detected");
          break;
        }
      }
      return signalPassFailure();
    }

    // Process in topological order (callees first).
    for (auto idx : order) {
      auto &[funcOp, analysis] = funcs[idx];
      PhaseSplitter2 splitter(*analysis, symbolTable);
      if (failed(splitter.run()))
        return signalPassFailure();
    }
  }
};

} // namespace
