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
  int16_t getValuePhase(Value value) const;
  LogicalResult checkCallArgPhases();

  UnifiedFuncOp funcOp;
  DenseMap<Operation *, int16_t> opPhases;

  /// Phases for all values: body block args and op results.
  DenseMap<Value, int16_t> valuePhases;
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
    } else if (!isa<ExprOp>(op) && mlir::isMemoryEffectFree(op)) {
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

namespace {
struct PhaseSplit {
  FuncOp funcOp;
  int16_t phase;
  SmallVector<Value> returnValues;
  SmallVector<Value> returnTypeOperands;
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

  // Create a separate function for each phase. Phases ≤ 0 are named
  // `.constN` (N = -phase); phases > 0 are named `.dynN`. We iterate in
  // reverse execution order so each new insertion goes before the previous,
  // resulting in the earlier-executing (more-const) phases appearing first
  // in the module.
  OpBuilder builder(funcOp);
  auto privateAttr = builder.getStringAttr("private");
  SmallVector<PhaseSplit> splits(maxPhase - minPhase + 1);
  for (int16_t phase = maxPhase; phase >= minPhase; --phase) {
    auto name = phase <= 0 ? builder.getStringAttr(funcOp.getSymName() +
                                                   ".const" + Twine(-phase))
                           : builder.getStringAttr(funcOp.getSymName() +
                                                   ".dyn" + Twine(phase));
    auto emptyArray = builder.getArrayAttr({});
    auto phaseFuncOp = FuncOp::create(builder, funcOp.getLoc(), name,
                                      privateAttr, emptyArray, emptyArray);
    if (phase != 0)
      phaseFuncOp.getBody().emplaceBlock();
    symbolTable.insert(phaseFuncOp);
    builder.setInsertionPoint(phaseFuncOp);
    splits[phase - minPhase].funcOp = phaseFuncOp;
    splits[phase - minPhase].phase = phase;
  }

  // Handle the return operation. Return values and their type operands are
  // added to phase 0's split. The return op itself is erased since each phase
  // will get its own.
  auto returnOp = funcOp.getReturnOp();
  for (auto value : returnOp.getValues())
    splits[0 - minPhase].returnValues.push_back(value);
  for (auto type : returnOp.getTypeOfValues())
    splits[0 - minPhase].returnTypeOperands.push_back(type);
  returnOp.erase();

  // Move all operations to phase 0 initially.
  splits[0 - minPhase].funcOp.getBody().takeBody(funcOp.getBody());

  // Track where each unified func arg ended up after const arg movement.
  // Const args map to their new block arg in the origin phase function;
  // phase-0 args map to the remaining block args in the phase-0 function.
  SmallVector<Value> unifiedArgValues(funcOp.getArgPhases().size());

  // Move shifted-phase body block arguments to their respective phase
  // functions. We create block args in each const arg's origin phase, replace
  // all uses, and let cross-phase value threading handle multi-hop forwarding.
  {
    auto &phase0Block = splits[0 - minPhase].funcOp.getBody().front();

    // Collect const args and their phases first to avoid iterator invalidation.
    SmallVector<std::pair<unsigned, int16_t>> constArgs;
    for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
      int16_t argPhase = static_cast<int16_t>(phase);
      if (argPhase < 0)
        constArgs.push_back({idx, argPhase});
    }
    llvm::sort(constArgs);

    // Step 1: Create a block arg in each const arg's origin phase function.
    SmallVector<std::pair<unsigned, Value>> replacements;
    SmallVector<unsigned> argsToErase;
    for (auto [idx, argPhase] : constArgs) {
      auto bodyArg = phase0Block.getArgument(idx);
      LLVM_DEBUG(llvm::dbgs() << "- Moving body arg " << idx << " to phase "
                              << argPhase << "\n");
      auto &constBlock = splits[argPhase - minPhase].funcOp.getBody().front();
      Value ownArg =
          constBlock.addArgument(bodyArg.getType(), bodyArg.getLoc());
      unifiedArgValues[idx] = ownArg;
      replacements.push_back({idx, ownArg});
      argsToErase.push_back(idx);
    }

    // Step 2: Replace all uses of old body block args with the origin-phase
    // values. Cross-phase value threading will handle forwarding to later
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
      if (unifiedArgValues[i]) // already filled (const arg)
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

      // Discover all split functions by name. If any are missing, the callee
      // wasn't split; skip this call.
      SmallVector<std::pair<int16_t, FuncOp>> splitFuncs;
      for (int16_t phase = calleeMinPhase; phase <= calleeMaxPhase; ++phase) {
        auto name = phase <= 0 ? (calleeName + ".const" + Twine(-phase)).str()
                               : (calleeName + ".dyn" + Twine(phase)).str();
        auto func = symbolTable.lookup<FuncOp>(name);
        if (!func) {
          splitFuncs.clear();
          break;
        }
        splitFuncs.push_back({phase, func});
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

      // Chain calls from earliest phase to latest. Each call gets its own
      // phase's arguments plus all results from the previous phase's call.
      SmallVector<Value> prevResults;
      for (auto &[phase, splitFunc] : splitFuncs) {
        SmallVector<Value> callArgs(phaseArgs[phase]);
        SmallVector<Value> callTypeOfArgs(phaseTypeOfArgs[phase]);

        // Thread results from the previous phase.
        for (auto result : prevResults) {
          callArgs.push_back(result);
          auto inferrable = InferrableOp::create(callBuilder, loc);
          analysis.opPhases[inferrable] = phase;
          analysis.valuePhases[inferrable.getResult()] = phase;
          callTypeOfArgs.push_back(inferrable.getResult());
        }

        // The final phase uses the original unified_call's result types.
        // Intermediate phases determine their result count from the split
        // function's return op.
        bool isFinal = (phase == calleeMaxPhase);
        SmallVector<Value> callTypeOfResults;
        SmallVector<Type> resultTypes;
        if (isFinal) {
          callTypeOfResults.append(callOp.getTypeOfResults().begin(),
                                   callOp.getTypeOfResults().end());
          resultTypes.append(callOp.getResultTypes().begin(),
                             callOp.getResultTypes().end());
        } else {
          auto retOp =
              cast<ReturnOp>(splitFunc.getBody().front().getTerminator());
          for (unsigned i = 0; i < retOp.getValues().size(); ++i) {
            auto inferrable = InferrableOp::create(callBuilder, loc);
            analysis.opPhases[inferrable] = phase;
            analysis.valuePhases[inferrable.getResult()] = phase;
            callTypeOfResults.push_back(inferrable.getResult());
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
      }

      // Update any return values that reference the old unified call's results.
      // These are not IR uses (the unified_return was already erased), so
      // replaceAllUsesWith won't touch them.
      for (auto &rv : splits[0 - minPhase].returnValues)
        for (auto [oldResult, newResult] :
             llvm::zip(callOp.getResults(), prevResults))
          if (rv == oldResult)
            rv = newResult;

      callOp.replaceAllUsesWith(prevResults);
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
        llvm::dbgs() << "- Moving to phase " << opPhase << ": ";
        op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
        llvm::dbgs() << "\n";
      });
      auto &split = splits[opPhase - minPhase];
      auto *block = &split.funcOp.getBody().back();
      op->moveBefore(block, block->end());
    }

    // If this is an `ExprOp`, push its nested operations onto the worklist,
    // since those might move to a different phase as well.
    if (auto exprOp = dyn_cast<ExprOp>(op)) {
      LLVM_DEBUG({
        llvm::dbgs() << "- Descending into ";
        op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
        llvm::dbgs() << "\n";
      });
      for (auto &block : llvm::reverse(exprOp.getBody()))
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
                llvm::dbgs() << "- Cloning into phase " << phase << ": ";
                defOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
                llvm::dbgs() << "\n";
              });
              auto &block = split.funcOp.getBody().front();
              OpBuilder cloneBuilder(&block, block.begin());
              auto *cloned = cloneBuilder.clone(*defOp);
              value = cloned->getResult(
                  cast<OpResult>(operand.get()).getResultNumber());
            } else {
              // Add an argument to the current split function.
              LLVM_DEBUG({
                llvm::dbgs() << "- Creating arg in phase " << phase << " for ";
                operand.get().print(llvm::dbgs(),
                                    OpPrintingFlags().skipRegions());
                llvm::dbgs() << "\n";
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
    // arg followed by an opaque_unpack.
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
  // Set Argument and Result Names on Phase Functions
  //
  // Now that each phase function has its final block args and return values,
  // assign meaningful names. Own args get their original names from the
  // unified func; the opaque context arg (if any) gets named "ctx". Phase 0's
  // own results get the unified func's result names; context returns get "ctx".
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

    // Build resultNames: unified func's result names for phase 0's own
    // results, "ctx" for context returns.
    SmallVector<Attribute> phaseResultNames;
    if (auto retOp = split.funcOp.getReturnOp()) {
      unsigned ownReturns = numOwnReturns[phase - minPhase];
      if (phase == 0) {
        auto rn = funcOp.getResultNames();
        for (unsigned i = 0; i < ownReturns && i < rn.size(); ++i)
          phaseResultNames.push_back(rn[i]);
      }
      if (retOp.getValues().size() > ownReturns)
        phaseResultNames.push_back(builder.getStringAttr("ctx"));
    }
    split.funcOp.setResultNamesAttr(builder.getArrayAttr(phaseResultNames));
  }

  //===--------------------------------------------------------------------===//
  // Emit Structural Ops
  //
  // Build a split_func recording the full phase-to-function mapping, and a
  // multiphase_func wrapping compile-time sub-functions when there are 2+
  // compile-time phases. The unified func is erased afterward.
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

  // Build phase-to-function mapping arrays.
  SmallVector<int32_t> phaseNumbers;
  SmallVector<Attribute> phaseFuncAttrs;
  for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
    phaseNumbers.push_back(phase);
    phaseFuncAttrs.push_back(FlatSymbolRefAttr::get(
        ctx, splits[phase - minPhase].funcOp.getSymName()));
  }

  // Create the split_func.
  auto splitFuncOp = SplitFuncOp::create(
      sfBuilder, sfLoc, sfName,
      /*sym_visibility=*/StringAttr{}, sfArgNames,
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

  // Emit a multiphase_func wrapping the compile-time sub-functions if there
  // are 2+ compile-time phases (phases < 0).
  if (minPhase <= -2) {
    SmallVector<Attribute> constFuncAttrs;
    for (int16_t phase = minPhase; phase < 0; ++phase)
      constFuncAttrs.push_back(FlatSymbolRefAttr::get(
          ctx, splits[phase - minPhase].funcOp.getSymName()));

    // Only include compile-time args (phase < 0). An arg is "first" if it
    // enters at the earliest compile-time phase, "last" otherwise.
    SmallVector<Attribute> mpArgNames;
    SmallVector<bool> mpArgIsFirst;
    for (auto [name, phase] : llvm::zip(sfArgNames, sfArgPhases)) {
      if (phase < 0) {
        mpArgNames.push_back(name);
        mpArgIsFirst.push_back(phase == minPhase);
      }
    }

    // One result per value threaded from the last compile-time phase to phase
    // 0.
    auto lastConstRetOp = cast<ReturnOp>(
        splits[-1 - minPhase].funcOp.getBody().back().getTerminator());
    unsigned numCtx = lastConstRetOp.getValues().size();
    SmallVector<Attribute> mpResultNames;
    if (numCtx == 1) {
      mpResultNames.push_back(sfBuilder.getStringAttr("ctx"));
    } else {
      for (unsigned i = 0; i < numCtx; ++i)
        mpResultNames.push_back(sfBuilder.getStringAttr("ctx" + Twine(i)));
    }

    MultiphaseFuncOp::create(
        sfBuilder, sfLoc, sfBuilder.getStringAttr(sfName.getValue() + ".const"),
        /*sym_visibility=*/StringAttr{}, sfBuilder.getArrayAttr(mpArgNames),
        sfBuilder.getDenseBoolArrayAttr(mpArgIsFirst),
        sfBuilder.getArrayAttr(mpResultNames),
        sfBuilder.getArrayAttr(constFuncAttrs));
  }

  // Erase the unified func now that all data has been extracted.
  symbolTable.erase(funcOp);
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
