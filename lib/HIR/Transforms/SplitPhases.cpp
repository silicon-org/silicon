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
    auto phaseFuncOp =
        FuncOp::create(builder, funcOp.getLoc(), name, privateAttr);
    if (phase != 0)
      phaseFuncOp.getBody().emplaceBlock();
    symbolTable.insert(phaseFuncOp);
    builder.setInsertionPoint(phaseFuncOp);
    splits[phase - minPhase].funcOp = phaseFuncOp;
    splits[phase - minPhase].phase = phase;
  }

  // Handle the return operation. Return values are added to phase 0's return
  // values. The return op itself is erased since each phase will get its own.
  auto returnOp = funcOp.getReturnOp();
  for (auto operand : returnOp.getOperands())
    splits[0 - minPhase].returnValues.push_back(operand);
  returnOp.erase();

  // Move all operations to phase 0 initially (using index 0 - minPhase).
  SmallVector<std::tuple<int16_t, Block::iterator, Block::iterator>> worklist;
  for (auto &block : llvm::reverse(funcOp.getBody()))
    worklist.push_back({0, block.begin(), block.end()});

  splits[0 - minPhase].funcOp.getBody().takeBody(funcOp.getBody());

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
      replacements.push_back({idx, ownArg});
      argsToErase.push_back(idx);
    }

    // Step 2: Replace all uses of old body block args with the origin-phase
    // values. Cross-phase value threading will handle forwarding to later
    // phases as needed.
    for (auto &[idx, ownArg] : replacements) {
      auto bodyArg = phase0Block.getArgument(idx);
      bodyArg.replaceAllUsesWith(ownArg);
    }

    // Erase old body block args in reverse order.
    for (auto idx : llvm::reverse(argsToErase))
      phase0Block.eraseArgument(idx);
  }

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

  // Add return operations to all phase functions and plumb values from earlier
  // phases to later phases. We iterate in reverse execution order (most-runtime
  // first, most-const last) so that thread-through adds values to later splits
  // before their return ops are emitted.
  DenseMap<Value, Value> mapping;
  SmallDenseSet<Operation *, 4> closedFuncs;
  for (int16_t phase = maxPhase; phase >= minPhase; --phase) {
    auto &split = splits[phase - minPhase];
    closedFuncs.insert(split.funcOp);

    // Add a return operation to this split.
    builder.setInsertionPointToEnd(&split.funcOp.getBody().back());
    ReturnOp::create(builder, funcOp.getLoc(), split.returnValues);

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
            splits[phase - 1 - minPhase].returnValues.push_back(operand.get());
          }
        }
        operand.set(value);
      }
    });
    mapping.clear();
  }
}

namespace {
struct SplitPhasesPass
    : public hir::impl::SplitPhasesPassBase<SplitPhasesPass> {
  void runOnOperation() override;
};
} // namespace

/// Rewrite UnifiedCallOps to call the split phase functions directly.
///
/// For each unified call, we discover the split functions for the callee,
/// partition arguments by phase, and emit a chain of calls from the earliest
/// phase to the latest. Each call receives its own phase's arguments plus all
/// results from the previous phase's call.
static void rewriteUnifiedCalls(ModuleOp moduleOp, SymbolTable &symbolTable) {
  moduleOp.walk([&](UnifiedCallOp callOp) {
    auto calleeName = callOp.getCallee();
    auto argPhases = callOp.getArgPhases();
    auto resultPhases = callOp.getResultPhases();

    // Compute the phase range from argument and result phases.
    int16_t minPhase = 0, maxPhase = 0;
    for (auto p : argPhases) {
      minPhase = std::min(minPhase, static_cast<int16_t>(p));
      maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
    }
    for (auto p : resultPhases) {
      minPhase = std::min(minPhase, static_cast<int16_t>(p));
      maxPhase = std::max(maxPhase, static_cast<int16_t>(p));
    }

    // Discover all split functions by name. If any are missing, this callee
    // wasn't split.
    SmallVector<std::pair<int16_t, FuncOp>> splitFuncs;
    for (int16_t phase = minPhase; phase <= maxPhase; ++phase) {
      auto name = phase <= 0 ? (calleeName + ".const" + Twine(-phase)).str()
                             : (calleeName + ".dyn" + Twine(phase)).str();
      auto func = symbolTable.lookup<FuncOp>(name);
      if (!func)
        return;
      splitFuncs.push_back({phase, func});
    }

    OpBuilder builder(callOp);
    auto loc = callOp.getLoc();

    // Partition arguments and their type operands by phase.
    DenseMap<int16_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
    for (auto [arg, type, phase] :
         llvm::zip(callOp.getArguments(), callOp.getTypeOfArgs(), argPhases)) {
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
        callTypeOfArgs.push_back(
            InferrableOp::create(builder, loc).getResult());
      }

      // The final phase uses the original unified_call's result types.
      // Intermediate phases determine their result count from the split
      // function's return op.
      bool isFinal = (phase == maxPhase);
      SmallVector<Value> callTypeOfResults;
      SmallVector<Type> resultTypes;
      if (isFinal) {
        callTypeOfResults.append(callOp.getTypeOfResults().begin(),
                                 callOp.getTypeOfResults().end());
        resultTypes.append(callOp.getResultTypes().begin(),
                           callOp.getResultTypes().end());
      } else {
        auto returnOp =
            cast<ReturnOp>(splitFunc.getBody().front().getTerminator());
        for (unsigned i = 0; i < returnOp.getNumOperands(); ++i) {
          callTypeOfResults.push_back(
              InferrableOp::create(builder, loc).getResult());
          resultTypes.push_back(returnOp.getOperand(i).getType());
        }
      }

      auto call = CallOp::create(builder, loc, resultTypes,
                                 builder.getStringAttr(splitFunc.getSymName()),
                                 callArgs, callTypeOfArgs, callTypeOfResults);
      prevResults.assign(call.getResults().begin(), call.getResults().end());
    }

    callOp.replaceAllUsesWith(prevResults);
    callOp.erase();
  });
}

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

  // Split each function using its pre-computed analysis.
  for (auto &[funcOp, analysis] : analyses) {
    PhaseSplitter splitter(analysis, symbolTable);
    splitter.run();
    funcOp.erase();
  }

  // Rewrite unified calls to use the split phase functions.
  rewriteUnifiedCalls(getOperation(), symbolTable);
}
