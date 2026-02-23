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
/// A helper struct to analyze the phase split of a function and assign concrete
/// phase numbers to individual operations.
struct PhaseAnalysis {
  PhaseAnalysis(UnifiedFuncOp funcOp) : funcOp(funcOp) {}
  void analyze();
  void addToWorklist(Operation *op);
  void drainWorklist();

  struct Item {
    Operation *op;
    bool checkParent;
    Operation::user_iterator userIt;
  };
  UnifiedFuncOp funcOp;
  DenseSet<Operation *> seenOps;
  SmallVector<Item, 0> worklist;
  DenseMap<Operation *, int16_t> opPhases;

  /// Phases assigned to body block arguments based on constness of the
  /// corresponding UnifiedArgOp in the signature.
  DenseMap<Value, int16_t> argPhases;
};
} // namespace

void PhaseAnalysis::analyze() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                          << "\n");

  // Determine which body block arguments have a non-zero (shifted) phase based
  // on the argPhases attribute. Negative phases are const (earlier), positive
  // phases are dyn (later).
  auto bodyArgs = funcOp.getBody().getArguments();
  for (auto [idx, phase] : llvm::enumerate(funcOp.getArgPhases())) {
    int16_t argPhase = static_cast<int16_t>(phase);
    if (argPhase < 0) {
      argPhases[bodyArgs[idx]] = argPhase;
      LLVM_DEBUG(llvm::dbgs()
                 << "- Arg " << idx << " has phase " << argPhase << "\n");
    }
  }

  opPhases.insert({funcOp, 0});
  funcOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *op) {
    addToWorklist(op);
    drainWorklist();
  });
}

void PhaseAnalysis::addToWorklist(Operation *op) {
  if (opPhases.contains(op))
    return;

  if (!seenOps.insert(op).second) {
    LLVM_DEBUG({
      llvm::dbgs() << "- Cycle detected; op already seen: ";
      op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    return;
  }

  // If the op has side-effects, we do not adjust its phase based on its users.
  // If the op has no side-effects, visit all users first.
  auto userIt = op->user_end();
  if (mlir::isMemoryEffectFree(op))
    userIt = op->user_begin();
  worklist.push_back({op, true, userIt});
}

void PhaseAnalysis::drainWorklist() {
  while (!worklist.empty()) {
    auto &item = worklist.back();

    // Visit the parent op if we haven't done so yet.
    if (item.checkParent) {
      item.checkParent = false;
      addToWorklist(item.op->getParentOp());
      continue;
    }

    // Visit the next user if we have any left.
    if (item.userIt != item.op->user_end()) {
      addToWorklist(*item.userIt);
      ++item.userIt;
      continue;
    }

    // All dependencies have been processed, compute the phase.
    // At the most basic level we inherit the phase of our parent.
    int16_t phase = opPhases.at(item.op->getParentOp());

    // If this is an `hir.expr` op, its phase can be shifted based on its
    // `const` attribute.
    IntegerAttr constAttr;
    if (isa<ExprOp>(item.op) &&
        (constAttr = item.op->getAttrOfType<IntegerAttr>("const"))) {
      phase += constAttr.getInt();
    }

    // If this op has no side-effects (and is not an ExprOp with an explicit
    // phase attribute), shift its phase to accommodate all users.
    if (!constAttr &&
        (isa<ExprOp>(item.op) || mlir::isMemoryEffectFree(item.op)))
      for (auto *user : item.op->getUsers())
        phase = std::max(phase, opPhases.at(user));

    LLVM_DEBUG({
      llvm::dbgs() << "- Computed phase " << phase << " for: ";
      item.op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
      llvm::dbgs() << "\n";
    });
    opPhases.insert({item.op, phase});
    worklist.pop_back();
  }
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

  // Determine the phase range, considering both operation phases and argument
  // phases. Negative phases are const (run earlier), positive are dyn (later).
  int16_t minPhase = 0, maxPhase = 0;
  for (auto &[op, phase] : analysis.opPhases) {
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  }
  for (auto &[value, phase] : analysis.argPhases) {
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
  // functions. For each shifted arg, we create a block arg in that phase's
  // function, add it to that phase's return values, create a receiving block
  // arg in phase 0, and replace all uses of the original body block arg.
  {
    auto &phase0Block = splits[0 - minPhase].funcOp.getBody().front();

    // Collect const args and their phases first to avoid iterator invalidation.
    SmallVector<std::pair<unsigned, int16_t>> constArgs;
    for (auto &[value, phase] : analysis.argPhases) {
      auto bodyArg = cast<BlockArgument>(value);
      constArgs.push_back({bodyArg.getArgNumber(), phase});
    }
    llvm::sort(constArgs);

    // Process each const arg.
    SmallVector<unsigned> argsToErase;
    for (auto [idx, argPhase] : constArgs) {
      auto bodyArg = phase0Block.getArgument(idx);
      LLVM_DEBUG(llvm::dbgs() << "- Moving body arg " << idx << " to phase "
                              << argPhase << "\n");

      // Create a block arg in the shifted-phase function to receive this value.
      auto &constBlock = splits[argPhase - minPhase].funcOp.getBody().front();
      auto constArg =
          constBlock.addArgument(bodyArg.getType(), bodyArg.getLoc());

      // The shifted-phase function returns this value so it flows to phase 0.
      splits[argPhase - minPhase].returnValues.push_back(constArg);

      // Create a new block arg in phase 0 to receive the value from the const
      // phase, and replace all uses of the old body block arg.
      auto newPhase0Arg =
          phase0Block.addArgument(bodyArg.getType(), bodyArg.getLoc());
      bodyArg.replaceAllUsesWith(newPhase0Arg);
      argsToErase.push_back(idx);
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
    builder.create<ReturnOp>(funcOp.getLoc(), split.returnValues);

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

/// Rewrite UnifiedCallOps to call the split phase functions directly. For each
/// unified call, the const arguments go to the const-phase function and the
/// runtime arguments go to the runtime-phase function. The const-phase results
/// are threaded through to the runtime-phase call.
static void rewriteUnifiedCalls(ModuleOp moduleOp, SymbolTable &symbolTable) {
  moduleOp.walk([&](UnifiedCallOp callOp) {
    auto calleeName = callOp.getCallee();

    // Look up the split functions. If the const-phase function doesn't exist,
    // this callee wasn't split.
    auto constFuncName = (calleeName + ".const1").str();
    auto runtimeFuncName = (calleeName + ".const0").str();
    auto constFunc = symbolTable.lookup<FuncOp>(constFuncName);
    auto runtimeFunc = symbolTable.lookup<FuncOp>(runtimeFuncName);
    if (!constFunc || !runtimeFunc)
      return;

    OpBuilder builder(callOp);
    auto loc = callOp.getLoc();
    auto argPhases = callOp.getArgPhases();

    // Partition arguments and their types into const and runtime based on
    // the phase annotations from the unified call.
    SmallVector<Value> constArgs, runtimeArgs;
    SmallVector<Value> constTypeOfArgs, runtimeTypeOfArgs;
    for (auto [arg, type, phase] :
         llvm::zip(callOp.getArguments(), callOp.getTypeOfArgs(), argPhases)) {
      if (phase < 0) {
        constArgs.push_back(arg);
        constTypeOfArgs.push_back(type);
      } else {
        runtimeArgs.push_back(arg);
        runtimeTypeOfArgs.push_back(type);
      }
    }

    // Result types for the const call are not yet known; use inferrables.
    SmallVector<Value> constCallTypeOfResults;
    for (size_t i = 0; i < callOp.getResultTypes().size(); ++i)
      constCallTypeOfResults.push_back(
          InferrableOp::create(builder, loc).getResult());

    // Call the const-phase function with the const arguments.
    auto constCall =
        CallOp::create(builder, loc, callOp.getResultTypes(),
                       builder.getStringAttr(constFuncName), constArgs,
                       constTypeOfArgs, constCallTypeOfResults);

    // The const-phase function returns values that the runtime phase needs.
    // Thread them through as additional arguments to the runtime-phase call,
    // with inferrable type placeholders since the types are resolved later.
    SmallVector<Value> fullRuntimeArgs(runtimeArgs);
    SmallVector<Value> fullRuntimeTypeOfArgs(runtimeTypeOfArgs);
    for (auto result : constCall.getResults()) {
      fullRuntimeArgs.push_back(result);
      fullRuntimeTypeOfArgs.push_back(
          InferrableOp::create(builder, loc).getResult());
    }

    // Call the runtime-phase function with the combined argument list.
    auto runtimeCall =
        CallOp::create(builder, loc, callOp.getResultTypes(),
                       builder.getStringAttr(runtimeFuncName), fullRuntimeArgs,
                       fullRuntimeTypeOfArgs, callOp.getTypeOfResults());

    callOp.replaceAllUsesWith(runtimeCall.getResults());
    callOp.erase();
  });
}

void SplitPhasesPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<UnifiedFuncOp>())) {
    PhaseAnalysis analysis(op);
    analysis.analyze();
    PhaseSplitter splitter(analysis, symbolTable);
    splitter.run();
    op.erase();
  }

  // Rewrite unified calls to use the split phase functions.
  rewriteUnifiedCalls(getOperation(), symbolTable);
}
