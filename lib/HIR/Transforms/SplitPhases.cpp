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
  PhaseAnalysis(UncheckedFuncOp funcOp) : funcOp(funcOp) {}
  void analyze();
  void addToWorklist(Operation *op);
  void drainWorklist();

  struct Item {
    Operation *op;
    bool checkParent;
    Operation::user_iterator userIt;
  };
  UncheckedFuncOp funcOp;
  DenseSet<Operation *> seenOps;
  SmallVector<Item, 0> worklist;
  DenseMap<Operation *, int16_t> opPhases;

  /// Phases assigned to body block arguments based on constness of the
  /// corresponding UncheckedArgOp in the signature.
  DenseMap<Value, int16_t> argPhases;
};
} // namespace

void PhaseAnalysis::analyze() {
  LLVM_DEBUG(llvm::dbgs() << "Analyzing phases in " << funcOp.getSymNameAttr()
                          << "\n");

  // Determine which body block arguments are const based on the constness of
  // the corresponding UncheckedArgOp in the signature.
  auto sigTerminator = funcOp.getSignatureOp();
  auto bodyArgs = funcOp.getBody().getArguments();
  for (auto [idx, argValue] : llvm::enumerate(sigTerminator.getArgValues())) {
    auto argOp = cast<UncheckedArgOp>(argValue.getDefiningOp());
    int16_t constness = argOp.getConstness();
    if (constness > 0) {
      argPhases[bodyArgs[idx]] = constness;
      LLVM_DEBUG(llvm::dbgs() << "- Arg " << idx << " has constness "
                               << constness << "\n");
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

    // If this op has no side-effects, shift its phase to accommodate all users.
    if ((isa<ExprOp>(item.op) && !constAttr) ||
        mlir::isMemoryEffectFree(item.op))
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
  UncheckedFuncOp funcOp;
};
} // namespace

void PhaseSplitter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Splitting " << funcOp.getSymNameAttr() << "\n");

  // Determine the maximum phase number, considering both operation phases and
  // argument constness levels.
  int16_t maxPhase = 0;
  for (auto &[op, phase] : analysis.opPhases)
    if (maxPhase < phase)
      maxPhase = phase;
  for (auto &[value, phase] : analysis.argPhases)
    if (maxPhase < phase)
      maxPhase = phase;
  LLVM_DEBUG(llvm::dbgs() << "- Phases range [0, " << maxPhase << "]\n");

  // Create a separate function for each phase.
  OpBuilder builder(funcOp);
  auto privateAttr = builder.getStringAttr("private");
  SmallVector<PhaseSplit> splits;
  for (int16_t phase = 0; phase <= maxPhase; ++phase) {
    auto name =
        builder.getStringAttr(funcOp.getSymName() + ".const" + Twine(phase));
    auto phaseFuncOp =
        FuncOp::create(builder, funcOp.getLoc(), name, privateAttr);
    if (phase != 0)
      phaseFuncOp.getBody().emplaceBlock();
    symbolTable.insert(phaseFuncOp);
    builder.setInsertionPoint(phaseFuncOp);
    auto &split = splits.emplace_back();
    split.funcOp = phaseFuncOp;
    split.phase = phase;
  }

  // Handle the return operation. Return values are added to phase 0's return
  // values. The return op itself is erased since each phase will get its own.
  auto returnOp = funcOp.getReturnOp();
  for (auto operand : returnOp.getOperands())
    splits[0].returnValues.push_back(operand);
  returnOp.erase();

  // Move all operations to phase 0 initially.
  SmallVector<std::tuple<int16_t, Block::iterator, Block::iterator>> worklist;
  for (auto &block : llvm::reverse(funcOp.getBody()))
    worklist.push_back({0, block.begin(), block.end()});

  splits[0].funcOp.getBody().takeBody(funcOp.getBody());

  // Move const body block arguments to their respective const-phase functions.
  // For each const arg, we create a block arg in the const-phase function, add
  // it to that phase's return values, create a receiving block arg in phase 0,
  // and replace all uses of the original body block arg.
  {
    auto &phase0Block = splits[0].funcOp.getBody().front();

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

      // Create a block arg in the const-phase function to receive this value.
      auto &constBlock = splits[argPhase].funcOp.getBody().front();
      auto constArg =
          constBlock.addArgument(bodyArg.getType(), bodyArg.getLoc());

      // The const-phase function returns this value so it flows to phase 0.
      splits[argPhase].returnValues.push_back(constArg);

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
      auto &split = splits[opPhase];
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
  // phases to later phases.
  DenseMap<Value, Value> mapping;
  SmallDenseSet<Operation *, 4> closedFuncs;
  for (int16_t phase = 0; phase <= maxPhase; ++phase) {
    auto &split = splits[phase];
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

            // Add the value as a result to the earlier phase function.
            assert(phase < maxPhase);
            splits[phase + 1].returnValues.push_back(operand.get());
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

void SplitPhasesPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  for (auto op :
       llvm::make_early_inc_range(getOperation().getOps<UncheckedFuncOp>())) {
    PhaseAnalysis analysis(op);
    analysis.analyze();
    PhaseSplitter splitter(analysis, symbolTable);
    splitter.run();
    op.erase();
  }
}
