//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/SymbolTable.h>

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "interpret"

namespace silicon {
#define GEN_PASS_DEF_INTERPRETPASS
#include "silicon/Transforms/Passes.h.inc"
} // namespace silicon

namespace {
struct CallFrame {
  Operation *currentOp;
  DenseMap<Value, Attribute> values;
  SmallVector<Value, 2> results;
};

struct Interpreter {
  SymbolTable &symbolTable;
  SmallVector<CallFrame> callStack;

  Interpreter(SymbolTable &symbolTable) : symbolTable(symbolTable) {}
  LogicalResult run();
};
} // namespace

LogicalResult Interpreter::run() {
  SmallVector<Attribute> operands;
  while (auto *op = callStack.back().currentOp) {
    auto &frame = callStack.back();
    LLVM_DEBUG(llvm::dbgs().indent(callStack.size() * 2)
               << "Executing " << *op << "\n");

    operands.resize(op->getNumOperands());
    for (auto [attr, operand] : llvm::zip(operands, op->getOpOperands())) {
      attr = frame.values.lookup(operand.get());
      if (!attr)
        return op->emitError()
               << "missing value for operand #" << operand.getOperandNumber();
    }

    // Special handling for call operations.
    if (auto callOp = dyn_cast<mir::CallOp>(op)) {
      auto calleeOp = symbolTable.lookup<hir::FuncOp>(callOp.getCallee());
      auto &newFrame = callStack.emplace_back();
      newFrame.currentOp = &calleeOp.getBodies()[0].front().front();
      for (auto [arg, attr] :
           llvm::zip(calleeOp.getBodies()[0].getArguments(), operands))
        newFrame.values[arg] = attr;
      continue;
    }

    // Special handling for return operations.
    if (auto returnOp = dyn_cast<mir::ReturnOp>(op)) {
      // If this is the top-most call frame, we are interpreting the body of a
      // function without any arguments. Materialize the constant return values
      // directly such that the result of interpreting the function is available
      // in the IR.
      if (callStack.size() == 1) {
        auto loc = op->getLoc();
        auto &region = *op->getParentRegion();
        region.getBlocks().clear();
        auto &block = region.emplaceBlock();
        auto builder = OpBuilder::atBlockBegin(&block);
        SmallVector<Value> results;
        results.reserve(operands.size());
        for (auto operand : operands)
          results.push_back(
              mir::ConstantOp::create(builder, loc, cast<TypedAttr>(operand)));
        mir::ReturnOp::create(builder, loc, results);
        return success();
      }

      // Otherwise copy the results into the parent call frame and continue
      // there.
      auto &callerFrame = callStack[callStack.size() - 2];
      for (auto [result, attr] :
           llvm::zip(callerFrame.currentOp->getResults(), operands))
        callerFrame.values[result] = attr;
      callerFrame.currentOp = callerFrame.currentOp->getNextNode();
      callStack.pop_back();
      continue;
    }

    // Handle all other operations.
    if (auto constOp = dyn_cast<mir::ConstantOp>(op)) {
      frame.values[constOp] = constOp.getValue();
    } else if (auto specializeFuncOp = dyn_cast<mir::SpecializeFuncOp>(op)) {
      auto attr = specializeFuncOp.interpret(
          mir::SpecializeFuncOp::FoldAdaptor(operands, specializeFuncOp));
      if (!attr)
        return op->emitError() << "interpretation failed";
      frame.values[specializeFuncOp] = attr;
    } else {
      return op->emitError() << "operation not supported by interpreter";
    }

    frame.currentOp = op->getNextNode();
  }
  return success();
}

namespace {
struct InterpretPass : public silicon::impl::InterpretPassBase<InterpretPass> {
  void runOnOperation() override;
};
} // namespace

void InterpretPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Interpret all `hir.func`s that take no arguments.
  for (auto func : getOperation().getOps<hir::FuncOp>()) {
    if (func.getBodies().empty() || func.getBodies()[0].getNumArguments() > 0)
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Interpreting @" << func.getSymName() << "\n");
    Interpreter interpreter(symbolTable);
    interpreter.callStack.emplace_back();
    interpreter.callStack.back().currentOp =
        &func.getBodies()[0].front().front();
    if (failed(interpreter.run()))
      return signalPassFailure();
  }
}
