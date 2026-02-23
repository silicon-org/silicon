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
      newFrame.currentOp = &calleeOp.getBody().front().front();
      for (auto [arg, attr] :
           llvm::zip(calleeOp.getBody().getArguments(), operands))
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
    } else if (auto binaryOp = dyn_cast<mir::BinaryOp>(op)) {
      // For now, interpret binary ops on integers as addition.
      // TODO: Add binary op kind tracking and handle all ops properly.
      auto lhs = cast<mir::IntAttr>(operands[0]);
      auto rhs = cast<mir::IntAttr>(operands[1]);
      auto result =
          mir::IntAttr::get(op->getContext(), lhs.getValue() + rhs.getValue());
      frame.values[binaryOp] = result;
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

/// Parse a phase function name like "Foo.const2" into its base name ("Foo")
/// and phase number (2). Returns std::nullopt if the name doesn't match the
/// naming convention.
static std::optional<std::pair<StringRef, unsigned>>
parsePhaseFunc(StringRef name) {
  auto dotPos = name.rfind('.');
  if (dotPos == StringRef::npos)
    return std::nullopt;
  auto suffix = name.substr(dotPos + 1);
  if (!suffix.starts_with("const"))
    return std::nullopt;
  unsigned phaseNum;
  if (suffix.substr(5).getAsInteger(10, phaseNum))
    return std::nullopt;
  return std::make_pair(name.substr(0, dotPos), phaseNum);
}

void InterpretPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Iteratively interpret zero-arg functions and chain phase results into the
  // next phase. This handles callers that were split into multiple phases by
  // the unified_call decomposition in split-phases.
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto func : getOperation().getOps<hir::FuncOp>()) {
      if (func.getBody().getNumArguments() > 0)
        continue;
      // Only execute functions whose body contains MIR ops (already lowered).
      if (!func.getBody().front().getTerminator() ||
          !isa<mir::ReturnOp>(func.getBody().front().getTerminator()))
        continue;
      // Skip already-evaluated functions (only constants and a return).
      bool alreadyEvaluated = true;
      for (auto &op : func.getBody().front()) {
        if (!isa<mir::ConstantOp, mir::ReturnOp>(op)) {
          alreadyEvaluated = false;
          break;
        }
      }
      if (alreadyEvaluated) {
        // Chain constants into the next phase function if applicable.
        auto parsed = parsePhaseFunc(func.getSymName());
        if (!parsed || parsed->second == 0)
          continue;
        auto nextName =
            (parsed->first + ".const" + Twine(parsed->second - 1)).str();
        auto nextFunc = symbolTable.lookup<hir::FuncOp>(nextName);
        if (!nextFunc)
          continue;

        // Collect materialized constant return values.
        auto returnOp =
            cast<mir::ReturnOp>(func.getBody().front().getTerminator());
        if (returnOp.getNumOperands() != nextFunc.getBody().getNumArguments())
          continue;

        SmallVector<Attribute> returnConsts;
        for (auto operand : returnOp.getOperands()) {
          auto constOp = operand.getDefiningOp<mir::ConstantOp>();
          if (!constOp)
            break;
          returnConsts.push_back(constOp.getValue());
        }
        if (returnConsts.size() != returnOp.getNumOperands())
          continue;

        // Replace the next phase's block args with materialized constants.
        LLVM_DEBUG(llvm::dbgs() << "Chaining @" << func.getSymName() << " -> @"
                                << nextName << "\n");
        auto &nextBlock = nextFunc.getBody().front();
        OpBuilder builder(&nextBlock, nextBlock.begin());
        for (auto [arg, attr] :
             llvm::zip(nextBlock.getArguments(), returnConsts)) {
          auto constOp = mir::ConstantOp::create(builder, arg.getLoc(),
                                                 cast<TypedAttr>(attr));
          arg.replaceAllUsesWith(constOp);
        }
        nextBlock.eraseArguments(0, nextBlock.getNumArguments());
        changed = true;
        continue;
      }

      LLVM_DEBUG(llvm::dbgs() << "Interpreting @" << func.getSymName() << "\n");
      Interpreter interpreter(symbolTable);
      interpreter.callStack.emplace_back();
      interpreter.callStack.back().currentOp = &func.getBody().front().front();
      if (failed(interpreter.run()))
        return signalPassFailure();
      changed = true;
    }
  }
}
