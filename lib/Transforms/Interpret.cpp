//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/MIR/Types.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "llvm/ADT/STLExtras.h"
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
      auto calleeOp = symbolTable.lookup<mir::FuncOp>(callOp.getCallee());
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
    } else if (isa<mir::AddOp, mir::SubOp, mir::MulOp, mir::DivOp, mir::ModOp,
                   mir::AndOp, mir::OrOp, mir::XorOp, mir::ShlOp, mir::ShrOp,
                   mir::EqOp, mir::NeqOp, mir::LtOp, mir::GtOp, mir::GeqOp,
                   mir::LeqOp>(op)) {
      auto lhs = cast<mir::IntAttr>(operands[0]).getValue();
      auto rhs = cast<mir::IntAttr>(operands[1]).getValue();
      DynamicAPInt value;
      if (isa<mir::AddOp>(op))
        value = lhs + rhs;
      else if (isa<mir::SubOp>(op))
        value = lhs - rhs;
      else if (isa<mir::MulOp>(op))
        value = lhs * rhs;
      else if (isa<mir::DivOp>(op))
        value = lhs / rhs;
      else if (isa<mir::ModOp>(op))
        value = lhs % rhs;
      // Bitwise and shift ops use int64_t since DynamicAPInt does not
      // support these operations directly.
      else if (isa<mir::AndOp>(op))
        value = DynamicAPInt(int64_t(lhs) & int64_t(rhs));
      else if (isa<mir::OrOp>(op))
        value = DynamicAPInt(int64_t(lhs) | int64_t(rhs));
      else if (isa<mir::XorOp>(op))
        value = DynamicAPInt(int64_t(lhs) ^ int64_t(rhs));
      else if (isa<mir::ShlOp>(op))
        value = DynamicAPInt(int64_t(lhs) << int64_t(rhs));
      else if (isa<mir::ShrOp>(op))
        value = DynamicAPInt(int64_t(lhs) >> int64_t(rhs));
      else if (isa<mir::EqOp>(op))
        value = DynamicAPInt(lhs == rhs ? 1 : 0);
      else if (isa<mir::NeqOp>(op))
        value = DynamicAPInt(lhs != rhs ? 1 : 0);
      else if (isa<mir::LtOp>(op))
        value = DynamicAPInt(lhs < rhs ? 1 : 0);
      else if (isa<mir::GtOp>(op))
        value = DynamicAPInt(lhs > rhs ? 1 : 0);
      else if (isa<mir::GeqOp>(op))
        value = DynamicAPInt(lhs >= rhs ? 1 : 0);
      else if (isa<mir::LeqOp>(op))
        value = DynamicAPInt(lhs <= rhs ? 1 : 0);
      frame.values[op->getResult(0)] =
          mir::IntAttr::get(op->getContext(), value);
    } else if (auto specializeFuncOp = dyn_cast<mir::SpecializeFuncOp>(op)) {
      auto attr = specializeFuncOp.interpret(
          mir::SpecializeFuncOp::FoldAdaptor(operands, specializeFuncOp));
      if (!attr)
        return op->emitError() << "interpretation failed";
      frame.values[specializeFuncOp] = attr;
    } else if (auto packOp = dyn_cast<mir::MIROpaquePackOp>(op)) {
      frame.values[packOp] = mir::OpaqueAttr::get(op->getContext(), operands);
    } else if (auto unpackOp = dyn_cast<mir::MIROpaqueUnpackOp>(op)) {
      auto opaqueAttr = cast<mir::OpaqueAttr>(operands[0]);
      for (auto [result, elem] :
           llvm::zip(unpackOp.getResults(), opaqueAttr.getElements()))
        frame.values[result] = elem;
    } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
      for (auto [result, attr] : llvm::zip(castOp.getResults(), operands))
        frame.values[result] = attr;
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

/// Check whether a function body has been fully evaluated, i.e., consists only
/// of `mir.constant` and `mir.return` operations.
static bool isFullyEvaluated(mir::FuncOp func) {
  return llvm::all_of(func.getBody().front(), [](Operation &op) {
    return isa<mir::ConstantOp, mir::ReturnOp>(op);
  });
}

/// Collect the constant return values from a fully-evaluated function body.
/// Returns an empty vector if any return operand is not a constant.
static SmallVector<Attribute> collectReturnConstants(mir::FuncOp func) {
  auto returnOp = cast<mir::ReturnOp>(func.getBody().front().getTerminator());
  SmallVector<Attribute> result;
  for (auto operand : returnOp.getOperands()) {
    auto constOp = operand.getDefiningOp<mir::ConstantOp>();
    if (!constOp)
      return {};
    result.push_back(constOp.getValue());
  }
  return result;
}

//===----------------------------------------------------------------------===//
// InterpretPass
//
// Iteratively interprets zero-argument MIR-body functions, then uses
// `hir.split_func` phase maps to chain evaluated results into the next
// sub-function. For `hir.multiphase_func` ops, collapses them as their
// sub-functions get evaluated. Repeats until no more progress is made.
//===----------------------------------------------------------------------===//

void InterpretPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  bool changed = true;
  while (changed) {
    changed = false;

    // Interpret zero-argument functions whose bodies contain MIR ops.
    for (auto func : getOperation().getOps<mir::FuncOp>()) {
      if (func.getBody().getNumArguments() > 0)
        continue;
      if (!func.getBody().front().getTerminator() ||
          !isa<mir::ReturnOp>(func.getBody().front().getTerminator()))
        continue;
      if (isFullyEvaluated(func))
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Interpreting @" << func.getSymName() << "\n");
      Interpreter interpreter(symbolTable);
      interpreter.callStack.emplace_back();
      interpreter.callStack.back().currentOp = &func.getBody().front().front();
      if (failed(interpreter.run()))
        return signalPassFailure();
      // The body was rewritten; update argNames and function_type to match.
      func.setArgNamesAttr(ArrayAttr::get(&getContext(), {}));
      SmallVector<Type> resultTypes;
      if (auto returnOp = func.getReturnOp())
        for (auto operand : returnOp.getOperands())
          resultTypes.push_back(operand.getType());
      func.setFunctionTypeAttr(mlir::TypeAttr::get(
          FunctionType::get(&getContext(), {}, resultTypes)));
      changed = true;
    }

    // Use split_func phase maps to chain evaluated results into the next
    // sub-function. The phaseNumbers and phaseFuncs are parallel arrays stored
    // in ascending phase order. When a phase's function is fully evaluated,
    // materialize its return constants as the trailing block arguments of the
    // next phase's function.
    for (auto splitFunc : getOperation().getOps<hir::SplitFuncOp>()) {
      auto phaseFuncs = splitFunc.getPhaseFuncs();
      for (unsigned i = 0; i + 1 < phaseFuncs.size(); ++i) {
        auto curSym = cast<FlatSymbolRefAttr>(phaseFuncs[i]);
        auto curFunc = symbolTable.lookup<mir::FuncOp>(curSym.getValue());
        if (!curFunc || curFunc.getBody().getNumArguments() > 0)
          continue;
        if (!isFullyEvaluated(curFunc))
          continue;

        auto returnConsts = collectReturnConstants(curFunc);
        if (returnConsts.empty() &&
            curFunc.getBody().front().getTerminator()->getNumOperands() > 0)
          continue;

        auto nextSym = cast<FlatSymbolRefAttr>(phaseFuncs[i + 1]);
        auto nextFunc = symbolTable.lookup<mir::FuncOp>(nextSym.getValue());
        if (!nextFunc)
          continue;

        // The chained values occupy the trailing block arguments. The leading
        // arguments are the "own" arguments of the next sub-function.
        unsigned numChained = returnConsts.size();
        if (numChained > nextFunc.getBody().getNumArguments())
          continue;
        unsigned numOwn = nextFunc.getBody().getNumArguments() - numChained;

        LLVM_DEBUG(llvm::dbgs()
                   << "Chaining " << curSym << " -> " << nextSym << "\n");
        auto &nextBlock = nextFunc.getBody().front();
        OpBuilder builder(&nextBlock, nextBlock.begin());
        for (unsigned j = 0; j < numChained; ++j) {
          auto arg = nextBlock.getArgument(numOwn + j);
          auto constOp = mir::ConstantOp::create(
              builder, arg.getLoc(), cast<TypedAttr>(returnConsts[j]));
          arg.replaceAllUsesWith(constOp);
        }
        nextBlock.eraseArguments(numOwn, numChained);
        // Update argNames and function_type to match the new block arg count.
        SmallVector<Attribute> newArgNames(nextFunc.getArgNames().begin(),
                                           nextFunc.getArgNames().end());
        if (newArgNames.size() > numOwn)
          newArgNames.resize(numOwn);
        nextFunc.setArgNamesAttr(ArrayAttr::get(&getContext(), newArgNames));
        SmallVector<Type> newArgTypes;
        for (unsigned j = 0; j < numOwn; ++j)
          newArgTypes.push_back(nextBlock.getArgument(j).getType());
        SmallVector<Type> resultTypes;
        if (auto returnOp = nextFunc.getReturnOp())
          for (auto operand : returnOp.getOperands())
            resultTypes.push_back(operand.getType());
        nextFunc.setFunctionTypeAttr(mlir::TypeAttr::get(
            FunctionType::get(&getContext(), newArgTypes, resultTypes)));
        changed = true;
      }
    }

    // Collapse multiphase_func ops whose sub-functions have all been evaluated.
    // When the first sub-function is fully evaluated, remove it from the list.
    // When only one sub-function remains, erase the multiphase_func entirely.
    for (auto mpFunc : llvm::make_early_inc_range(
             getOperation().getOps<hir::MultiphaseFuncOp>())) {
      auto phaseFuncs = mpFunc.getPhaseFuncs();

      // Count how many leading sub-functions have been fully evaluated.
      unsigned numDone = 0;
      for (auto sym : phaseFuncs) {
        auto func = symbolTable.lookup<mir::FuncOp>(
            cast<FlatSymbolRefAttr>(sym).getValue());
        if (!func || !isFullyEvaluated(func))
          break;
        ++numDone;
      }
      if (numDone == 0)
        continue;

      // If all or all-but-one sub-functions are done, erase the op entirely.
      if (phaseFuncs.size() - numDone <= 1) {
        mpFunc.erase();
      } else {
        SmallVector<Attribute> newPhaseFuncs(phaseFuncs.begin() + numDone,
                                             phaseFuncs.end());
        auto *ctx = mpFunc.getContext();
        mpFunc.setPhaseFuncsAttr(ArrayAttr::get(ctx, newPhaseFuncs));

        // Remove args that were classified as "first" since they belonged to
        // the now-evaluated sub-functions.
        auto argIsFirst = mpFunc.getArgIsFirst();
        SmallVector<StringRef> newArgNames;
        SmallVector<bool> newArgIsFirst;
        auto argNames = mpFunc.getArgNames();
        for (unsigned k = 0; k < argNames.size(); ++k) {
          if (!argIsFirst[k]) {
            newArgNames.push_back(cast<StringAttr>(argNames[k]).getValue());
            newArgIsFirst.push_back(false);
          }
        }
        // The first remaining "last" arg becomes "first" in the new ordering.
        if (!newArgIsFirst.empty())
          newArgIsFirst[0] = true;
        SmallVector<Attribute> nameAttrs;
        for (auto n : newArgNames)
          nameAttrs.push_back(StringAttr::get(ctx, n));
        mpFunc.setArgNamesAttr(ArrayAttr::get(ctx, nameAttrs));
        mpFunc.setArgIsFirstAttr(DenseBoolArrayAttr::get(ctx, newArgIsFirst));
      }
      changed = true;
    }
  }
}
