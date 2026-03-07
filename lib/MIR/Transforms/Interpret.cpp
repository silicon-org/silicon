//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/MIR/Passes.h"
#include "silicon/Support/MLIR.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "interpret"

namespace silicon {
namespace mir {
#define GEN_PASS_DEF_INTERPRETPASS
#include "silicon/MIR/Passes.h.inc"
} // namespace mir
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
  FailureOr<SmallVector<Attribute>> run();

  /// Execute a single non-control-flow op within the given value map. Returns
  /// failure if the op is unsupported. Handles constants, binary ops, opaque
  /// pack/unpack, if ops, and conversion casts.
  LogicalResult executeOp(Operation *op, ArrayRef<Attribute> operands,
                          DenseMap<Value, Attribute> &values);
};
} // namespace

//===----------------------------------------------------------------------===//
// Interpreter Core
//
// Walk through the operations of a function body, evaluating each one.
// Calls are resolved by pushing a new frame onto the call stack. When the
// top-most frame encounters a return, the constant attribute values are
// returned without modifying the IR. The executeOp helper handles individual
// non-control-flow ops and is reused for evaluating ops inside if regions.
//===----------------------------------------------------------------------===//

/// Execute a single non-control-flow op, storing results in the value map.
LogicalResult Interpreter::executeOp(Operation *op,
                                     ArrayRef<Attribute> operands,
                                     DenseMap<Value, Attribute> &values) {
  if (auto constOp = dyn_cast<mir::ConstantOp>(op)) {
    values[constOp] = constOp.getValue();
  } else if (isa<mir::AddOp, mir::SubOp, mir::MulOp, mir::DivOp, mir::ModOp,
                 mir::AndOp, mir::OrOp, mir::XorOp, mir::ShlOp, mir::ShrOp,
                 mir::EqOp, mir::NeqOp, mir::LtOp, mir::GtOp, mir::GeqOp,
                 mir::LeqOp>(op)) {
    auto lhs = cast<base::IntAttr>(operands[0]).getValue();
    auto rhs = cast<base::IntAttr>(operands[1]).getValue();
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
    values[op->getResult(0)] = base::IntAttr::get(op->getContext(), value);
  } else if (auto ifOp = dyn_cast<mir::IfOp>(op)) {
    // Evaluate the condition and execute the chosen region's block.
    auto condAttr = cast<base::IntAttr>(operands[0]);
    bool condTrue = condAttr.getValue() != 0;
    auto &region = condTrue ? ifOp.getThenRegion() : ifOp.getElseRegion();

    for (auto &innerOp : region.front()) {
      if (auto yieldOp = dyn_cast<mir::YieldOp>(&innerOp)) {
        for (auto [result, yieldOperand] :
             llvm::zip(ifOp.getResults(), yieldOp.getOperands()))
          values[result] = values.lookup(yieldOperand);
        break;
      }

      // Gather operands for the inner op.
      SmallVector<Attribute> innerOperands(innerOp.getNumOperands());
      for (auto [attr, operand] :
           llvm::zip(innerOperands, innerOp.getOpOperands())) {
        attr = values.lookup(operand.get());
        if (!attr)
          return innerOp.emitError()
                 << "missing value for operand #" << operand.getOperandNumber();
      }

      if (failed(executeOp(&innerOp, innerOperands, values)))
        return failure();
    }
  } else if (auto packOp = dyn_cast<mir::MIROpaquePackOp>(op)) {
    values[packOp] = base::OpaqueAttr::get(op->getContext(), operands);
  } else if (auto unpackOp = dyn_cast<mir::MIROpaqueUnpackOp>(op)) {
    auto opaqueAttr = cast<base::OpaqueAttr>(operands[0]);
    for (auto [result, elem] :
         llvm::zip(unpackOp.getResults(), opaqueAttr.getElements()))
      values[result] = elem;
  } else if (auto castOp = dyn_cast<UnrealizedConversionCastOp>(op)) {
    for (auto [result, attr] : llvm::zip(castOp.getResults(), operands))
      values[result] = attr;
  } else {
    return op->emitError() << "operation not supported by interpreter";
  }
  return success();
}

FailureOr<SmallVector<Attribute>> Interpreter::run() {
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
      if (!calleeOp)
        return op->emitError()
               << "callee @" << callOp.getCallee()
               << " is not a mir.func (may not have been lowered yet)";
      auto &newFrame = callStack.emplace_back();
      newFrame.currentOp = &calleeOp.getBody().front().front();
      for (auto [arg, attr] :
           llvm::zip(calleeOp.getBody().getArguments(), operands))
        newFrame.values[arg] = attr;
      continue;
    }

    // Special handling for return operations.
    if (auto returnOp = dyn_cast<mir::ReturnOp>(op)) {
      // Top-most frame: return the result attributes.
      if (callStack.size() == 1)
        return SmallVector<Attribute>(operands);

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

    // Handle all other operations via the shared executeOp helper.
    if (failed(executeOp(op, operands, frame.values)))
      return failure();

    frame.currentOp = op->getNextNode();
  }
  return SmallVector<Attribute>{};
}

//===----------------------------------------------------------------------===//
// InterpretPass
//
// Evaluates zero-argument MIR functions and produces `mir.evaluated_func` ops
// with the constant result attributes. The original `mir.func` is erased.
// Chaining and specialization are handled separately by SpecializeFuncsPass.
//===----------------------------------------------------------------------===//

namespace {
struct InterpretPass
    : public silicon::mir::impl::InterpretPassBase<InterpretPass> {
  void runOnOperation() override;
};
} // namespace

/// Check whether a function transitively calls any module function. Module
/// functions represent hardware and cannot be interpreted, so any caller that
/// depends on a module result must also be kept as a function.
static bool callsModule(mir::FuncOp func, SymbolTable &symbolTable) {
  SmallVector<mir::FuncOp> worklist = {func};
  DenseSet<Operation *> visited;
  while (!worklist.empty()) {
    auto current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;
    bool found = false;
    current.walk([&](mir::CallOp callOp) {
      if (found)
        return;
      auto callee = symbolTable.lookup<mir::FuncOp>(callOp.getCallee());
      if (!callee)
        return;
      if (callee.getIsModuleAttr()) {
        found = true;
        return;
      }
      worklist.push_back(callee);
    });
    if (found)
      return true;
  }
  return false;
}

void InterpretPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Collect functions to evaluate and their results. We collect first to avoid
  // invalidating the symbol table during iteration (callees must remain
  // available while being resolved by the interpreter).
  SmallVector<std::pair<mir::FuncOp, SmallVector<Attribute>>> evaluated;

  for (auto func : getOperation().getOps<mir::FuncOp>()) {
    if (func.getBody().getNumArguments() > 0)
      continue;
    if (!func.getBody().front().getTerminator() ||
        !isa<mir::ReturnOp>(func.getBody().front().getTerminator()))
      continue;
    // Module functions represent hardware and must not be interpreted away.
    // Functions that transitively call modules also cannot be interpreted,
    // since their results depend on hardware outputs.
    if (func.getIsModuleAttr() || callsModule(func, symbolTable))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Interpreting @" << func.getSymName() << "\n");
    Interpreter interpreter(symbolTable);
    interpreter.callStack.emplace_back();
    interpreter.callStack.back().currentOp = &func.getBody().front().front();
    auto result = interpreter.run();
    if (failed(result))
      return signalPassFailure();

    evaluated.push_back({func, std::move(*result)});
  }

  // Replace each evaluated function with a mir.evaluated_func op.
  for (auto &[func, resultAttrs] : evaluated) {
    LLVM_DEBUG(llvm::dbgs() << "Evaluated @" << func.getSymName() << " -> [");
    LLVM_DEBUG(llvm::interleaveComma(resultAttrs, llvm::dbgs()));
    LLVM_DEBUG(llvm::dbgs() << "]\n");

    OpBuilder builder(func);
    mir::EvaluatedFuncOp::create(builder, func.getLoc(), func.getSymNameAttr(),
                                 func.getSymVisibilityAttr(),
                                 builder.getArrayAttr(resultAttrs));
    symbolTable.erase(func);
  }
}
