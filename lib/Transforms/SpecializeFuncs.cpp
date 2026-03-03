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
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "specialize-funcs"

namespace silicon {
#define GEN_PASS_DEF_SPECIALIZEFUNCSPASS
#include "silicon/Transforms/Passes.h.inc"
} // namespace silicon

namespace {
struct SpecializeFuncsPass
    : public silicon::impl::SpecializeFuncsPassBase<SpecializeFuncsPass> {
  void runOnOperation() override;
  mir::FuncOp specialize(mir::FuncOp originalFunc,
                         mir::SpecializedFuncAttr spec);
};
} // namespace

void SpecializeFuncsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  DenseMap<mir::SpecializedFuncAttr, mir::FuncOp> funcs;
  SmallVector<Operation *> worklist;
  worklist.push_back(getOperation());
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    auto result = op->walk([&](mir::ConstantOp op) {
      auto attr = dyn_cast<mir::SpecializedFuncAttr>(op.getValue());
      if (!attr)
        return WalkResult::advance();
      auto &func = funcs[attr];
      if (!func) {
        LLVM_DEBUG(llvm::dbgs() << "Specializing " << attr << "\n");
        func = specialize(
            symbolTable.lookup<mir::FuncOp>(attr.getFunc().getAttr()), attr);
        if (!func)
          return WalkResult::interrupt();
        symbolTable.insert(func);
        worklist.push_back(func);
      }
      OpBuilder builder(op);
      auto newAttr = mir::FuncAttr::get(
          &getContext(),
          FunctionType::get(&getContext(), attr.getArgs(), attr.getResults()),
          FlatSymbolRefAttr::get(func.getSymNameAttr()));
      auto newOp = mir::ConstantOp::create(builder, op.getLoc(), newAttr);
      LLVM_DEBUG(llvm::dbgs()
                 << "Replacing " << op << " with " << newOp << "\n");
      op.getResult().replaceAllUsesWith(newOp);
      op.erase();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Specialize a mir::FuncOp
//
// Clones the original function, materializes constant arguments at the start
// of the body, erases the corresponding block arguments, and updates the
// function_type to reflect the remaining arguments.
//===----------------------------------------------------------------------===//

mir::FuncOp SpecializeFuncsPass::specialize(mir::FuncOp originalFunc,
                                            mir::SpecializedFuncAttr spec) {
  // Make sure we have the correct number of arguments.
  auto argsExpected = spec.getArgs().size() + spec.getConsts().size();
  auto argsActual = originalFunc.getBody().getNumArguments();
  if (argsExpected != argsActual) {
    emitBug(originalFunc.getLoc())
        << "function has " << argsActual
        << " arguments, but specialization expects " << argsExpected << " ("
        << spec.getArgs().size() << " args and " << spec.getConsts().size()
        << " consts)";
    return {};
  }

  // Create a clone of the function.
  OpBuilder builder(originalFunc);
  builder.setInsertionPointAfter(originalFunc);
  auto func = cast<mir::FuncOp>(builder.clone(*originalFunc));

  // Skip past the arg-typed arguments to the const arguments.
  auto &block = func.getBody().front();
  builder.setInsertionPointToStart(&block);
  unsigned argIdx = spec.getArgs().size();

  // Materialize constant arguments. Since block args now have MIR types, the
  // constant's type should match the block arg's type directly.
  auto firstConstIdx = argIdx;
  for (auto attr : spec.getConsts()) {
    auto arg = block.getArgument(argIdx++);
    auto constOp =
        mir::ConstantOp::create(builder, arg.getLoc(), cast<TypedAttr>(attr));
    arg.replaceAllUsesWith(constOp);
  }
  block.eraseArguments(firstConstIdx, argIdx - firstConstIdx);

  // Update argNames to match the new argument count after erasing consts.
  SmallVector<Attribute> newArgNames(func.getArgNames().begin(),
                                     func.getArgNames().end());
  newArgNames.erase(newArgNames.begin() + firstConstIdx,
                    newArgNames.begin() + argIdx);
  func.setArgNamesAttr(builder.getArrayAttr(newArgNames));

  // Update function_type to reflect the remaining arg types and result types.
  SmallVector<Type> newArgTypes;
  for (auto arg : block.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionTypeAttr(mlir::TypeAttr::get(FunctionType::get(
      &getContext(), newArgTypes, func.getFunctionType().getResults())));

  return func;
}
