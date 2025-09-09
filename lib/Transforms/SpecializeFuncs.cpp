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
  hir::FuncOp specialize(hir::FuncOp originalFunc,
                         mir::SpecializedFuncAttr spec);
};
} // namespace

void SpecializeFuncsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();
  DenseMap<mir::SpecializedFuncAttr, hir::FuncOp> funcs;
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
            symbolTable.lookup<hir::FuncOp>(attr.getFunc().getAttr()), attr);
        if (!func)
          return WalkResult::interrupt();
        symbolTable.insert(func);
        worklist.push_back(func);
      }
      OpBuilder builder(op);
      auto newOp = hir::ConstantFuncOp::create(builder, op.getLoc(),
                                               func.getSymNameAttr());
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

hir::FuncOp SpecializeFuncsPass::specialize(hir::FuncOp originalFunc,
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
  auto func = cast<hir::FuncOp>(builder.clone(*originalFunc));

  // Apply argument type specialization.
  auto &block = func.getBody().front();
  builder.setInsertionPointToStart(&block);
  unsigned argIdx = 0;
  for (auto type : spec.getArgs()) {
    auto arg = block.getArgument(argIdx++);
    auto castOp = UnrealizedConversionCastOp::create(
        builder, arg.getLoc(), arg.getType(), ValueRange{arg});
    arg.setType(type);
    arg.replaceAllUsesExcept(castOp.getResult(0), castOp);
  }

  // Materialize constant arguments.
  auto firstConstIdx = argIdx;
  for (auto attr : spec.getConsts()) {
    auto arg = block.getArgument(argIdx++);
    auto constOp =
        mir::ConstantOp::create(builder, arg.getLoc(), cast<TypedAttr>(attr));
    if (constOp.getType() == arg.getType()) {
      arg.replaceAllUsesWith(constOp);
      continue;
    }
    auto castOp = UnrealizedConversionCastOp::create(
        builder, arg.getLoc(), arg.getType(), ValueRange{constOp});
    arg.replaceAllUsesExcept(castOp.getResult(0), castOp);
  }
  block.eraseArguments(firstConstIdx, argIdx - firstConstIdx);

  return func;
}
