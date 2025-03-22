//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/Liveness.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "silicon/Dialect/HIR/HIROps.h"
#include "silicon/Dialect/HIR/HIRPasses.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/GenericIteratedDominanceFrontier.h"

#define DEBUG_TYPE "eval-const-regions"

using namespace mlir;
using namespace silicon;
using namespace hir;

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_EVALCONSTREGIONSPASS
#include "silicon/Dialect/HIR/HIRPasses.h.inc"
} // namespace hir
} // namespace silicon

namespace {
struct EvalConstRegionsPass
    : public hir::impl::EvalConstRegionsPassBase<EvalConstRegionsPass> {
  void runOnOperation() override;
  LogicalResult handle(ConstRegionOp op, ModuleOp jitModule,
                       SymbolTable &jitSymbols);
  LogicalResult handle(func::FuncOp op, ModuleOp jitModule,
                       SymbolTable &jitSymbols);
  LogicalResult updateCalls(Operation *op);

  DenseMap<StringAttr, StringAttr> outlinedFunctions;
};
} // namespace

void EvalConstRegionsPass::runOnOperation() {
  auto builder = OpBuilder::atBlockBegin(getOperation().getBody());
  auto jitModule = builder.create<ModuleOp>(getOperation().getLoc());
  auto jitSymbols = SymbolTable(jitModule);

  // Perform a depth-first search through the call graph and const regions.
  SmallVector<Operation *, 0> worklist;
  SmallPtrSet<Operation *, 16> visited;
  std::function<void(Operation *)> collectWorklist = [&](Operation *op) {
    for (auto &region : op->getRegions())
      for (auto &block : region)
        for (auto &op : block)
          collectWorklist(&op);
    if (auto callOp = dyn_cast<CallOpInterface>(op))
      if (visited.insert(op).second)
        collectWorklist(callOp.resolveCallable());
    if (isa<ConstRegionOp, func::FuncOp>(op))
      if (visited.insert(op).second)
        worklist.push_back(op);
  };
  getOperation().walk([&](ConstRegionOp op) { collectWorklist(op); });

  // Process elements of the worklist.
  for (auto *op : worklist) {
    llvm::dbgs() << "Processing ";
    op->print(llvm::dbgs(), OpPrintingFlags().skipRegions());
    llvm::dbgs() << "\n";
    LogicalResult result = TypeSwitch<Operation *, LogicalResult>(op)
                               .Case<ConstRegionOp, func::FuncOp>([&](auto op) {
                                 return handle(op, jitModule, jitSymbols);
                               });
    if (failed(result))
      return signalPassFailure();
  }
}

LogicalResult EvalConstRegionsPass::handle(ConstRegionOp op, ModuleOp jitModule,
                                           SymbolTable &jitSymbols) {
  if (op.getNumResults() != 1 ||
      !op.getResult(0).getType().isSignlessInteger(32))
    return success();

  SmallString<128> name;
  name += "const_region";
  Operation *parent = op;
  while (parent) {
    if (auto nameAttr = parent->getAttrOfType<StringAttr>("sym_name")) {
      name += '.';
      name += nameAttr.getValue();
    }
    parent = parent->getParentOp();
  }

  // Outline the region to a function in the JIT module.
  auto builder = OpBuilder::atBlockEnd(jitModule.getBody());
  auto clonedOp = cast<ConstRegionOp>(builder.clone(*op));
  auto funcType = builder.getFunctionType({}, op.getResultTypes());
  auto funcOp = builder.create<func::FuncOp>(op.getLoc(), name, funcType);
  funcOp->setAttr(LLVM::LLVMDialect::getEmitCWrapperAttrName(),
                  UnitAttr::get(&getContext()));
  funcOp.getBody().takeBody(clonedOp.getBody());
  clonedOp.erase();
  for (auto &block : funcOp.getBody()) {
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp)
      continue;
    builder.setInsertionPoint(yieldOp);
    builder.create<func::ReturnOp>(yieldOp.getLoc(), yieldOp.getResults());
    yieldOp.erase();
  }
  jitSymbols.insert(funcOp);
  auto funcName = funcOp.getSymName();
  if (failed(updateCalls(funcOp)))
    return failure();

  // Lower the outlined module to LLVM IR.
  LLVMConversionTarget target(getContext());
  LLVMTypeConverter converter(&getContext());
  RewritePatternSet patterns(&getContext());
  target.addLegalOp<mlir::ModuleOp>();
  populateFuncToLLVMConversionPatterns(converter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);
  arith::populateArithToLLVMConversionPatterns(converter, patterns);
  index::populateIndexToLLVMConversionPatterns(converter, patterns);
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);
  if (failed(applyFullConversion(jitModule, target, std::move(patterns))))
    signalPassFailure();

  // Call the outlined function.
  ExecutionEngineOptions engineOptions;
  // engineOptions.jitCodeGenOptLevel = llvm::CodeGenOptLevel::Aggressive;
  // engineOptions.transformer = makeOptimizingTransformer(
  //     /*optLevel=*/3, /*sizeLevel=*/0,
  //     /*targetMachine=*/nullptr);
  // engineOptions.sharedLibPaths = sharedLibraries;

  auto executionEngine = ExecutionEngine::create(jitModule, engineOptions);
  if (!executionEngine)
    return op.emitError() << "cannot create execution engine";
  int32_t result = 0;
  auto error =
      (*executionEngine)->invoke(funcName, ExecutionEngine::result(result));
  if (error)
    return op.emitError() << "cannot invoke " << funcName
                          << " in execution engine";
  LLVM_DEBUG(llvm::dbgs() << "Got result " << result << "\n");

  builder.setInsertionPoint(op);
  auto constOp = builder.create<arith::ConstantOp>(
      op.getLoc(), builder.getI32IntegerAttr(result));
  op.replaceAllUsesWith(constOp);
  op.erase();
  return success();
}

LogicalResult EvalConstRegionsPass::handle(func::FuncOp op, ModuleOp jitModule,
                                           SymbolTable &jitSymbols) {
  auto builder = OpBuilder::atBlockEnd(jitModule.getBody());
  auto clonedOp = cast<func::FuncOp>(builder.clone(*op));
  jitSymbols.insert(clonedOp);
  outlinedFunctions.insert({op.getSymNameAttr(), clonedOp.getSymNameAttr()});
  return updateCalls(clonedOp);
}

LogicalResult EvalConstRegionsPass::updateCalls(Operation *op) {
  for (auto &region : op->getRegions()) {
    for (auto &block : region) {
      for (auto callOp : block.getOps<func::CallOp>()) {
        auto &outlinedFunction =
            outlinedFunctions[callOp.getCalleeAttr().getAttr()];
        if (!outlinedFunction) {
          return callOp.emitError() << "function " << callOp.getCalleeAttr()
                                    << " has not been outlined";
        }
        callOp.setCalleeAttr(FlatSymbolRefAttr::get(outlinedFunction));
      }
    }
  }
  return success();
}
