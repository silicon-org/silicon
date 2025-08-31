//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "hir-to-mir"

namespace silicon {
#define GEN_PASS_DEF_HIRTOMIRPASS
#include "silicon/Conversion/Passes.h.inc"
} // namespace silicon

namespace {
struct HIRToMIRPass : public silicon::impl::HIRToMIRPassBase<HIRToMIRPass> {
  void runOnOperation() override;
};
} // namespace

static LogicalResult convert(hir::IntTypeOp op, PatternRewriter &rewriter) {
  auto type = mir::IntType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::ConstantIntOp op, PatternRewriter &rewriter) {
  auto attr = mir::IntAttr::get(op.getContext(), op.getValue().getValue());
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::ReturnOp op, PatternRewriter &rewriter) {
  auto func = op->getParentOfType<hir::FuncOp>();
  assert(func);

  auto specFunc = mir::SpecializeFuncOp::create(
      rewriter, op.getLoc(), FlatSymbolRefAttr::get(func.getSymNameAttr()),
      op.getArgs(), ValueRange{}, op.getFreeze());

  // auto attr = mir::IntAttr::get(op.getContext(), op.getValue().getValue());
  rewriter.replaceOpWithNewOp<mir::ReturnOp>(op, ValueRange{specFunc});
  return success();
}

static LogicalResult convert(hir::CallOp op, PatternRewriter &rewriter) {
  auto constCallee = op.getCallee().getDefiningOp<hir::ConstantFuncOp>();
  if (!constCallee)
    return failure();
  rewriter.replaceOpWithNewOp<mir::CallOp>(
      op, TypeRange{mir::SpecializedFuncType::get(rewriter.getContext())},
      constCallee.getValue(), op.getArguments());
  return success();
}

void HIRToMIRPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering first region of @"
                          << getOperation().getSymName() << "\n");

  RewritePatternSet patterns(&getContext());
  patterns.add<hir::IntTypeOp>(convert);
  patterns.add<hir::ConstantIntOp>(convert);
  patterns.add<hir::ReturnOp>(convert);
  patterns.add<hir::CallOp>(convert);

  if (failed(applyPatternsGreedily(getOperation().getBodies()[0],
                                   std::move(patterns)))) {
    emitBug(getOperation().getLoc()) << "HIR-to-MIR lowering failed to "
                                        "converge";
    return signalPassFailure();
  }
}
