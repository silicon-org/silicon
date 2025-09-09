//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Types.h"
#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/STLExtras.h"
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

static LogicalResult convert(hir::SpecializeFuncOp op,
                             PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::SpecializeFuncOp>(
      op, op.getFuncAttr(), op.getTypeOfArgs(), op.getTypeOfResults(),
      op.getConsts());
  return success();
}

static LogicalResult convert(hir::ReturnOp op, PatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::ReturnOp>(op, op.getOperands());
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

static LogicalResult convert(UnrealizedConversionCastOp op,
                             PatternRewriter &rewriter) {
  if (op.getNumResults() != 1 || op.getNumOperands() != 1)
    return failure();
  if (!isa<hir::ValueType, hir::TypeType, hir::FuncType>(
          op.getResult(0).getType()) ||
      !isa<mir::MIRDialect>(op.getOperand(0).getType().getDialect()))
    return failure();
  rewriter.replaceOp(op, op.getOperands());
  return success();
}

void HIRToMIRPass::runOnOperation() {
  if (llvm::any_of(getOperation().getBody().getArgumentTypes(), [](auto type) {
        return !isa<mir::MIRDialect>(type.getDialect());
      }))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Lowering @" << getOperation().getSymName()
                          << "\n");

  RewritePatternSet patterns(&getContext());
  patterns.add<hir::IntTypeOp>(convert);
  patterns.add<hir::ConstantIntOp>(convert);
  patterns.add<hir::SpecializeFuncOp>(convert);
  patterns.add<hir::ReturnOp>(convert);
  patterns.add<hir::CallOp>(convert);
  patterns.add<UnrealizedConversionCastOp>(convert);

  if (failed(applyPatternsGreedily(getOperation().getBody(),
                                   std::move(patterns)))) {
    emitBug(getOperation().getLoc()) << "HIR-to-MIR lowering failed to "
                                        "converge";
    return signalPassFailure();
  }
}
