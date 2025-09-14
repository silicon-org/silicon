//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Types.h"
#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "circt/Support/ConversionPatternSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using circt::ConversionPatternSet;

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

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

static LogicalResult convert(hir::ConstantIntOp op,
                             hir::ConstantIntOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto attr = mir::IntAttr::get(op.getContext(), op.getValue().getValue());
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::ConstantFuncOp op,
                             hir::ConstantFuncOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto attr = mir::FuncAttr::get(
      op.getContext(), FlatSymbolRefAttr::get(op.getContext(), op.getValue()));
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Constructors
//===----------------------------------------------------------------------===//

static LogicalResult convert(hir::IntTypeOp op, hir::IntTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = mir::IntType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::UIntTypeOp op,
                             hir::UIntTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto widthOperand = adaptor.getWidth();
  mir::IntAttr widthAttr;
  if (!matchPattern(widthOperand, m_Constant(&widthAttr)))
    return emitBug(op.getLoc()) << "non-constant uint width";
  const auto &width = widthAttr.getValue();

  // Make sure the width is zero or positive.
  if (width < 0)
    return emitBug(op.getLoc()) << "negative uint width " << width;

  // Make sure the width is not too large.
  if (width >= std::numeric_limits<int64_t>::max())
    return emitBug(op.getLoc()) << "excessive uint width " << width;

  // Materialize the constant uint type.
  auto type = mir::UIntType::get(op.getContext(), static_cast<int64_t>(width));
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::FuncTypeOp op,
                             hir::FuncTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Gather the argument types.
  SmallVector<Type> argTypes;
  argTypes.reserve(adaptor.getTypeOfArgs().size());
  for (auto value : adaptor.getTypeOfArgs()) {
    mir::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant argument type";
    argTypes.push_back(attr.getValue());
  }

  // Gather the result types.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    mir::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant result type";
    resultTypes.push_back(attr.getValue());
  }

  // Materialize the constant function type.
  auto type = FunctionType::get(op.getContext(), argTypes, resultTypes);
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

static LogicalResult convert(hir::SpecializeFuncOp op,
                             hir::SpecializeFuncOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::SpecializeFuncOp>(
      op, op.getFuncAttr(), adaptor.getTypeOfArgs(), adaptor.getTypeOfResults(),
      adaptor.getConsts());
  return success();
}

static LogicalResult convert(hir::ReturnOp op, hir::ReturnOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::ReturnOp>(op, adaptor.getOperands());
  return success();
}

static LogicalResult convert(hir::CallOp op, hir::CallOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto constCallee = adaptor.getCallee().getDefiningOp<mir::ConstantOp>();
  if (!constCallee)
    return failure();
  auto calleeAttr = dyn_cast<mir::FuncAttr>(constCallee.getValue());
  if (!calleeAttr)
    return failure();
  rewriter.replaceOpWithNewOp<mir::CallOp>(
      op, TypeRange{mir::SpecializedFuncType::get(rewriter.getContext())},
      calleeAttr.getFunc(), adaptor.getArguments());
  return success();
}

static LogicalResult convert(UnrealizedConversionCastOp op,
                             UnrealizedConversionCastOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Map MIR-to-HIR casts to the MIR input value directly.
  if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
    if (isa<hir::HIRDialect>(op.getResult(0).getType().getDialect())) {
      if (isa<mir::MIRDialect>(op.getOperand(0).getType().getDialect())) {
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
      }
    }
  }
  return failure();
}

void HIRToMIRPass::runOnOperation() {
  if (llvm::any_of(getOperation().getBody().getArgumentTypes(), [](auto type) {
        return !isa<mir::MIRDialect>(type.getDialect());
      }))
    return;
  LLVM_DEBUG(llvm::dbgs() << "Lowering @" << getOperation().getSymName()
                          << "\n");

  // Setup the type conversion.
  TypeConverter converter;

  // All MIR types are fine.
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<mir::MIRDialect>(type.getDialect()))
      return type;
    return std::nullopt;
  });

  // Allow any casts from MIR types. This is very relaxed, but allows us to get
  // something up and running quickly. In the future, we'll want to take the MIR
  // type and map it to the corresponding HIR type, and only allow casts that
  // make sense.
  converter.addConversion([](Value value) -> std::optional<Type> {
    if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
        auto type = castOp.getOperand(0).getType();
        if (isa<mir::MIRDialect>(type.getDialect()))
          return type;
      }
    }
    return std::nullopt;
  });

  // Gather the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add<hir::ConstantIntOp>(convert);
  patterns.add<hir::ConstantFuncOp>(convert);
  patterns.add<hir::IntTypeOp>(convert);
  patterns.add<hir::UIntTypeOp>(convert);
  patterns.add<hir::FuncTypeOp>(convert);
  patterns.add<hir::SpecializeFuncOp>(convert);
  patterns.add<hir::ReturnOp>(convert);
  patterns.add<hir::CallOp>(convert);
  patterns.add<UnrealizedConversionCastOp>(convert);

  // Setup the legal ops.
  ConversionTarget target(getContext());
  target.addLegalDialect<mir::MIRDialect>();
  target.addIllegalDialect<hir::HIRDialect>();
  target.addLegalOp<hir::FuncOp>();

  // Disable pattern rollback to use the faster one-shot dialect conversion.
  ConversionConfig config;
  config.allowPatternRollback = false;

  if (failed(applyFullConversion(getOperation(), target, std::move(patterns),
                                 config))) {
    emitBug(getOperation().getLoc()) << "HIR-to-MIR lowering failed";
    return signalPassFailure();
  }
}
