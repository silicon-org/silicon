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

static LogicalResult convert(hir::ConstantUnitOp op,
                             hir::ConstantUnitOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto attr = mir::UnitAttr::get(op.getContext());
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::ConstantFuncOp op,
                             hir::ConstantFuncOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Determine the function type.
  mir::TypeAttr funcTypeAttr;
  if (!matchPattern(adaptor.getFuncType(), m_Constant(&funcTypeAttr)))
    return emitBug(op.getLoc()) << "non-constant function type";

  auto funcType = dyn_cast<FunctionType>(funcTypeAttr.getValue());
  if (!funcType)
    return emitBug(op.getLoc()) << "non-function type";

  // Materialize the constant function.
  auto attr = mir::FuncAttr::get(
      op.getContext(), funcType,
      FlatSymbolRefAttr::get(op.getContext(), op.getValue()));
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

static LogicalResult convert(hir::AnyfuncTypeOp op,
                             hir::AnyfuncTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = mir::AnyfuncType::get(op.getContext());
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

static LogicalResult convert(hir::InferrableOp op,
                             hir::InferrableOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Lower an inferrable type placeholder to a type constant for !hir.any.
  // Phase-split code inserts `hir.inferrable` for result types that are not
  // yet resolved; these always carry !hir.any at the MLIR level, matching the
  // original behavior of the removed DirectCallOp.
  auto type = hir::AnyType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::TypeOfOp op, hir::TypeOfOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // TypeOfOp results are consumed as type metadata by BinaryOp and ReturnOp,
  // which discard them during lowering. Replace with a dummy type constant.
  auto type = hir::AnyType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::UnifyOp op, hir::UnifyOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // UnifyOp results are consumed as type metadata by BinaryOp, which discards
  // them during lowering. Replace with a dummy type constant.
  auto type = hir::AnyType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::CoerceTypeOp op,
                             hir::CoerceTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOp(op, adaptor.getInput());
  return success();
}

static LogicalResult convert(hir::BinaryOp op, hir::BinaryOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::BinaryOp>(op, adaptor.getLhs(),
                                             adaptor.getRhs());
  return success();
}

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
  rewriter.replaceOpWithNewOp<mir::ReturnOp>(op, adaptor.getValues());
  return success();
}

static LogicalResult convert(hir::CallOp op, hir::CallOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Validate argument types from constant type operands (not used directly;
  // the converted operand types are the ground truth).
  for (auto value : adaptor.getTypeOfArgs()) {
    mir::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant argument type";
  }

  // Extract result types from constant type operands and bake them into
  // the MIR call.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    mir::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant result type";
    resultTypes.push_back(attr.getValue());
  }

  rewriter.replaceOpWithNewOp<mir::CallOp>(op, resultTypes, op.getCalleeAttr(),
                                           adaptor.getArguments());
  return success();
}

static LogicalResult convert(hir::OpaquePackOp op,
                             hir::OpaquePackOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto opaqueType = mir::OpaqueType::get(op.getContext());
  rewriter.replaceOpWithNewOp<mir::MIROpaquePackOp>(op, opaqueType,
                                                    adaptor.getOperands());
  return success();
}

static LogicalResult convert(hir::OpaqueUnpackOp op,
                             hir::OpaqueUnpackOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::MIROpaqueUnpackOp>(op, op.getResultTypes(),
                                                      adaptor.getInput());
  return success();
}

static LogicalResult convert(UnrealizedConversionCastOp op,
                             UnrealizedConversionCastOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // Map MIR-to-HIR casts to the MIR input value directly.
  if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
    auto input = op.getOperand(0).getType();
    auto output = op.getResult(0).getType();
    if (isa<hir::HIRDialect>(output.getDialect())) {
      if (isa<mir::MIRDialect>(input.getDialect()) ||
          isa<FunctionType>(input)) {
        rewriter.replaceOp(op, adaptor.getOperands()[0]);
        return success();
      }
    }
  }
  return failure();
}

void HIRToMIRPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "Lowering @" << getOperation().getSymName()
                          << "\n");

  // Setup the type conversion.
  TypeConverter converter;

  // All MIR types, HIR types, and a handful of builtin types are fine.
  // HIR types (like !hir.any) are kept during partial lowering; they'll be
  // resolved in later passes.
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<mir::MIRDialect>(type.getDialect()) ||
        isa<hir::HIRDialect>(type.getDialect()) || isa<FunctionType>(type))
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
        if (isa<mir::MIRDialect>(type.getDialect()) || isa<FunctionType>(type))
          return type;
      }
    }
    return std::nullopt;
  });

  // Gather the conversion patterns.
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add<hir::ConstantIntOp>(convert);
  patterns.add<hir::ConstantUnitOp>(convert);
  patterns.add<hir::ConstantFuncOp>(convert);
  patterns.add<hir::IntTypeOp>(convert);
  patterns.add<hir::UIntTypeOp>(convert);
  patterns.add<hir::AnyfuncTypeOp>(convert);
  patterns.add<hir::FuncTypeOp>(convert);
  patterns.add<hir::InferrableOp>(convert);
  patterns.add<hir::TypeOfOp>(convert);
  patterns.add<hir::UnifyOp>(convert);
  patterns.add<hir::CoerceTypeOp>(convert);
  patterns.add<hir::BinaryOp>(convert);
  patterns.add<hir::SpecializeFuncOp>(convert);
  patterns.add<hir::ReturnOp>(convert);
  patterns.add<hir::CallOp>(convert);
  patterns.add<hir::OpaquePackOp>(convert);
  patterns.add<hir::OpaqueUnpackOp>(convert);
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
