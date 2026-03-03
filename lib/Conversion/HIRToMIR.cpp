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

static LogicalResult convert(hir::UnitTypeOp op,
                             hir::UnitTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = mir::UnitType::get(op.getContext());
  auto attr = mir::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::TypeTypeOp op,
                             hir::TypeTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = mir::TypeType::get(op.getContext());
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

//===----------------------------------------------------------------------===//
// HIRToMIR Pass
//
// For each hir.func in the module, convert its body ops from HIR to MIR via
// dialect conversion, then convert the hir.func shell itself to mir.func by
// resolving block argument types from unrealized_conversion_casts and gathering
// result types from the mir.return operands.
//===----------------------------------------------------------------------===//

void HIRToMIRPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Set up the type conversion.
  TypeConverter converter;
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<mir::MIRDialect>(type.getDialect()) ||
        isa<hir::HIRDialect>(type.getDialect()) || isa<FunctionType>(type))
      return type;
    return std::nullopt;
  });
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

  // Set up the conversion target and config, shared across all functions.
  ConversionTarget target(getContext());
  target.addLegalDialect<mir::MIRDialect>();
  target.addIllegalDialect<hir::HIRDialect>();
  target.addLegalOp<hir::FuncOp>();

  ConversionConfig config;
  config.allowPatternRollback = false;

  for (auto func : llvm::make_early_inc_range(moduleOp.getOps<hir::FuncOp>())) {
    LLVM_DEBUG(llvm::dbgs() << "Lowering @" << func.getSymName() << "\n");

    // Step 1: Convert body ops from HIR to MIR.
    ConversionPatternSet patterns(&getContext(), converter);
    patterns.add<hir::ConstantIntOp>(convert);
    patterns.add<hir::ConstantUnitOp>(convert);
    patterns.add<hir::ConstantFuncOp>(convert);
    patterns.add<hir::IntTypeOp>(convert);
    patterns.add<hir::UnitTypeOp>(convert);
    patterns.add<hir::TypeTypeOp>(convert);
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

    if (failed(
            applyFullConversion(func, target, std::move(patterns), config))) {
      emitBug(func.getLoc()) << "HIR-to-MIR lowering failed";
      return signalPassFailure();
    }

    // Step 2: Convert the hir.func shell to mir.func.
    // Resolve block argument types by folding unrealized_conversion_casts that
    // bridge from !hir.any to MIR types.
    if (func.getBody().empty())
      continue;
    auto &entryBlock = func.getBody().front();
    for (auto arg : entryBlock.getArguments()) {
      for (auto *user : llvm::make_early_inc_range(arg.getUsers())) {
        auto castOp = dyn_cast<UnrealizedConversionCastOp>(user);
        if (!castOp || castOp.getNumOperands() != 1 ||
            castOp.getNumResults() != 1)
          continue;
        auto inputType = castOp.getOperand(0).getType();
        auto resultType = castOp.getResult(0).getType();
        if (!isa<hir::HIRDialect>(inputType.getDialect()))
          continue;
        if (!isa<mir::MIRDialect>(resultType.getDialect()) &&
            !isa<FunctionType>(resultType))
          continue;
        // Replace the cast with the block arg and retype the arg.
        castOp.getResult(0).replaceAllUsesWith(arg);
        castOp.erase();
        arg.setType(resultType);
      }
    }

    // Gather arg types from the (now-retyped) block arguments.
    SmallVector<Type> argTypes;
    for (auto arg : entryBlock.getArguments())
      argTypes.push_back(arg.getType());

    // Gather result types from the mir.return operands.
    SmallVector<Type> resultTypes;
    if (auto returnOp =
            dyn_cast<mir::ReturnOp>(func.getBody().back().getTerminator()))
      for (auto operand : returnOp.getOperands())
        resultTypes.push_back(operand.getType());

    // Create mir::FuncOp with the materialized function type.
    auto funcType = FunctionType::get(&getContext(), argTypes, resultTypes);
    OpBuilder builder(func);
    auto mirFunc = mir::FuncOp::create(
        builder, func.getLoc(), func.getSymNameAttr(),
        func.getSymVisibilityAttr(), mlir::TypeAttr::get(funcType),
        func.getArgNamesAttr(), func.getResultNamesAttr());

    // Move the body from hir.func to mir.func and erase the old op.
    mirFunc.getBody().takeBody(func.getBody());
    func.erase();
  }
}
