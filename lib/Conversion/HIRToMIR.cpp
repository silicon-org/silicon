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

// CoerceTypeOp annotates a value with a type. If the type operand resolves to
// a mir.constant carrying a TypeAttr, extract the concrete MIR type and emit an
// unrealized_conversion_cast. The Step 2 fixup below folds these casts into
// block argument retypes. We use matchPattern/m_Constant to look through any
// materialization casts the conversion framework may have inserted.
static LogicalResult convert(hir::CoerceTypeOp op,
                             hir::CoerceTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  mir::TypeAttr typeAttr;
  if (matchPattern(adaptor.getTypeOperand(), m_Constant(&typeAttr))) {
    auto mirType = typeAttr.getValue();
    if (!isa<hir::AnyType>(mirType)) {
      rewriter.replaceOpWithNewOp<UnrealizedConversionCastOp>(
          op, mirType, adaptor.getInput());
      return success();
    }
  }
  rewriter.replaceOp(op, adaptor.getInput());
  return success();
}

// Arithmetic/bitwise HIR binary ops lower to the corresponding MIR ops by
// discarding the type operand and forwarding the converted lhs/rhs operands.
// Comparison ops additionally pass the result type explicitly since operand
// and result types may differ in MIR (e.g. uint<1> for comparisons).
//
// When one operand has been retyped to a concrete MIR type (via coerce_type)
// and the other is still !hir.any (e.g. from opaque_unpack), we insert an
// unrealized_conversion_cast on the !hir.any operand to match. Step 2 folds
// these casts by retyping the source value.

/// If lhs and rhs have mismatched types where one is !hir.any and the other
/// is a concrete MIR type, insert an unrealized_conversion_cast on the
/// !hir.any operand to match. These casts are folded in Step 2.
static void reconcileOperandTypes(ConversionPatternRewriter &rewriter,
                                  Location loc, Value &lhs, Value &rhs) {
  if (lhs.getType() == rhs.getType())
    return;
  if (isa<hir::AnyType>(lhs.getType()) && !isa<hir::AnyType>(rhs.getType())) {
    lhs = UnrealizedConversionCastOp::create(rewriter, loc, rhs.getType(), lhs)
              .getResult(0);
  } else if (!isa<hir::AnyType>(lhs.getType()) &&
             isa<hir::AnyType>(rhs.getType())) {
    rhs = UnrealizedConversionCastOp::create(rewriter, loc, lhs.getType(), rhs)
              .getResult(0);
  }
}

#define CONVERT_BINARY_OP(HirOp, MirOp)                                        \
  static LogicalResult convert(hir::HirOp op, hir::HirOp::Adaptor adaptor,     \
                               ConversionPatternRewriter &rewriter) {          \
    Value lhs = adaptor.getLhs();                                              \
    Value rhs = adaptor.getRhs();                                              \
    reconcileOperandTypes(rewriter, op.getLoc(), lhs, rhs);                    \
    rewriter.replaceOpWithNewOp<mir::MirOp>(op, lhs, rhs);                     \
    return success();                                                          \
  }

#define CONVERT_CMP_OP(HirOp, MirOp)                                           \
  static LogicalResult convert(hir::HirOp op, hir::HirOp::Adaptor adaptor,     \
                               ConversionPatternRewriter &rewriter) {          \
    Value lhs = adaptor.getLhs();                                              \
    Value rhs = adaptor.getRhs();                                              \
    reconcileOperandTypes(rewriter, op.getLoc(), lhs, rhs);                    \
    rewriter.replaceOpWithNewOp<mir::MirOp>(op, lhs.getType(), lhs, rhs);      \
    return success();                                                          \
  }

CONVERT_BINARY_OP(AddOp, AddOp)
CONVERT_BINARY_OP(SubOp, SubOp)
CONVERT_BINARY_OP(MulOp, MulOp)
CONVERT_BINARY_OP(DivOp, DivOp)
CONVERT_BINARY_OP(ModOp, ModOp)
CONVERT_BINARY_OP(AndOp, AndOp)
CONVERT_BINARY_OP(OrOp, OrOp)
CONVERT_BINARY_OP(XorOp, XorOp)
CONVERT_BINARY_OP(ShlOp, ShlOp)
CONVERT_BINARY_OP(ShrOp, ShrOp)
CONVERT_CMP_OP(EqOp, EqOp)
CONVERT_CMP_OP(NeqOp, NeqOp)
CONVERT_CMP_OP(LtOp, LtOp)
CONVERT_CMP_OP(GtOp, GtOp)
CONVERT_CMP_OP(GeqOp, GeqOp)
CONVERT_CMP_OP(LeqOp, LeqOp)

#undef CONVERT_BINARY_OP
#undef CONVERT_CMP_OP

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
  // Extract result types from constant type operands and bake them into
  // the MIR call. Fall back to !hir.any for unresolvable types (e.g. type
  // operands threaded via opaque_unpack that haven't been specialized yet).
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    mir::TypeAttr attr;
    if (matchPattern(value, m_Constant(&attr)))
      resultTypes.push_back(attr.getValue());
    else
      resultTypes.push_back(hir::AnyType::get(op.getContext()));
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

  // Allow HIR→MIR casts to persist through conversion. CoerceTypeOp creates
  // these to annotate block args with concrete MIR types; Step 2 folds them
  // into block arg retypes.
  target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
      [](UnrealizedConversionCastOp op) {
        if (op.getNumOperands() == 1 && op.getNumResults() == 1) {
          auto inputType = op.getOperand(0).getType();
          auto resultType = op.getResult(0).getType();
          if (isa<hir::HIRDialect>(inputType.getDialect()) &&
              (isa<mir::MIRDialect>(resultType.getDialect()) ||
               isa<FunctionType>(resultType)))
            return true;
        }
        return false;
      });

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
    patterns.add<hir::AddOp>(convert);
    patterns.add<hir::SubOp>(convert);
    patterns.add<hir::MulOp>(convert);
    patterns.add<hir::DivOp>(convert);
    patterns.add<hir::ModOp>(convert);
    patterns.add<hir::AndOp>(convert);
    patterns.add<hir::OrOp>(convert);
    patterns.add<hir::XorOp>(convert);
    patterns.add<hir::ShlOp>(convert);
    patterns.add<hir::ShrOp>(convert);
    patterns.add<hir::EqOp>(convert);
    patterns.add<hir::NeqOp>(convert);
    patterns.add<hir::LtOp>(convert);
    patterns.add<hir::GtOp>(convert);
    patterns.add<hir::GeqOp>(convert);
    patterns.add<hir::LeqOp>(convert);
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
    // Resolve value types by folding unrealized_conversion_casts that bridge
    // from !hir.any to MIR types. This handles casts from coerce_type (on
    // block args) and from reconcileOperandTypes (on opaque_unpack results
    // and other non-block-arg values).
    //
    // When retyping a value, we propagate the type through ops that have
    // SameOperandsAndResultType (like mir.add): if the result changes, the
    // operands must change too, and vice versa. This avoids breaking type
    // constraints when a cast on a binary op's result retypes it without
    // updating its operands.
    if (func.getBody().empty())
      continue;
    auto &entryBlock = func.getBody().front();

    // Propagate a concrete type through a value and its defining op chain.
    // When a value's type is changed from !hir.any to a concrete MIR type,
    // SameOperandsAndResultType ops need their operands retyped too.
    std::function<void(Value, Type)> propagateType = [&](Value value,
                                                         Type newType) {
      if (value.getType() == newType)
        return;
      value.setType(newType);
      if (auto opResult = dyn_cast<OpResult>(value)) {
        auto *defOp = opResult.getOwner();
        if (defOp->hasTrait<OpTrait::SameOperandsAndResultType>()) {
          for (auto operand : defOp->getOperands())
            propagateType(operand, newType);
        }
      }
    };

    for (auto &op : llvm::make_early_inc_range(entryBlock)) {
      auto castOp = dyn_cast<UnrealizedConversionCastOp>(&op);
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
      auto source = castOp.getOperand(0);
      castOp.getResult(0).replaceAllUsesWith(source);
      castOp.erase();
      propagateType(source, resultType);
    }

    // Check that all block arg types have been resolved. Skip private
    // functions, since const fragments from split_func legitimately have
    // !hir.any args whose types are resolved during specialization.
    if (!func.isPrivate()) {
      for (auto arg : entryBlock.getArguments()) {
        if (isa<hir::AnyType>(arg.getType())) {
          emitError(arg.getLoc())
              << "block argument type could not be determined "
              << "during HIR-to-MIR lowering; add hir.coerce_type";
          return signalPassFailure();
        }
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
