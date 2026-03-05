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
#include "llvm/ADT/DenseSet.h"
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

// CoerceTypeOp annotates a value with a type. After signature conversion,
// the input already has the correct concrete MIR type from the function
// boundary. Verify that the input type matches the type operand's constant
// value, then forward the input.
static LogicalResult convert(hir::CoerceTypeOp op,
                             hir::CoerceTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  mir::TypeAttr typeAttr;
  if (matchPattern(adaptor.getTypeOperand(), m_Constant(&typeAttr))) {
    auto expectedType = typeAttr.getValue();
    auto actualType = adaptor.getInput().getType();
    if (actualType != expectedType && !isa<hir::AnyType>(expectedType))
      return emitBug(op.getLoc())
             << "coerce_type type mismatch: input has " << actualType
             << " but type operand says " << expectedType;
  }
  rewriter.replaceOp(op, adaptor.getInput());
  return success();
}

// Arithmetic/bitwise HIR binary ops lower to the corresponding MIR ops by
// discarding the type operand and forwarding the converted lhs/rhs operands.
// After signature conversion, operands already have concrete MIR types.
// Comparison ops additionally pass the result type explicitly since operand
// and result types may differ in MIR (e.g. uint<1> for comparisons).

#define CONVERT_BINARY_OP(HirOp, MirOp)                                        \
  static LogicalResult convert(hir::HirOp op, hir::HirOp::Adaptor adaptor,     \
                               ConversionPatternRewriter &rewriter) {          \
    rewriter.replaceOpWithNewOp<mir::MirOp>(op, adaptor.getLhs(),              \
                                            adaptor.getRhs());                 \
    return success();                                                          \
  }

#define CONVERT_CMP_OP(HirOp, MirOp)                                           \
  static LogicalResult convert(hir::HirOp op, hir::HirOp::Adaptor adaptor,     \
                               ConversionPatternRewriter &rewriter) {          \
    rewriter.replaceOpWithNewOp<mir::MirOp>(                                   \
        op, adaptor.getLhs().getType(), adaptor.getLhs(), adaptor.getRhs());   \
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
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    mir::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant call result type";
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
// Lower hir.func ops to mir.func ops using MLIR dialect conversion with
// proper signature conversion. The FuncOp pattern determines argument types
// from coerce_type ops and result types from the hir.return's typeOfValues,
// then uses convertRegionTypes to retype block args. Body op patterns see
// the retyped values through their adaptors.
//
// Only functions whose type operands are all constants are lowered. Functions
// with unresolved types are skipped and will be lowered in a later pipeline
// iteration after specialization.
//===----------------------------------------------------------------------===//

/// Check whether a type value will resolve to a concrete MIR type after
/// conversion. Requires the defining op to be a known HIR type constructor
/// or an already-lowered mir.constant with a TypeAttr.
static bool isResolvableType(Value typeVal) {
  auto *defOp = typeVal.getDefiningOp();
  if (!defOp)
    return false; // block arg — unresolved cross-phase value

  // Simple type constructors always resolve.
  if (isa<hir::IntTypeOp, hir::UnitTypeOp, hir::TypeTypeOp, hir::AnyfuncTypeOp>(
          defOp))
    return true;

  // UIntTypeOp needs its width to be a constant integer.
  if (auto uintOp = dyn_cast<hir::UIntTypeOp>(defOp))
    return uintOp.getWidth().getDefiningOp<hir::ConstantIntOp>() != nullptr;

  // FuncTypeOp needs all of its type operands to be resolvable.
  if (auto funcTypeOp = dyn_cast<hir::FuncTypeOp>(defOp)) {
    for (auto arg : funcTypeOp.getTypeOfArgs())
      if (!isResolvableType(arg))
        return false;
    for (auto res : funcTypeOp.getTypeOfResults())
      if (!isResolvableType(res))
        return false;
    return true;
  }

  // Already-lowered mir.constant with a TypeAttr (from specialization).
  mir::TypeAttr typeAttr;
  if (matchPattern(typeVal, m_Constant(&typeAttr)))
    return !isa<hir::AnyType>(typeAttr.getValue());

  return false;
}

/// Check whether a function is ready for HIR-to-MIR lowering. A function is
/// ready when all type operands across all ops can be resolved to concrete
/// MIR types during conversion. We check every op that has type operands
/// used to determine output types: coerce_type, call, return, constant_func,
/// uint_type, func_type, and specialize_func.
///
/// Functions with unresolved types are skipped and will be lowered in a later
/// pipeline iteration after specialization resolves their types.
static bool shouldLower(hir::FuncOp func) {
  if (func.getBody().empty())
    return false;
  for (auto &op : func.getBody().front()) {
    if (auto coerce = dyn_cast<hir::CoerceTypeOp>(&op)) {
      if (!isResolvableType(coerce.getTypeOperand()))
        return false;
    } else if (auto call = dyn_cast<hir::CallOp>(&op)) {
      for (auto val : call.getTypeOfResults())
        if (!isResolvableType(val))
          return false;
    } else if (auto ret = dyn_cast<hir::ReturnOp>(&op)) {
      for (auto val : ret.getTypeOfValues())
        if (!isResolvableType(val))
          return false;
    } else if (auto constFunc = dyn_cast<hir::ConstantFuncOp>(&op)) {
      if (!isResolvableType(constFunc.getFuncType()))
        return false;
    } else if (auto uintType = dyn_cast<hir::UIntTypeOp>(&op)) {
      if (!uintType.getWidth().getDefiningOp<hir::ConstantIntOp>())
        return false;
    } else if (auto funcType = dyn_cast<hir::FuncTypeOp>(&op)) {
      for (auto val : funcType.getTypeOfArgs())
        if (!isResolvableType(val))
          return false;
      for (auto val : funcType.getTypeOfResults())
        if (!isResolvableType(val))
          return false;
    } else if (auto spec = dyn_cast<hir::SpecializeFuncOp>(&op)) {
      for (auto val : spec.getTypeOfArgs())
        if (!isResolvableType(val))
          return false;
      for (auto val : spec.getTypeOfResults())
        if (!isResolvableType(val))
          return false;
    }
  }
  return true;
}

/// Resolve the concrete MIR type from an unconverted HIR type value. Used
/// by the FuncOp pattern to determine block argument types from coerce_type
/// ops and result types from hir.return's typeOfValues.
static Type resolveHIRType(Value typeVal) {
  auto *defOp = typeVal.getDefiningOp();
  if (!defOp)
    return hir::AnyType::get(typeVal.getContext());

  auto *ctx = defOp->getContext();
  if (isa<hir::IntTypeOp>(defOp))
    return mir::IntType::get(ctx);
  if (isa<hir::UnitTypeOp>(defOp))
    return mir::UnitType::get(ctx);
  if (isa<hir::TypeTypeOp>(defOp))
    return mir::TypeType::get(ctx);
  if (isa<hir::AnyfuncTypeOp>(defOp))
    return mir::AnyfuncType::get(ctx);
  if (auto uintOp = dyn_cast<hir::UIntTypeOp>(defOp)) {
    if (auto widthOp = uintOp.getWidth().getDefiningOp<hir::ConstantIntOp>()) {
      auto width = static_cast<int64_t>(widthOp.getValue().getValue());
      return mir::UIntType::get(ctx, width);
    }
  }
  if (auto funcTypeOp = dyn_cast<hir::FuncTypeOp>(defOp)) {
    SmallVector<Type> argTypes, resultTypes;
    for (auto arg : funcTypeOp.getTypeOfArgs()) {
      auto t = resolveHIRType(arg);
      if (isa<hir::AnyType>(t))
        return hir::AnyType::get(ctx);
      argTypes.push_back(t);
    }
    for (auto res : funcTypeOp.getTypeOfResults()) {
      auto t = resolveHIRType(res);
      if (isa<hir::AnyType>(t))
        return hir::AnyType::get(ctx);
      resultTypes.push_back(t);
    }
    return FunctionType::get(ctx, argTypes, resultTypes);
  }

  // For mir.constant (from specialization), extract the type directly.
  mir::TypeAttr typeAttr;
  if (matchPattern(typeVal, m_Constant(&typeAttr)))
    return typeAttr.getValue();

  return hir::AnyType::get(typeVal.getContext());
}

//===----------------------------------------------------------------------===//
// FuncOp Conversion Pattern
//
// Lower hir.func → mir.func by scanning the body for type information:
// argument types from coerce_type ops, result types from hir.return's
// typeOfValues. Uses MLIR signature conversion to retype block args from
// !hir.any to concrete MIR types; the framework inserts materialization
// casts that body op patterns resolve through their adaptors.
//===----------------------------------------------------------------------===//

namespace {
class FuncOpConversion : public OpConversionPattern<hir::FuncOp> {
public:
  FuncOpConversion(TypeConverter &converter, MLIRContext *ctx,
                   const DenseSet<Operation *> *funcsToLower)
      : OpConversionPattern(converter, ctx), funcsToLower(funcsToLower) {}

  LogicalResult
  matchAndRewrite(hir::FuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!funcsToLower->contains(func))
      return failure();
    LLVM_DEBUG(llvm::dbgs() << "Lowering @" << func.getSymName() << "\n");

    auto &entryBlock = func.getBody().front();

    // Determine argument types from coerce_type ops on block args.
    SmallVector<Type> argTypes;
    for (auto arg : entryBlock.getArguments()) {
      Type argType;
      for (auto *user : arg.getUsers()) {
        if (auto coerce = dyn_cast<hir::CoerceTypeOp>(user)) {
          argType = resolveHIRType(coerce.getTypeOperand());
          break;
        }
      }
      if (!argType)
        return emitBug(func.getLoc())
               << "block argument type could not be determined during "
                  "HIR-to-MIR lowering; add hir.coerce_type";
      argTypes.push_back(argType);
    }

    // Determine result types from hir.return typeOfValues.
    SmallVector<Type> resultTypes;
    if (auto returnOp = dyn_cast<hir::ReturnOp>(entryBlock.getTerminator())) {
      for (auto typeVal : returnOp.getTypeOfValues())
        resultTypes.push_back(resolveHIRType(typeVal));
    }

    // Create the new mir.func with materialized types.
    auto funcType = FunctionType::get(func.getContext(), argTypes, resultTypes);
    auto mirFunc = mir::FuncOp::create(
        rewriter, func.getLoc(), func.getSymNameAttr(),
        func.getSymVisibilityAttr(), mlir::TypeAttr::get(funcType),
        func.getArgNamesAttr(), func.getResultNamesAttr());

    // Move the body and apply signature conversion to retype block args.
    rewriter.inlineRegionBefore(func.getBody(), mirFunc.getBody(),
                                mirFunc.getBody().end());
    TypeConverter::SignatureConversion sigConversion(
        entryBlock.getNumArguments());
    for (unsigned i = 0; i < argTypes.size(); ++i)
      sigConversion.addInputs(i, argTypes[i]);
    if (failed(rewriter.convertRegionTypes(
            &mirFunc.getBody(), *getTypeConverter(), &sigConversion)))
      return failure();

    rewriter.eraseOp(func);
    return success();
  }

private:
  const DenseSet<Operation *> *funcsToLower;
};
} // namespace

void HIRToMIRPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Pre-compute which functions are ready for lowering.
  DenseSet<Operation *> funcsToLower;
  for (auto func : moduleOp.getOps<hir::FuncOp>()) {
    if (shouldLower(func))
      funcsToLower.insert(func);
  }
  LLVM_DEBUG(llvm::dbgs() << "HIR-to-MIR: " << funcsToLower.size()
                          << " function(s) ready for lowering\n");
  if (funcsToLower.empty())
    return;

  // # Type Converter
  //
  // Only MIR types and FunctionType are legal. HIR types (including !hir.any)
  // must be converted. The value-based callback resolves
  // unrealized_conversion_cast values: when a value is defined by a cast from
  // a concrete MIR type to !hir.any, the adaptor maps it to the MIR-typed
  // input. This is how signature conversion materializations and pre-existing
  // casts get resolved.
  TypeConverter converter;
  converter.addConversion([](Type type) -> std::optional<Type> {
    if (isa<mir::MIRDialect>(type.getDialect()) || isa<FunctionType>(type))
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
  converter.addSourceMaterialization([](OpBuilder &builder, Type type,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });
  converter.addTargetMaterialization([](OpBuilder &builder, Type type,
                                        ValueRange inputs,
                                        Location loc) -> Value {
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  });

  // # Conversion Target
  //
  // Functions in the lowering set are illegal and converted to mir.func by
  // the FuncOp pattern. Functions not in the set are recursively legal,
  // so the framework skips them and their body ops entirely.
  ConversionTarget target(getContext());
  target.addLegalDialect<mir::MIRDialect>();
  target.addIllegalDialect<hir::HIRDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<hir::FuncOp>(
      [&](hir::FuncOp func) { return !funcsToLower.contains(func); });
  target.markOpRecursivelyLegal<hir::FuncOp>();
  target.addLegalOp<hir::SplitFuncOp>();
  target.markOpRecursivelyLegal<hir::SplitFuncOp>();

  // # Patterns
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add(std::make_unique<FuncOpConversion>(converter, &getContext(),
                                                  &funcsToLower));
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

  ConversionConfig config;
  config.allowPatternRollback = false;

  if (failed(
          applyFullConversion(moduleOp, target, std::move(patterns), config)))
    return signalPassFailure();
}
