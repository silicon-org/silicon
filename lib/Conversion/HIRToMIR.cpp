//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/Base/Dialect.h"
#include "silicon/Base/Types.h"
#include "silicon/Conversion/Passes.h" // IWYU pragma: keep
#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Types.h"
#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "circt/Support/ConversionPatternSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h" // IWYU pragma: keep
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
  auto attr = base::IntAttr::get(op.getContext(), op.getValue().getValue());
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::ConstantUnitOp op,
                             hir::ConstantUnitOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto attr = base::UnitAttr::get(op.getContext());
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Constructors
//===----------------------------------------------------------------------===//

static LogicalResult convert(hir::IntTypeOp op, hir::IntTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = base::IntType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::UnitTypeOp op,
                             hir::UnitTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = base::UnitType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::TypeTypeOp op,
                             hir::TypeTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = base::TypeType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::UIntTypeOp op,
                             hir::UIntTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto widthOperand = adaptor.getWidth();
  base::IntAttr widthAttr;
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
  auto type = base::UIntType::get(op.getContext(), static_cast<int64_t>(width));
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::AnyfuncTypeOp op,
                             hir::AnyfuncTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = base::AnyfuncType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::OpaqueTypeOp op,
                             hir::OpaqueTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  auto type = base::OpaqueType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
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
    base::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant argument type";
    argTypes.push_back(attr.getValue());
  }

  // Gather the result types.
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    base::TypeAttr attr;
    if (!matchPattern(value, m_Constant(&attr)))
      return emitBug(op.getLoc()) << "non-constant result type";
    resultTypes.push_back(attr.getValue());
  }

  // Materialize the constant function type.
  auto type = FunctionType::get(op.getContext(), argTypes, resultTypes);
  auto attr = base::TypeAttr::get(op.getContext(), type);
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
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::TypeOfOp op, hir::TypeOfOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // TypeOfOp results are consumed as type metadata by BinaryOp and ReturnOp,
  // which discard them during lowering. Replace with a dummy type constant.
  auto type = hir::AnyType::get(op.getContext());
  auto attr = base::TypeAttr::get(op.getContext(), type);
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, attr);
  return success();
}

static LogicalResult convert(hir::UnifyOp op, hir::UnifyOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  // UnifyOp constrains two type values to be equal. InferTypes and
  // canonicalization should eliminate most unify ops before HIR-to-MIR
  // lowering. Surviving unify ops are handled here by comparing the
  // converted operands: if they are the same SSA value or have the same
  // constant value, we forward one of them. Otherwise, this indicates a
  // type mismatch that earlier passes should have caught.
  if (adaptor.getLhs() == adaptor.getRhs()) {
    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }

  // Check if both operands are constants with the same value, which can
  // happen when two separate type constructor ops produce the same type.
  Attribute lhsAttr, rhsAttr;
  if (matchPattern(adaptor.getLhs(), m_Constant(&lhsAttr)) &&
      matchPattern(adaptor.getRhs(), m_Constant(&rhsAttr)) &&
      lhsAttr == rhsAttr) {
    rewriter.replaceOp(op, adaptor.getLhs());
    return success();
  }

  return emitBug(op.getLoc())
         << "hir.unify survived to HIR-to-MIR lowering with "
            "different operands; InferTypes should have eliminated it";
}

// CoerceTypeOp annotates a value with a type. After signature conversion,
// the input already has the correct concrete MIR type from the function
// boundary. Verify that the input type matches the type operand's constant
// value, then forward the input.
static LogicalResult convert(hir::CoerceTypeOp op,
                             hir::CoerceTypeOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  base::TypeAttr typeAttr;
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

template <typename HirOpT, typename MirOpT>
static LogicalResult convertBinaryOp(HirOpT op,
                                     typename HirOpT::Adaptor adaptor,
                                     ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<MirOpT>(op, adaptor.getLhs(), adaptor.getRhs());
  return success();
}

template <typename HirOpT, typename MirOpT>
static LogicalResult convertCmpOp(HirOpT op, typename HirOpT::Adaptor adaptor,
                                  ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<MirOpT>(op, adaptor.getLhs().getType(),
                                      adaptor.getLhs(), adaptor.getRhs());
  return success();
}

static LogicalResult convert(hir::MIRConstantOp op,
                             hir::MIRConstantOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::ConstantOp>(op, op.getValue());
  return success();
}

static LogicalResult convert(hir::ReturnOp op, hir::ReturnOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::ReturnOp>(op, adaptor.getValues());
  return success();
}

/// Lower hir.yield to mir.yield, forwarding operands.
static LogicalResult convert(hir::YieldOp op, hir::YieldOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  rewriter.replaceOpWithNewOp<mir::YieldOp>(op, adaptor.getOperands());
  return success();
}

static LogicalResult convert(hir::CallOp op, hir::CallOp::Adaptor adaptor,
                             ConversionPatternRewriter &rewriter) {
  SmallVector<Type> resultTypes;
  resultTypes.reserve(adaptor.getTypeOfResults().size());
  for (auto value : adaptor.getTypeOfResults()) {
    base::TypeAttr attr;
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
  auto opaqueType = base::OpaqueType::get(op.getContext());
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
      if (isa<base::BaseDialect, mir::MIRDialect>(input.getDialect()) ||
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
  if (isa<hir::IntTypeOp, hir::UnitTypeOp, hir::TypeTypeOp, hir::AnyfuncTypeOp,
          hir::OpaqueTypeOp>(defOp))
    return true;

  // UIntTypeOp needs its width to be a constant integer.
  if (auto uintOp = dyn_cast<hir::UIntTypeOp>(defOp))
    return uintOp.getWidth().getDefiningOp<hir::ConstantIntOp>() != nullptr;

  // FuncTypeOp needs all of its type operands to be resolvable.
  if (auto funcTypeOp = dyn_cast<hir::FuncTypeOp>(defOp)) {
    return llvm::all_of(funcTypeOp.getTypeOfArgs(), isResolvableType) &&
           llvm::all_of(funcTypeOp.getTypeOfResults(), isResolvableType);
  }

  // UnifyOp is resolvable if both operands are resolvable.
  if (auto unifyOp = dyn_cast<hir::UnifyOp>(defOp))
    return isResolvableType(unifyOp.getLhs()) &&
           isResolvableType(unifyOp.getRhs());

  // CoerceTypeOp forwards its input; resolvable if the input is.
  if (auto coerceOp = dyn_cast<hir::CoerceTypeOp>(defOp))
    return isResolvableType(coerceOp.getInput());

  // ConstantLike op with a TypeAttr (e.g., mir.constant or hir.mir_constant
  // from specialization).
  base::TypeAttr typeAttr;
  if (matchPattern(typeVal, m_Constant(&typeAttr)))
    return !isa<hir::AnyType>(typeAttr.getValue());

  return false;
}

/// Check whether a function is ready for HIR-to-MIR lowering. A function is
/// ready when all type operands across all ops can be resolved to concrete
/// MIR types during conversion. We check every op that has type operands
/// used to determine output types: coerce_type, call, return, uint_type,
/// func_type, and specialize_func.
///
/// Functions with unresolved types are skipped and will be lowered in a later
/// pipeline iteration after specialization resolves their types.
static bool shouldLower(hir::FuncOp func) {
  if (func.getBody().empty())
    return false;
  bool ready = true;
  func.walk([&](Operation *op) {
    if (!ready)
      return;
    if (auto coerce = dyn_cast<hir::CoerceTypeOp>(op)) {
      if (!isResolvableType(coerce.getTypeOperand()))
        ready = false;
    } else if (auto call = dyn_cast<hir::CallOp>(op)) {
      for (auto val : call.getTypeOfArgs())
        if (!isResolvableType(val))
          ready = false;
      for (auto val : call.getTypeOfResults())
        if (!isResolvableType(val))
          ready = false;
    } else if (auto ret = dyn_cast<hir::ReturnOp>(op)) {
      for (auto val : ret.getTypeOfValues())
        if (!isResolvableType(val))
          ready = false;
    } else if (auto uintType = dyn_cast<hir::UIntTypeOp>(op)) {
      if (!uintType.getWidth().getDefiningOp<hir::ConstantIntOp>())
        ready = false;
    } else if (auto funcType = dyn_cast<hir::FuncTypeOp>(op)) {
      for (auto val : funcType.getTypeOfArgs())
        if (!isResolvableType(val))
          ready = false;
      for (auto val : funcType.getTypeOfResults())
        if (!isResolvableType(val))
          ready = false;
    } else if (isa<hir::OpaqueUnpackOp>(op)) {
      // Opaque unpack ops indicate that the function has not yet been
      // specialized with the results of a previous phase's evaluation.
      // Defer lowering until specialization replaces the unpack with
      // concrete constant values.
      ready = false;
    }
  });
  return ready;
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
    return base::IntType::get(ctx);
  if (isa<hir::UnitTypeOp>(defOp))
    return base::UnitType::get(ctx);
  if (isa<hir::TypeTypeOp>(defOp))
    return base::TypeType::get(ctx);
  if (isa<hir::AnyfuncTypeOp>(defOp))
    return base::AnyfuncType::get(ctx);
  if (isa<hir::OpaqueTypeOp>(defOp))
    return base::OpaqueType::get(ctx);
  if (auto uintOp = dyn_cast<hir::UIntTypeOp>(defOp)) {
    if (auto widthOp = uintOp.getWidth().getDefiningOp<hir::ConstantIntOp>()) {
      auto width = static_cast<int64_t>(widthOp.getValue().getValue());
      return base::UIntType::get(ctx, width);
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

  // UnifyOp resolves to the lhs type (both sides should be the same after
  // type inference; the convert pattern for UnifyOp will catch mismatches).
  if (auto unifyOp = dyn_cast<hir::UnifyOp>(defOp))
    return resolveHIRType(unifyOp.getLhs());

  // CoerceTypeOp forwards its input value, so resolve the input's type.
  if (auto coerceOp = dyn_cast<hir::CoerceTypeOp>(defOp))
    return resolveHIRType(coerceOp.getInput());

  // ConstantLike op with a TypeAttr (e.g., mir.constant or hir.mir_constant
  // from specialization).
  base::TypeAttr typeAttr;
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

/// Lower hir.if to mir.if, converting both regions in place. Result types are
/// resolved from the HIR type information available on the yield values
/// (via getTypeOf), falling back to !hir.any for unknown types.
namespace {
class IfOpConversion : public OpConversionPattern<hir::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hir::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Determine result types from the type information on yield values.
    SmallVector<Type> resultTypes;
    for (auto result : op.getResults()) {
      auto typeVal = hir::getTypeOf(result);
      if (typeVal)
        resultTypes.push_back(resolveHIRType(typeVal));
      else
        resultTypes.push_back(adaptor.getCondition().getType());
    }

    auto mirIf = mir::IfOp::create(rewriter, op.getLoc(), resultTypes,
                                   adaptor.getCondition());

    // Move then region.
    rewriter.inlineRegionBefore(op.getThenRegion(), mirIf.getThenRegion(),
                                mirIf.getThenRegion().end());

    // Move else region.
    rewriter.inlineRegionBefore(op.getElseRegion(), mirIf.getElseRegion(),
                                mirIf.getElseRegion().end());

    rewriter.replaceOp(op, mirIf.getResults());
    return success();
  }
};
} // namespace

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
    // CoerceTypeOp is not Pure, so it survives DCE even for unused args.
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
        func.getArgNamesAttr(), func.getResultNamesAttr(),
        func.getIsModuleAttr());

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
    if (isa<base::BaseDialect, mir::MIRDialect>(type.getDialect()) ||
        isa<FunctionType>(type))
      return type;
    return std::nullopt;
  });
  converter.addConversion([](Value value) -> std::optional<Type> {
    if (auto castOp = value.getDefiningOp<UnrealizedConversionCastOp>()) {
      if (castOp.getNumOperands() == 1 && castOp.getNumResults() == 1) {
        auto type = castOp.getOperand(0).getType();
        if (isa<base::BaseDialect, mir::MIRDialect>(type.getDialect()) ||
            isa<FunctionType>(type))
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
  target.addLegalDialect<base::BaseDialect, mir::MIRDialect>();
  target.addIllegalDialect<hir::HIRDialect>();
  target.addLegalOp<ModuleOp>();
  target.addDynamicallyLegalOp<hir::FuncOp>(
      [&](hir::FuncOp func) { return !funcsToLower.contains(func); });
  target.markOpRecursivelyLegal<hir::FuncOp>();
  target.addLegalOp<hir::SplitFuncOp>();
  target.markOpRecursivelyLegal<hir::SplitFuncOp>();
  target.addLegalOp<hir::MultiphaseFuncOp>();

  // # Patterns
  ConversionPatternSet patterns(&getContext(), converter);
  patterns.add(std::make_unique<FuncOpConversion>(converter, &getContext(),
                                                  &funcsToLower));
  patterns.add<hir::ConstantIntOp>(convert);
  patterns.add<hir::ConstantUnitOp>(convert);
  patterns.add<hir::IntTypeOp>(convert);
  patterns.add<hir::UnitTypeOp>(convert);
  patterns.add<hir::TypeTypeOp>(convert);
  patterns.add<hir::UIntTypeOp>(convert);
  patterns.add<hir::AnyfuncTypeOp>(convert);
  patterns.add<hir::OpaqueTypeOp>(convert);
  patterns.add<hir::FuncTypeOp>(convert);
  patterns.add<hir::InferrableOp>(convert);
  patterns.add<hir::TypeOfOp>(convert);
  patterns.add<hir::UnifyOp>(convert);
  patterns.add<hir::CoerceTypeOp>(convert);
  patterns.add<hir::AddOp>(convertBinaryOp<hir::AddOp, mir::AddOp>);
  patterns.add<hir::SubOp>(convertBinaryOp<hir::SubOp, mir::SubOp>);
  patterns.add<hir::MulOp>(convertBinaryOp<hir::MulOp, mir::MulOp>);
  patterns.add<hir::DivOp>(convertBinaryOp<hir::DivOp, mir::DivOp>);
  patterns.add<hir::ModOp>(convertBinaryOp<hir::ModOp, mir::ModOp>);
  patterns.add<hir::AndOp>(convertBinaryOp<hir::AndOp, mir::AndOp>);
  patterns.add<hir::OrOp>(convertBinaryOp<hir::OrOp, mir::OrOp>);
  patterns.add<hir::XorOp>(convertBinaryOp<hir::XorOp, mir::XorOp>);
  patterns.add<hir::ShlOp>(convertBinaryOp<hir::ShlOp, mir::ShlOp>);
  patterns.add<hir::ShrOp>(convertBinaryOp<hir::ShrOp, mir::ShrOp>);
  patterns.add<hir::EqOp>(convertCmpOp<hir::EqOp, mir::EqOp>);
  patterns.add<hir::NeqOp>(convertCmpOp<hir::NeqOp, mir::NeqOp>);
  patterns.add<hir::LtOp>(convertCmpOp<hir::LtOp, mir::LtOp>);
  patterns.add<hir::GtOp>(convertCmpOp<hir::GtOp, mir::GtOp>);
  patterns.add<hir::GeqOp>(convertCmpOp<hir::GeqOp, mir::GeqOp>);
  patterns.add<hir::LeqOp>(convertCmpOp<hir::LeqOp, mir::LeqOp>);
  patterns.add<hir::MIRConstantOp>(convert);
  patterns.add<hir::ReturnOp>(convert);
  patterns.add<hir::YieldOp>(convert);
  patterns.add(std::make_unique<IfOpConversion>(converter, &getContext()));
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
