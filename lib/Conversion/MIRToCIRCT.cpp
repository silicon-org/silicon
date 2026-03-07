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
#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "mir-to-circt"

namespace silicon {
#define GEN_PASS_DEF_MIRTOCIRCTPASS
#include "silicon/Conversion/Passes.h.inc"
} // namespace silicon

namespace {
struct MIRToCIRCTPass
    : public silicon::impl::MIRToCIRCTPassBase<MIRToCIRCTPass> {
  void runOnOperation() override;
};
} // namespace

//===----------------------------------------------------------------------===//
// Type Conversion
//
// Convert Silicon types to CIRCT-compatible integer types:
// - `!si.int` → `i64` (temporary; TODO: error once bitwidth inference exists)
// - `!si.uint<N>` → `iN`
// - `FunctionType` → recursively convert arg/result types
// - Standard `IntegerType` → pass through
//===----------------------------------------------------------------------===//

/// Convert a Silicon type to a CIRCT-compatible type.
static std::optional<Type> convertType(Type type) {
  // Standard integer types pass through.
  if (isa<IntegerType>(type))
    return type;

  // `!si.int` → `i64`.
  if (isa<base::IntType>(type))
    return IntegerType::get(type.getContext(), 64);

  // `!si.uint<N>` → `iN`.
  if (auto uintType = dyn_cast<base::UIntType>(type))
    return IntegerType::get(type.getContext(), uintType.getWidth());

  // Recursively convert function types.
  if (auto funcType = dyn_cast<FunctionType>(type)) {
    SmallVector<Type> inputs, results;
    for (auto t : funcType.getInputs()) {
      auto converted = convertType(t);
      if (!converted)
        return std::nullopt;
      inputs.push_back(*converted);
    }
    for (auto t : funcType.getResults()) {
      auto converted = convertType(t);
      if (!converted)
        return std::nullopt;
      results.push_back(*converted);
    }
    return FunctionType::get(type.getContext(), inputs, results);
  }

  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// Convert mir.func → hw.module. Builds PortInfo from argNames/resultNames
/// and converted types.
class FuncOpConversion : public OpConversionPattern<mir::FuncOp> {
public:
  FuncOpConversion(TypeConverter &converter, MLIRContext *ctx,
                   const DenseSet<Operation *> *funcsToLower)
      : OpConversionPattern(converter, ctx), funcsToLower(funcsToLower) {}

  LogicalResult
  matchAndRewrite(mir::FuncOp func, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!funcsToLower->contains(func))
      return failure();

    auto funcType = func.getFunctionType();
    auto loc = func.getLoc();

    // Build port list from function signature.
    SmallVector<circt::hw::PortInfo> ports;

    // Input ports from arguments.
    auto argNames = func.getArgNames();
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i) {
      auto convertedType = convertType(funcType.getInput(i));
      if (!convertedType)
        return emitBug(loc)
               << "unsupported argument type " << funcType.getInput(i);

      circt::hw::PortInfo port;
      port.name = cast<StringAttr>(argNames[i]);
      port.type = *convertedType;
      port.dir = circt::hw::ModulePort::Direction::Input;
      port.argNum = i;
      port.loc = loc;
      ports.push_back(port);
    }

    // Output ports from results.
    auto resultNames = func.getResultNames();
    for (unsigned i = 0; i < funcType.getNumResults(); ++i) {
      auto convertedType = convertType(funcType.getResult(i));
      if (!convertedType)
        return emitBug(loc)
               << "unsupported result type " << funcType.getResult(i);

      circt::hw::PortInfo port;
      port.name = cast<StringAttr>(resultNames[i]);
      port.type = *convertedType;
      port.dir = circt::hw::ModulePort::Direction::Output;
      port.argNum = i;
      port.loc = loc;
      ports.push_back(port);
    }

    // Create hw.module with shouldEnsureTerminator=false so we can move
    // the body in and let the conversion framework handle inner ops.
    circt::hw::ModulePortInfo portInfo(ports);
    auto hwModule = circt::hw::HWModuleOp::create(
        rewriter, loc, func.getSymNameAttr(), portInfo,
        /*parameters=*/ArrayAttr{}, /*attributes=*/{}, /*comment=*/StringAttr{},
        /*shouldEnsureTerminator=*/false);

    // Move the body into the new hw.module. The builder may have created
    // an empty block, so erase it first.
    if (!hwModule.getBody().empty())
      rewriter.eraseBlock(&hwModule.getBody().front());
    rewriter.inlineRegionBefore(func.getBody(), hwModule.getBody(),
                                hwModule.getBody().end());
    auto &entry = hwModule.getBody().front();

    // Retype block arguments.
    for (unsigned i = 0; i < funcType.getNumInputs(); ++i)
      entry.getArgument(i).setType(ports[i].type);

    rewriter.eraseOp(func);
    return success();
  }

private:
  const DenseSet<Operation *> *funcsToLower;
};

/// Convert mir.return → hw.output.
class ReturnOpConversion : public OpConversionPattern<mir::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<circt::hw::OutputOp>(op, adaptor.getOperands());
    return success();
  }
};

/// Convert mir.constant → hw.constant. Extracts the IntAttr value and
/// produces an APInt of the target bit width.
class ConstantOpConversion : public OpConversionPattern<mir::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto intAttr = dyn_cast<base::IntAttr>(op.getValue());
    if (!intAttr)
      return emitBug(op.getLoc())
             << "mir.constant with non-integer attribute in module context";

    auto resultType = convertType(op.getType());
    if (!resultType)
      return emitBug(op.getLoc())
             << "unsupported constant type " << op.getType();

    auto intType = cast<IntegerType>(*resultType);
    unsigned width = intType.getWidth();

    // Convert DynamicAPInt → APInt of the target width.
    const auto &dynVal = intAttr.getValue();
    APInt apVal(width, static_cast<int64_t>(dynVal), /*isSigned=*/true);

    rewriter.replaceOpWithNewOp<circt::hw::ConstantOp>(op, apVal);
    return success();
  }
};

/// Convert mir.call → hw.instance.
class CallOpConversion : public OpConversionPattern<mir::CallOp> {
public:
  CallOpConversion(TypeConverter &converter, MLIRContext *ctx,
                   DenseMap<StringRef, unsigned> *instanceCounters)
      : OpConversionPattern(converter, ctx),
        instanceCounters(instanceCounters) {}

  LogicalResult
  matchAndRewrite(mir::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto callee = op.getCallee();

    // Look up the target module. It may have been converted already or may
    // be pending conversion.
    auto parentModule = op->getParentOfType<ModuleOp>();
    auto *target = SymbolTable::lookupSymbolIn(parentModule, callee);
    if (!target)
      return emitBug(op.getLoc()) << "callee @" << callee << " not found";

    // If the target hasn't been converted to hw.module yet (it's still
    // mir.func), we need to look it up as such.
    auto hwTarget = dyn_cast<circt::hw::HWModuleOp>(target);
    auto mirTarget = dyn_cast<mir::FuncOp>(target);
    if (!hwTarget && !mirTarget)
      return emitBug(op.getLoc())
             << "callee @" << callee << " is neither hw.module nor mir.func";

    // Generate a unique instance name.
    unsigned &counter = (*instanceCounters)[callee];
    std::string instanceName = (callee + "_inst" + Twine(counter++)).str();

    // Convert result types.
    SmallVector<Type> resultTypes;
    for (auto t : op.getResultTypes()) {
      auto converted = convertType(t);
      if (!converted)
        return emitBug(op.getLoc()) << "unsupported call result type " << t;
      resultTypes.push_back(*converted);
    }

    // Build input names and result names from the target.
    SmallVector<Attribute> argNameAttrs, resultNameAttrs;
    if (hwTarget) {
      auto modType = hwTarget.getModuleType();
      for (auto port : modType.getPorts()) {
        if (port.dir == circt::hw::ModulePort::Direction::Input)
          argNameAttrs.push_back(port.name);
        else if (port.dir == circt::hw::ModulePort::Direction::Output)
          resultNameAttrs.push_back(port.name);
      }
    } else {
      argNameAttrs.assign(mirTarget.getArgNames().begin(),
                          mirTarget.getArgNames().end());
      resultNameAttrs.assign(mirTarget.getResultNames().begin(),
                             mirTarget.getResultNames().end());
    }

    auto argNamesAttr = rewriter.getArrayAttr(argNameAttrs);
    auto resultNamesAttr = rewriter.getArrayAttr(resultNameAttrs);

    rewriter.replaceOpWithNewOp<circt::hw::InstanceOp>(
        op, resultTypes, rewriter.getStringAttr(instanceName),
        rewriter.getAttr<FlatSymbolRefAttr>(callee), adaptor.getArguments(),
        argNamesAttr, resultNamesAttr,
        /*parameters=*/rewriter.getArrayAttr({}),
        /*inner_sym=*/circt::hw::InnerSymAttr{});
    return success();
  }

private:
  DenseMap<StringRef, unsigned> *instanceCounters;
};

/// Convert a binary MIR arithmetic op to the corresponding comb op.
/// For variadic comb ops (add, mul, and, or, xor), pass {lhs, rhs}.
template <typename MIROp, typename CombOp>
class BinaryOpConversion : public OpConversionPattern<MIROp> {
public:
  using OpConversionPattern<MIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MIROp op, typename MIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultType = convertType(op.getType());
    if (!resultType)
      return emitBug(op.getLoc())
             << "unsupported type " << op.getType() << " in binary op";

    rewriter.replaceOpWithNewOp<CombOp>(op, adaptor.getLhs(), adaptor.getRhs());
    return success();
  }
};

/// Convert a MIR comparison op to comb.icmp with the given predicate.
template <typename MIROp, circt::comb::ICmpPredicate pred>
class CmpOpConversion : public OpConversionPattern<MIROp> {
public:
  using OpConversionPattern<MIROp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(MIROp op, typename MIROp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<circt::comb::ICmpOp>(op, pred, adaptor.getLhs(),
                                                     adaptor.getRhs());
    return success();
  }
};

/// Convert mir.if/mir.yield → comb.mux for single-result cases.
class IfOpConversion : public OpConversionPattern<mir::IfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mir::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Only support single-result if expressions.
    if (op.getNumResults() != 1)
      return op.emitError("mir.if with multiple results cannot be lowered to "
                          "comb.mux; only single-result is supported");

    // Inline the bodies before the mux. Clone ops from both regions
    // (excluding yield terminators) into the parent block, then create
    // the mux from the yielded values.
    IRMapping mapping;

    // Clone then region ops (excluding yield).
    auto &thenBlock = op.getThenRegion().front();
    for (auto &bodyOp :
         llvm::make_range(thenBlock.begin(), std::prev(thenBlock.end())))
      rewriter.clone(bodyOp, mapping);
    auto thenYield = cast<mir::YieldOp>(thenBlock.getTerminator());
    Value trueVal = mapping.lookupOrDefault(thenYield.getOperands()[0]);

    // Clone else region ops (excluding yield).
    auto &elseBlock = op.getElseRegion().front();
    for (auto &bodyOp :
         llvm::make_range(elseBlock.begin(), std::prev(elseBlock.end())))
      rewriter.clone(bodyOp, mapping);
    auto elseYield = cast<mir::YieldOp>(elseBlock.getTerminator());
    Value falseVal = mapping.lookupOrDefault(elseYield.getOperands()[0]);

    rewriter.replaceOpWithNewOp<circt::comb::MuxOp>(op, adaptor.getCondition(),
                                                    trueVal, falseVal);
    return success();
  }
};

/// Erase any remaining Silicon dialect op that has no uses.
class EraseUnusedSiliconOp : public ConversionPattern {
public:
  EraseUnusedSiliconOp(MLIRContext *ctx)
      : ConversionPattern(MatchAnyOpTypeTag(), /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Only match Silicon dialect ops.
    auto *dialect = op->getDialect();
    if (!isa<base::BaseDialect, hir::HIRDialect, mir::MIRDialect>(dialect))
      return failure();

    // Only erase if the op has no uses.
    if (!op->use_empty())
      return failure();

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// MIRToCIRCT Pass
//
// Find all mir.func ops marked isModule, collect their transitive call graph,
// convert them to hw.module with comb logic, and erase all remaining Silicon
// dialect ops.
//===----------------------------------------------------------------------===//

void MIRToCIRCTPass::runOnOperation() {
  auto moduleOp = getOperation();

  // Collect all surviving mir.func ops whose types can be converted.
  // Functions with unconvertible types (e.g., opaque) are leftovers from
  // phase evaluation and will be erased by the catch-all pattern.
  DenseSet<Operation *> funcsToLower;
  for (auto func : moduleOp.getOps<mir::FuncOp>()) {
    auto funcType = func.getFunctionType();
    bool convertible = true;
    for (auto t : funcType.getInputs())
      if (!convertType(t)) {
        convertible = false;
        break;
      }
    if (convertible)
      for (auto t : funcType.getResults())
        if (!convertType(t)) {
          convertible = false;
          break;
        }
    if (convertible)
      funcsToLower.insert(func);
  }
  LLVM_DEBUG(llvm::dbgs() << "MIR-to-CIRCT: " << funcsToLower.size()
                          << " function(s) to lower\n");

  // # Type Converter
  //
  // Silicon types become CIRCT integer types; standard types pass through.
  TypeConverter converter;
  converter.addConversion(
      [](Type type) -> std::optional<Type> { return convertType(type); });

  // # Conversion Target
  //
  // hw/comb and ModuleOp are legal. All Silicon dialects are illegal.
  ConversionTarget target(getContext());
  target.addLegalDialect<circt::hw::HWDialect, circt::comb::CombDialect>();
  target.addLegalOp<ModuleOp>();
  target
      .addIllegalDialect<base::BaseDialect, hir::HIRDialect, mir::MIRDialect>();

  // mir.func ops not in the call graph should be erased by the catch-all.
  // mir.func ops in the call graph are converted by FuncOpConversion.

  // # Patterns
  DenseMap<StringRef, unsigned> instanceCounters;
  RewritePatternSet patterns(&getContext());
  patterns.add<FuncOpConversion>(converter, &getContext(), &funcsToLower);
  patterns.add<ReturnOpConversion>(converter, &getContext());
  patterns.add<ConstantOpConversion>(converter, &getContext());
  patterns.add<CallOpConversion>(converter, &getContext(), &instanceCounters);
  patterns.add<BinaryOpConversion<mir::AddOp, circt::comb::AddOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::SubOp, circt::comb::SubOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::MulOp, circt::comb::MulOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::DivOp, circt::comb::DivSOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::ModOp, circt::comb::ModSOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::AndOp, circt::comb::AndOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::OrOp, circt::comb::OrOp>>(converter,
                                                                 &getContext());
  patterns.add<BinaryOpConversion<mir::XorOp, circt::comb::XorOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::ShlOp, circt::comb::ShlOp>>(
      converter, &getContext());
  patterns.add<BinaryOpConversion<mir::ShrOp, circt::comb::ShrUOp>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::EqOp, circt::comb::ICmpPredicate::eq>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::NeqOp, circt::comb::ICmpPredicate::ne>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::LtOp, circt::comb::ICmpPredicate::slt>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::GtOp, circt::comb::ICmpPredicate::sgt>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::LeqOp, circt::comb::ICmpPredicate::sle>>(
      converter, &getContext());
  patterns.add<CmpOpConversion<mir::GeqOp, circt::comb::ICmpPredicate::sge>>(
      converter, &getContext());
  patterns.add<IfOpConversion>(converter, &getContext());
  patterns.add<EraseUnusedSiliconOp>(&getContext());

  if (failed(applyFullConversion(moduleOp, target, std::move(patterns))))
    return signalPassFailure();
}
