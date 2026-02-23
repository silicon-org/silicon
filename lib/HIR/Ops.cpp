//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Attributes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Support/AsmParser.h"
#include "silicon/Support/MLIR.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

// Handle `custom<IntAttr>` parsing.
static ParseResult parseIntAttr(OpAsmParser &parser, IntAttr &value) {
  auto result = mlir::FieldParser<DynamicAPInt>::parse(parser);
  if (failed(result))
    return failure();
  value = IntAttr::get(parser.getContext(), *result);
  return success();
}

// Handle `custom<IntAttr>` printing.
static void printIntAttr(OpAsmPrinter &printer, Operation *op, IntAttr value) {
  printer << value.getValue();
}

SuccessorOperands ConstBranchOp::getSuccessorOperands(unsigned index) {
  assert(index == 0 && "invalid successor index");
  return SuccessorOperands(getDestOperandsMutable());
}

SuccessorOperands ConstCondBranchOp::getSuccessorOperands(unsigned index) {
  assert(index < getNumSuccessors() && "invalid successor index");
  return SuccessorOperands(index == 0 ? getTrueOperandsMutable()
                                      : getFalseOperandsMutable());
}

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/HIR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Infer result types for `hir.call`: one `!hir.any` result per element of
/// `typeOfResults`, so the parser can reconstruct the result count without
/// needing an explicit `type($results)` directive in the assembly format.
LogicalResult CallOp::inferReturnTypes(
    MLIRContext *ctx, std::optional<Location>, ValueRange operands,
    DictionaryAttr attrs, OpaqueProperties props, RegionRange,
    SmallVectorImpl<Type> &inferredReturnTypes) {
  CallOp::Adaptor adaptor(operands, attrs, props);
  auto anyType = AnyType::get(ctx);
  for (size_t i = 0; i < adaptor.getTypeOfResults().size(); ++i)
    inferredReturnTypes.push_back(anyType);
  return success();
}

//===----------------------------------------------------------------------===//
// TypeOfOp
//===----------------------------------------------------------------------===//

OpFoldResult TypeOfOp::fold(FoldAdaptor adaptor) {
  if (auto result = dyn_cast<OpResult>(getInput())) {
    // type_of(checked_call(..., resultTypes)) -> resultTypes[resultNumber]
    if (auto callOp = dyn_cast<CheckedCallOp>(result.getOwner()))
      return callOp.getTypeOfResults()[result.getResultNumber()];
  }
  return {};
}

LogicalResult TypeOfOp::canonicalize(TypeOfOp op, PatternRewriter &rewriter) {
  // type_of(constant_int) -> int_type
  if (op.getInput().getDefiningOp<ConstantIntOp>()) {
    rewriter.replaceOpWithNewOp<IntTypeOp>(op);
    return success();
  }
  // type_of(constant_unit) -> unit_type
  if (op.getInput().getDefiningOp<ConstantUnitOp>()) {
    rewriter.replaceOpWithNewOp<UnitTypeOp>(op);
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// UnifyOp
//===----------------------------------------------------------------------===//

OpFoldResult UnifyOp::fold(FoldAdaptor adaptor) {
  if (llvm::all_equal(getOperands()))
    return getOperands().front();
  return {};
}

//===----------------------------------------------------------------------===//
// UnifiedFuncOp
//===----------------------------------------------------------------------===//

LogicalResult UnifiedFuncOp::verify() {
  // Guard against a malformed signature region: verifyRegions() will report
  // the missing terminator, so we just skip phase-count validation here.
  auto sigOp =
      dyn_cast<UnifiedSignatureOp>(getSignature().back().getTerminator());
  if (!sigOp)
    return success();

  auto numArgs = sigOp.getTypeOfArgs().size();
  auto numResults = sigOp.getTypeOfResults().size();

  if (getArgPhases().size() != numArgs)
    return emitOpError() << "argPhases has " << getArgPhases().size()
                         << " entries but function has " << numArgs
                         << " arguments";

  if (getSignature().front().getNumArguments() != numArgs)
    return emitOpError() << "signature region has "
                         << getSignature().front().getNumArguments()
                         << " block arguments but function has " << numArgs
                         << " arguments";

  if (getResultPhases().size() != numResults)
    return emitOpError() << "resultPhases has " << getResultPhases().size()
                         << " entries but function has " << numResults
                         << " results";

  return success();
}

LogicalResult UnifiedFuncOp::verifyRegions() {
  // Make sure there are no signature/return terminators in the middle of the
  // signature or body region.
  for (auto *region : getRegions())
    for (auto &block : llvm::drop_end(*region))
      if (isa<UnifiedSignatureOp, UnifiedReturnOp>(block.getTerminator()))
        return block.getTerminator()->emitOpError()
               << "can only appear in the last block";

  // Check the signature terminator.
  if (!isa<UnifiedSignatureOp>(getSignature().back().getTerminator()))
    return emitOpError() << "requires `hir.unified_signature` terminator in "
                            "the signature";

  // Check the body terminator.
  if (!isa<UnifiedReturnOp>(getBody().back().getTerminator()))
    return emitOpError() << "requires `hir.unified_return` terminator in "
                            "the body";

  return success();
}

UnifiedSignatureOp UnifiedFuncOp::getSignatureOp() {
  return cast<UnifiedSignatureOp>(getSignature().back().getTerminator());
}

UnifiedReturnOp UnifiedFuncOp::getReturnOp() {
  return cast<UnifiedReturnOp>(getBody().back().getTerminator());
}

//===----------------------------------------------------------------------===//
// UnifiedCallOp
//===----------------------------------------------------------------------===//

LogicalResult
UnifiedCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = getCalleeAttr();
  auto func = symbolTable.lookupNearestSymbolFrom<UnifiedFuncOp>(
      getOperation(), callee);
  if (!func)
    return emitOpError() << "callee " << callee
                         << " does not reference a valid `hir.unified_func`";

  auto sigOp = func.getSignatureOp();

  if (getArguments().size() != sigOp.getTypeOfArgs().size())
    return emitOpError() << "has " << getArguments().size()
                         << " arguments, but " << callee << " expects "
                         << sigOp.getTypeOfArgs().size();

  if (getResults().size() != sigOp.getTypeOfResults().size())
    return emitOpError() << "has " << getResults().size() << " results, but "
                         << callee << " expects "
                         << sigOp.getTypeOfResults().size();

  if (getArgPhases().size() != getArguments().size())
    return emitOpError() << "argPhases has " << getArgPhases().size()
                         << " entries but call has " << getArguments().size()
                         << " arguments";

  if (getResultPhases().size() != getResults().size())
    return emitOpError() << "resultPhases has " << getResultPhases().size()
                         << " entries but call has " << getResults().size()
                         << " results";

  if (getArgPhases() != func.getArgPhases())
    return emitOpError() << "argPhases do not match callee " << callee;

  if (getResultPhases() != func.getResultPhases())
    return emitOpError() << "resultPhases do not match callee " << callee;

  return success();
}
