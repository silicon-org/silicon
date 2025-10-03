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
// UncheckedFuncOp
//===----------------------------------------------------------------------===//

LogicalResult UncheckedFuncOp::verifyRegions() {
  // Make sure there are no signature/return terminators in the middle of the
  // signature or body region.
  for (auto *region : getRegions())
    for (auto &block : llvm::drop_end(*region))
      if (isa<UncheckedSignatureOp, UncheckedReturnOp>(block.getTerminator()))
        return block.getTerminator()->emitOpError()
               << "can only appear in the last block";

  // Check the signature terminator.
  if (!isa<UncheckedSignatureOp>(getSignature().back().getTerminator()))
    return emitOpError() << "requires `hir.unchecked_signature` terminator in "
                            "the signature";

  // Check the body terminator.
  if (!isa<UncheckedReturnOp>(getBody().back().getTerminator()))
    return emitOpError() << "requires `hir.unchecked_return` terminator in "
                            "the body";

  return success();
}

UncheckedSignatureOp UncheckedFuncOp::getSignatureOp() {
  return cast<UncheckedSignatureOp>(getSignature().back().getTerminator());
}

UncheckedReturnOp UncheckedFuncOp::getReturnOp() {
  return cast<UncheckedReturnOp>(getBody().back().getTerminator());
}

//===----------------------------------------------------------------------===//
// UncheckedCallOp
//===----------------------------------------------------------------------===//

LogicalResult
UncheckedCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = getCalleeAttr();
  auto func = symbolTable.lookupNearestSymbolFrom<UncheckedFuncOp>(
      getOperation(), callee);
  if (!func)
    return emitOpError() << "callee " << callee
                         << " does not reference a valid `hir.unchecked_func`";

  auto sigOp = func.getSignatureOp();

  if (getArguments().size() != sigOp.getArgValues().size())
    return emitOpError() << "has " << getArguments().size()
                         << " arguments, but " << callee << " expects "
                         << sigOp.getArgValues().size();

  if (getResults().size() != sigOp.getTypeOfResults().size())
    return emitOpError() << "has " << getResults().size() << " results, but "
                         << callee << " expects "
                         << sigOp.getTypeOfResults().size();

  return success();
}
