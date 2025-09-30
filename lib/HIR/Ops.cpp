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
