//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Types.h"
#include "silicon/UIR/Ops.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace silicon::uir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Walk parent ops to find a transitive ancestor with the given op name.
static bool hasAncestorOp(Operation *op, StringRef name) {
  for (auto *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() == name)
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult BreakOp::verify() {
  if (!hasAncestorOp(*this, "uir.loop"))
    return emitOpError("must be nested inside a 'uir.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// ContinueOp
//===----------------------------------------------------------------------===//

LogicalResult ContinueOp::verify() {
  if (!hasAncestorOp(*this, "uir.loop"))
    return emitOpError("must be nested inside a 'uir.loop'");
  return success();
}

//===----------------------------------------------------------------------===//
// IfOp
//===----------------------------------------------------------------------===//

// Parse: uir.if %cond : %ty1, %ty2 { ... } else { ... }
//        uir.if %cond { ... }
ParseResult IfOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse condition.
  OpAsmParser::UnresolvedOperand condOperand;
  if (parser.parseOperand(condOperand) ||
      parser.resolveOperand(condOperand, anyTy, result.operands))
    return failure();

  // Parse optional result type operands: `: %ty1, %ty2`.
  SmallVector<OpAsmParser::UnresolvedOperand> resultTypeOperands;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseOperandList(resultTypeOperands))
      return failure();
    if (parser.resolveOperands(resultTypeOperands, anyTy, result.operands))
      return failure();
  }

  // Add result types (one !hir.any per result type operand).
  result.addTypes(SmallVector<Type>(resultTypeOperands.size(), anyTy));

  // Parse then region.
  auto *thenRegion = result.addRegion();
  if (parser.parseRegion(*thenRegion))
    return failure();

  // Parse optional else region.
  auto *elseRegion = result.addRegion();
  if (succeeded(parser.parseOptionalKeyword("else"))) {
    if (parser.parseRegion(*elseRegion))
      return failure();
  }

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void IfOp::print(OpAsmPrinter &p) {
  p << ' ' << getCondition();

  // Print result type operands if any.
  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleaveComma(getResultTypes(), p,
                          [&](Value v) { p.printOperand(v); });
  }

  // Print then region.
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

  // Print else region if non-empty.
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
  }

  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult IfOp::verify() {
  // Then region must have exactly one block.
  if (!getThenRegion().hasOneBlock())
    return emitOpError("'then' region must have exactly one block");

  // Else region must have 0 or 1 blocks.
  if (!getElseRegion().empty() && !getElseRegion().hasOneBlock())
    return emitOpError("'else' region must have exactly one block");

  // If no else region, no results allowed.
  if (getElseRegion().empty() && getNumResults() > 0)
    return emitOpError("if without 'else' cannot produce results");

  // Result count must match result type operand count.
  if (getNumResults() != getResultTypes().size())
    return emitOpError("result count must match result type operand count");

  return success();
}

//===----------------------------------------------------------------------===//
// LoopOp
//===----------------------------------------------------------------------===//

// Parse: uir.loop : %ty1, %ty2 { ... }
//        uir.loop { ... }
ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse optional result type operands: `: %ty1, %ty2`.
  SmallVector<OpAsmParser::UnresolvedOperand> resultTypeOperands;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseOperandList(resultTypeOperands))
      return failure();
    if (parser.resolveOperands(resultTypeOperands, anyTy, result.operands))
      return failure();
  }

  // Add result types.
  result.addTypes(SmallVector<Type>(resultTypeOperands.size(), anyTy));

  // Parse body region.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  // Print result type operands if any.
  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleaveComma(getResultTypes(), p,
                          [&](Value v) { p.printOperand(v); });
  }

  // Print body region.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  p.printOptionalAttrDict((*this)->getAttrs());
}

LogicalResult LoopOp::verify() {
  // SingleBlock trait checks body has one block.

  // Result count must match result type operand count.
  if (getNumResults() != getResultTypes().size())
    return emitOpError("result count must match result type operand count");

  return success();
}

//===----------------------------------------------------------------------===//
// ExprOp
//===----------------------------------------------------------------------===//

// Parse: uir.expr pin -1 : %ty { ... }
//        uir.expr pin : %ty { ... }
//        uir.expr : %ty { ... }
//        uir.expr { ... }
ParseResult ExprOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse optional `pin` keyword and phase shift.
  bool isPinned = succeeded(parser.parseOptionalKeyword("pin"));
  if (isPinned) {
    result.addAttribute("pin", builder.getUnitAttr());
    // Parse optional integer phase shift (defaults to 0).
    int32_t shift = 0;
    auto optResult = parser.parseOptionalInteger(shift);
    if (optResult.has_value() && failed(*optResult))
      return failure();
    result.addAttribute("phaseShift", builder.getSI32IntegerAttr(shift));
  } else {
    result.addAttribute("phaseShift", builder.getSI32IntegerAttr(0));
  }

  // Parse optional result type operands: `: %ty1, %ty2`.
  SmallVector<OpAsmParser::UnresolvedOperand> resultTypeOperands;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseOperandList(resultTypeOperands))
      return failure();
    if (parser.resolveOperands(resultTypeOperands, anyTy, result.operands))
      return failure();
  }

  // Add result types.
  result.addTypes(SmallVector<Type>(resultTypeOperands.size(), anyTy));

  // Parse body region.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void ExprOp::print(OpAsmPrinter &p) {
  // Print `pin` keyword and phase shift.
  if (getPin()) {
    p << " pin";
    if (getPhaseShift() != 0)
      p << ' ' << getPhaseShift();
  }

  // Print result type operands if any.
  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleaveComma(getResultTypes(), p,
                          [&](Value v) { p.printOperand(v); });
  }

  // Print body region.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);

  // Print attrs, eliding pin and phaseShift (printed above).
  p.printOptionalAttrDict((*this)->getAttrs(), {"pin", "phaseShift"});
}

LogicalResult ExprOp::verify() {
  // SingleBlock trait checks body has one block.

  // If not pinned, phaseShift must be 0.
  if (!getPin() && getPhaseShift() != 0)
    return emitOpError(
        "floating expression (no 'pin') must have phaseShift = 0");

  // Result count must match result type operand count.
  if (getNumResults() != getResultTypes().size())
    return emitOpError("result count must match result type operand count");

  return success();
}

//===----------------------------------------------------------------------===//
// PinOp
//===----------------------------------------------------------------------===//

// Parse: uir.pin %val, 0 : !hir.any
//        uir.pin %a, %b, -1 : !hir.any, !hir.any
// Operands are comma-separated SSA values, followed by a comma and the
// integer phase offset, then `: type(results)`.
ParseResult PinOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse first operand (required).
  SmallVector<OpAsmParser::UnresolvedOperand> operands;
  OpAsmParser::UnresolvedOperand operand;
  if (parser.parseOperand(operand))
    return failure();
  operands.push_back(operand);

  // Parse additional operands: `, %val`. Stop when the next `,` is
  // followed by an integer (the phase offset) rather than an SSA value.
  while (true) {
    if (parser.parseComma())
      return failure();
    // Try parsing an operand. If it fails, this must be the integer.
    auto optOperand = parser.parseOptionalOperand(operand);
    if (optOperand.has_value()) {
      if (failed(*optOperand))
        return failure();
      operands.push_back(operand);
    } else {
      break;
    }
  }

  // Parse the integer phase offset (the comma was already consumed).
  int32_t offset = 0;
  if (parser.parseInteger(offset))
    return failure();
  result.addAttribute("phaseOffset", builder.getSI32IntegerAttr(offset));

  // Resolve operands.
  if (parser.resolveOperands(operands, anyTy, result.operands))
    return failure();

  // Parse `: type(results)`.
  SmallVector<Type> resultTypes;
  if (parser.parseColonTypeList(resultTypes))
    return failure();
  result.addTypes(resultTypes);

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void PinOp::print(OpAsmPrinter &p) {
  p << ' ';
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p.printOperand(v); });
  p << ", " << getPhaseOffset();
  p << " : ";
  llvm::interleaveComma(getOutputs().getTypes(), p);

  p.printOptionalAttrDict((*this)->getAttrs(), {"phaseOffset"});
}

LogicalResult PinOp::verify() {
  if (getInputs().size() != getOutputs().size())
    return emitOpError("input count must match output count");
  return success();
}

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.cpp.inc"
