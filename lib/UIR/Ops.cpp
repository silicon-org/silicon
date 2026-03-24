//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Types.h"
#include "silicon/Support/AsmParser.h"
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

LogicalResult LoopOp::verifyRegions() {
  // A yield inside the loop body means "continue to next iteration" and must
  // not carry values. Use uir.break to exit the loop with values.
  auto *terminator = getBody().front().getTerminator();
  if (auto yieldOp = dyn_cast<YieldOp>(terminator)) {
    if (yieldOp.getValues().size() != 0)
      return yieldOp.emitOpError(
          "inside loop body must have no values (use 'uir.break' to exit with "
          "values)");
  }
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

//===----------------------------------------------------------------------===//
// SignatureOp
//===----------------------------------------------------------------------===//

LogicalResult SignatureOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  unsigned numArgs, numResults;

  if (auto func = dyn_cast<FuncOp>(parentOp)) {
    numArgs = func.getArgNames().size();
    numResults = func.getResultNames().size();
  } else if (auto func = dyn_cast<SplitFuncOp>(parentOp)) {
    numArgs = func.getArgNames().size();
    numResults = func.getResultNames().size();
  } else {
    return success();
  }

  if (getTypeOfArgs().size() != numArgs)
    return emitOpError() << "has " << getTypeOfArgs().size()
                         << " argument types but parent function has "
                         << numArgs << " arguments";

  if (getTypeOfResults().size() != numResults)
    return emitOpError() << "has " << getTypeOfResults().size()
                         << " result types but parent function has "
                         << numResults << " results";

  return success();
}

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// # Custom Parser for FuncOp
//
// The assembly format is:
//   uir.func @name(%arg: phase, ...) -> (result: phase, ...) {
//     <signature region>
//   } {
//     <body region>
//   }

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<FuncOp::Properties>();
  auto *ctx = parser.getContext();
  auto builder = OpBuilder(ctx);
  auto anyType = hir::AnyType::get(ctx);

  // Parse optional visibility.
  StringAttr visAttr;
  if (parseSymbolVisibility(parser, visAttr))
    return failure();
  if (visAttr)
    props.sym_visibility = visAttr;

  // Parse symbol name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr))
    return failure();
  props.sym_name = nameAttr;

  // Parse argument list: (%name: phase, ...).
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Attribute> argNames;
  SmallVector<int32_t> argPhases;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      auto &arg = args.emplace_back();
      arg.type = anyType;
      if (parser.parseArgument(arg) || parser.parseColon())
        return failure();
      int32_t phase;
      if (parser.parseInteger(phase))
        return failure();
      argNames.push_back(builder.getStringAttr(arg.ssaName.name.drop_front()));
      argPhases.push_back(phase);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.argNames = builder.getArrayAttr(argNames);
  props.argPhases = builder.getDenseI32ArrayAttr(argPhases);

  // Parse result list: -> (name: phase, ...).
  SmallVector<Attribute> resultNames;
  SmallVector<int32_t> resultPhases;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      int32_t phase;
      if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
          parser.parseInteger(phase))
        return failure();
      resultNames.push_back(builder.getStringAttr(name));
      resultPhases.push_back(phase);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.resultNames = builder.getArrayAttr(resultNames);
  props.resultPhases = builder.getDenseI32ArrayAttr(resultPhases);

  // Parse optional attributes.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse signature and body regions, both sharing the inline arg definitions.
  auto *sigRegion = result.addRegion();
  if (parser.parseRegion(*sigRegion, args))
    return failure();
  auto *bodyRegion = result.addRegion();
  if (parser.parseRegion(*bodyRegion, args))
    return failure();

  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  printSymbolVisibility(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());

  // Print argument list with phases.
  p << '(';
  auto argPhases = getArgPhases();
  if (!getBody().empty()) {
    auto args = getBody().front().getArguments();
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      if (i)
        p << ", ";
      p.printRegionArgument(args[i], {}, /*omitType=*/true);
      p << ": " << argPhases[i];
    }
  }
  p << ')';

  // Print result list with phases.
  p << " -> (";
  auto resultNames = getResultNames();
  auto resultPhases = getResultPhases();
  for (size_t i = 0, e = resultNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(resultNames[i]).getValue() << ": " << resultPhases[i];
  }
  p << ")";

  // Print optional attributes, excluding properties we've already printed.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getSymVisibilityAttrName(), getArgNamesAttrName(),
       getArgPhasesAttrName(), getResultNamesAttrName(),
       getResultPhasesAttrName()});

  // Print signature and body regions without entry block arguments.
  p << ' ';
  p.printRegion(getSignature(), /*printEntryBlockArgs=*/false);
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  auto numArgs = getArgPhases().size();
  auto numResults = getResultPhases().size();

  if (getArgNames().size() != numArgs)
    return emitOpError() << "argNames has " << getArgNames().size()
                         << " entries but function has " << numArgs
                         << " arguments";

  if (getResultNames().size() != numResults)
    return emitOpError() << "resultNames has " << getResultNames().size()
                         << " entries but function has " << numResults
                         << " results";

  // Check block arguments in both regions.
  if (getSignature().front().getNumArguments() != numArgs)
    return emitOpError() << "signature region has "
                         << getSignature().front().getNumArguments()
                         << " block arguments but function has " << numArgs
                         << " arguments";

  if (getBody().front().getNumArguments() != numArgs)
    return emitOpError() << "body region has "
                         << getBody().front().getNumArguments()
                         << " block arguments but function has " << numArgs
                         << " arguments";

  return success();
}

LogicalResult FuncOp::verifyRegions() {
  // Make sure there are no signature terminators in the body region.
  for (auto &block : getBody())
    if (isa<SignatureOp>(block.getTerminator()))
      return block.getTerminator()->emitOpError()
             << "cannot appear in function body";
  return success();
}

void FuncOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  if ((&region != &getSignature() && &region != &getBody()) || region.empty())
    return;
  auto argNames = getArgNames();
  for (auto [name, arg] : llvm::zip(argNames, region.front().getArguments()))
    setNameFn(arg, cast<StringAttr>(name).getValue());
}

//===----------------------------------------------------------------------===//
// SplitFuncOp
//===----------------------------------------------------------------------===//

// # Custom Parser for SplitFuncOp
//
// The assembly format is:
//   uir.split_func @name(%arg: phase, ...) -> (result: phase, ...) {
//     <signature region>
//   } [phase: @sym, ...]

ParseResult SplitFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<SplitFuncOp::Properties>();
  auto *ctx = parser.getContext();
  auto builder = OpBuilder(ctx);

  // Parse optional visibility.
  StringAttr visAttr;
  if (parseSymbolVisibility(parser, visAttr))
    return failure();
  if (visAttr)
    props.sym_visibility = visAttr;

  // Parse symbol name.
  StringAttr nameAttr;
  if (parser.parseSymbolName(nameAttr))
    return failure();
  props.sym_name = nameAttr;

  // Parse argument list: (%name: phase, ...).
  auto anyType = hir::AnyType::get(ctx);
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Attribute> argNames;
  SmallVector<int32_t> argPhases;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      auto &arg = args.emplace_back();
      arg.type = anyType;
      if (parser.parseArgument(arg) || parser.parseColon())
        return failure();
      int32_t phase;
      if (parser.parseInteger(phase))
        return failure();
      argNames.push_back(builder.getStringAttr(arg.ssaName.name.drop_front()));
      argPhases.push_back(phase);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.argNames = builder.getArrayAttr(argNames);
  props.argPhases = builder.getDenseI32ArrayAttr(argPhases);

  // Parse result list: -> (name: phase, ...).
  SmallVector<Attribute> resultNames;
  SmallVector<int32_t> resultPhases;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      int32_t phase;
      if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
          parser.parseInteger(phase))
        return failure();
      resultNames.push_back(builder.getStringAttr(name));
      resultPhases.push_back(phase);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.resultNames = builder.getArrayAttr(resultNames);
  props.resultPhases = builder.getDenseI32ArrayAttr(resultPhases);

  // Parse signature region with the inline arguments as entry block args.
  auto *region = result.addRegion();
  if (parser.parseRegion(*region, args))
    return failure();

  // Parse phase map: [phase: @sym, ...].
  SmallVector<int32_t> phaseNumbers;
  SmallVector<Attribute> phaseFuncs;
  if (parser.parseLSquare())
    return failure();
  if (failed(parser.parseOptionalRSquare())) {
    do {
      int32_t phase;
      FlatSymbolRefAttr sym;
      if (parser.parseInteger(phase) || parser.parseColon() ||
          parser.parseAttribute(sym))
        return failure();
      phaseNumbers.push_back(phase);
      phaseFuncs.push_back(sym);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();
  }
  props.phaseNumbers = builder.getDenseI32ArrayAttr(phaseNumbers);
  props.phaseFuncs = builder.getArrayAttr(phaseFuncs);

  return success();
}

void SplitFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  printSymbolVisibility(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());

  // Print argument list with phases.
  p << '(';
  auto argPhases = getArgPhases();
  if (!getSignature().empty()) {
    auto args = getSignature().front().getArguments();
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      if (i)
        p << ", ";
      p.printRegionArgument(args[i], {}, /*omitType=*/true);
      p << ": " << argPhases[i];
    }
  }
  p << ')';

  // Print result list.
  p << " -> (";
  auto resultNames = getResultNames();
  auto resultPhases = getResultPhases();
  for (size_t i = 0, e = resultNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(resultNames[i]).getValue() << ": " << resultPhases[i];
  }
  p << ") ";

  // Print signature region without entry block arguments.
  p.printRegion(getSignature(), /*printEntryBlockArgs=*/false);

  // Print phase map.
  p << " [";
  auto phaseNumbers = getPhaseNumbers();
  auto phaseFuncs = getPhaseFuncs();
  for (size_t i = 0, e = phaseNumbers.size(); i < e; ++i) {
    if (i)
      p << ',';
    p.printNewline();
    p << "  " << phaseNumbers[i] << ": ";
    p.printAttribute(phaseFuncs[i]);
  }
  p.printNewline();
  p << ']';
}

LogicalResult SplitFuncOp::verify() {
  if (getArgPhases().size() != getArgNames().size())
    return emitOpError() << "argPhases has " << getArgPhases().size()
                         << " entries but function has " << getArgNames().size()
                         << " arguments";

  if (getResultPhases().size() != getResultNames().size())
    return emitOpError() << "resultPhases has " << getResultPhases().size()
                         << " entries but function has "
                         << getResultNames().size() << " results";

  if (getPhaseNumbers().size() != getPhaseFuncs().size())
    return emitOpError() << "phaseNumbers has " << getPhaseNumbers().size()
                         << " entries but phaseFuncs has "
                         << getPhaseFuncs().size() << " entries";

  return success();
}

LogicalResult SplitFuncOp::verifyRegions() {
  // Make sure signature region is terminated by SignatureOp or
  // UnreachableOp (when all branches have uir.signature inside CF).
  if (getSignature().empty())
    return success();
  auto &block = getSignature().front();
  if (!isa<SignatureOp, UnreachableOp>(block.getTerminator()))
    return block.getTerminator()->emitOpError()
           << "expected 'uir.signature' or 'uir.unreachable' terminator "
              "in signature region";
  return success();
}

void SplitFuncOp::getAsmBlockArgumentNames(Region &region,
                                           OpAsmSetValueNameFn setNameFn) {
  if (&region != &getSignature() || region.empty())
    return;
  auto argNames = getArgNames();
  for (auto [name, arg] : llvm::zip(argNames, region.front().getArguments()))
    setNameFn(arg, cast<StringAttr>(name).getValue());
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // Verify that the callee exists. We accept both uir.func and uir.split_func
  // as callees, since a callee may already be split when the caller is
  // verified.
  auto callee = (*this)->getAttrOfType<FlatSymbolRefAttr>("callee");
  if (!callee)
    return emitOpError("requires a 'callee' symbol reference attribute");

  auto *calleeOp = symbolTable.lookupNearestSymbolFrom(getOperation(), callee);
  if (!calleeOp)
    return emitOpError() << "'" << callee.getValue()
                         << "' does not reference a valid function";

  return success();
}

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.cpp.inc"
