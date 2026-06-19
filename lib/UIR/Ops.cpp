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

/// Find the nearest enclosing `uir.loop`, or null if there is none.
static LoopOp getEnclosingLoop(Operation *op) {
  for (auto *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto loopOp = dyn_cast<LoopOp>(parent))
      return loopOp;
  return {};
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
  auto loopOp = getEnclosingLoop(*this);
  if (!loopOp)
    return emitOpError("must be nested inside a 'uir.loop'");
  if (getValues().size() != loopOp.getInits().size())
    return emitOpError("carried value count must match enclosing loop's "
                       "iteration argument count");
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

  // Parse optional attributes (before regions).
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

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

  // Print attributes before regions.
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs());

  // Print then region.
  p << ' ';
  p.printRegion(getThenRegion(), /*printEntryBlockArgs=*/false);

  // Print else region if non-empty.
  if (!getElseRegion().empty()) {
    p << " else ";
    p.printRegion(getElseRegion(), /*printEntryBlockArgs=*/false);
  }
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

// # Custom Parser for LoopOp
//
// The loop carries a list of initial values for its loop-carried iteration
// arguments, each with a type operand, followed by the result type operands.
// The iteration arguments become the body block's arguments. The accepted
// forms are:
//   uir.loop (%x = %a, %y = %b : %ta, %tb) : %r_ty1, %r_ty2 { ... }
//   uir.loop : %r_ty1, %r_ty2 { ... }
//   uir.loop (%x = %a : %ta) { ... }
//   uir.loop { ... }
ParseResult LoopOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse optional iteration arguments: `(%x = %init, ... : %ty, ...)`.
  SmallVector<OpAsmParser::Argument> iterArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> initOperands;
  SmallVector<OpAsmParser::UnresolvedOperand> initTypeOperands;
  if (succeeded(parser.parseOptionalLParen())) {
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        arg.type = anyTy;
        OpAsmParser::UnresolvedOperand init;
        if (parser.parseArgument(arg) || parser.parseEqual() ||
            parser.parseOperand(init))
          return failure();
        iterArgs.push_back(arg);
        initOperands.push_back(init);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseColon() || parser.parseOperandList(initTypeOperands) ||
          parser.parseRParen())
        return failure();
    }
  }

  // Parse optional result type operands: `: %ty1, %ty2`.
  SmallVector<OpAsmParser::UnresolvedOperand> resultTypeOperands;
  if (succeeded(parser.parseOptionalColon())) {
    if (parser.parseOperandList(resultTypeOperands))
      return failure();
  }

  // Resolve operands in segment order: inits, initTypes, resultTypes.
  if (parser.resolveOperands(initOperands, anyTy, result.operands) ||
      parser.resolveOperands(initTypeOperands, anyTy, result.operands) ||
      parser.resolveOperands(resultTypeOperands, anyTy, result.operands))
    return failure();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(initOperands.size()),
                           static_cast<int32_t>(initTypeOperands.size()),
                           static_cast<int32_t>(resultTypeOperands.size())}));

  // Add result types.
  result.addTypes(SmallVector<Type>(resultTypeOperands.size(), anyTy));

  // Parse optional attributes (before region).
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse body region with the iteration arguments as block arguments.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body, iterArgs))
    return failure();

  return success();
}

void LoopOp::print(OpAsmPrinter &p) {
  // Print iteration arguments if any.
  auto inits = getInits();
  if (!inits.empty()) {
    auto bodyArgs = getBody().front().getArguments();
    p << " (";
    for (auto [i, init] : llvm::enumerate(inits)) {
      if (i)
        p << ", ";
      p.printOperand(bodyArgs[i]);
      p << " = ";
      p.printOperand(init);
    }
    p << " : ";
    llvm::interleaveComma(getInitTypes(), p,
                          [&](Value v) { p.printOperand(v); });
    p << ')';
  }

  // Print result type operands if any.
  if (!getResultTypes().empty()) {
    p << " : ";
    llvm::interleaveComma(getResultTypes(), p,
                          [&](Value v) { p.printOperand(v); });
  }

  // Print attributes before region.
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {getOperandSegmentSizeAttr()});

  // Print body region.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult LoopOp::verify() {
  // SingleBlock trait checks body has one block.

  // Each initial value needs a corresponding type operand.
  if (getInits().size() != getInitTypes().size())
    return emitOpError("init count must match init type operand count");

  // Result count must match result type operand count.
  if (getNumResults() != getResultTypes().size())
    return emitOpError("result count must match result type operand count");

  return success();
}

LogicalResult LoopOp::verifyRegions() {
  // The body block has one argument per loop-carried iteration argument. The
  // arity of `uir.continue` terminators is checked by ContinueOp's verifier.
  auto &bodyBlock = getBody().front();
  if (bodyBlock.getNumArguments() != getInits().size())
    return emitOpError("body block argument count must match init count");

  // The body block must be terminated by an op that advances or exits the
  // loop. `uir.yield` only terminates `uir.expr` and `uir.if` regions.
  auto *terminator = bodyBlock.getTerminator();
  if (!isa<ContinueOp, BreakOp, ReturnOp, UnreachableOp>(terminator))
    return terminator->emitOpError(
        "cannot terminate a 'uir.loop' body; use 'uir.continue', 'uir.break', "
        "'uir.return', or 'uir.unreachable'");

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

  // Parse optional attributes (before region), eliding pin/phaseShift.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse body region.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
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

  // Print attrs before region, eliding pin and phaseShift (printed above).
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(),
                                     {"pin", "phaseShift"});

  // Print body region.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
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
  // Signature must end with uir.signature or uir.unreachable.
  auto *sigTerm = getSignature().front().getTerminator();
  if (!isa<SignatureOp, UnreachableOp>(sigTerm))
    return sigTerm->emitOpError()
           << "expected 'uir.signature' or 'uir.unreachable' terminator "
              "in signature region";

  // Signature must not contain uir.return anywhere (even nested).
  auto sigWalkResult = getSignature().walk([](ReturnOp op) {
    op.emitOpError() << "cannot appear in signature region";
    return WalkResult::interrupt();
  });
  if (sigWalkResult.wasInterrupted())
    return failure();

  // Body must end with uir.return or uir.unreachable.
  auto *bodyTerm = getBody().front().getTerminator();
  if (!isa<ReturnOp, UnreachableOp>(bodyTerm))
    return bodyTerm->emitOpError()
           << "expected 'uir.return' or 'uir.unreachable' terminator "
              "in function body";

  // Body must not contain uir.signature anywhere (even nested).
  auto bodyWalkResult = getBody().walk([](SignatureOp op) {
    op.emitOpError() << "cannot appear in function body";
    return WalkResult::interrupt();
  });
  if (bodyWalkResult.wasInterrupted())
    return failure();

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

  // Check block arguments in signature region.
  if (!getSignature().empty()) {
    auto numArgs = getArgPhases().size();
    if (getSignature().front().getNumArguments() != numArgs)
      return emitOpError() << "signature region has "
                           << getSignature().front().getNumArguments()
                           << " block arguments but function has " << numArgs
                           << " arguments";
  }

  return success();
}

LogicalResult SplitFuncOp::verifyRegions() {
  if (getSignature().empty())
    return success();

  // Signature must end with uir.signature or uir.unreachable.
  auto *sigTerm = getSignature().front().getTerminator();
  if (!isa<SignatureOp, UnreachableOp>(sigTerm))
    return sigTerm->emitOpError()
           << "expected 'uir.signature' or 'uir.unreachable' terminator "
              "in signature region";

  // Signature must not contain uir.return anywhere (even nested).
  auto walkResult = getSignature().walk([](ReturnOp op) {
    op.emitOpError() << "cannot appear in signature region";
    return WalkResult::interrupt();
  });
  if (walkResult.wasInterrupted())
    return failure();

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
