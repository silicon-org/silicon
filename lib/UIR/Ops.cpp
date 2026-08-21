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

std::pair<int32_t, int32_t> silicon::uir::DeclaredPhases::span() const {
  int32_t lo = 0, hi = 0;
  for (auto phase : argPhases) {
    lo = std::min(lo, phase);
    hi = std::max(hi, phase);
  }
  for (auto phase : resultPhases) {
    lo = std::min(lo, phase);
    hi = std::max(hi, phase);
  }
  return {lo, hi};
}

std::optional<silicon::uir::DeclaredPhases>
silicon::uir::getDeclaredPhases(Operation *op) {
  if (auto funcOp = dyn_cast_or_null<FuncOp>(op))
    return DeclaredPhases{funcOp.getArgPhases(), funcOp.getResultPhases()};
  if (auto funcOp = dyn_cast_or_null<SplitFuncOp>(op))
    return DeclaredPhases{funcOp.getArgPhases(), funcOp.getResultPhases()};
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  // A yield provides exactly one value per result of the op whose region it
  // terminates. Non-provided slots carry a `uir.unreachable_value`, never a
  // missing operand.
  auto *parentOp = (*this)->getParentOp();
  if (getValues().size() != parentOp->getNumResults())
    return emitOpError() << "provides " << getValues().size()
                         << " values but the enclosing '"
                         << parentOp->getName().getStringRef() << "' declares "
                         << parentOp->getNumResults() << " results";
  return success();
}

//===----------------------------------------------------------------------===//
// BreakOp
//===----------------------------------------------------------------------===//

LogicalResult BreakOp::verify() {
  auto loopOp = getEnclosingLoop(*this);
  if (!loopOp)
    return emitOpError("must be nested inside a 'uir.loop'");
  // A break provides exactly one value per result of the loop it exits.
  // Non-provided slots carry a `uir.unreachable_value`, never a missing
  // operand.
  if (getValues().size() != loopOp.getNumResults())
    return emitOpError() << "provides " << getValues().size()
                         << " values but the target 'uir.loop' declares "
                         << loopOp.getNumResults() << " results";
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
  if (!isa<ContinueOp, BreakOp, ReturnOp, PhaseReturnOp, UnreachableOp>(
          terminator))
    return terminator->emitOpError(
        "cannot terminate a 'uir.loop' body; use 'uir.continue', 'uir.break', "
        "'uir.return', 'uir.phase_return', or 'uir.unreachable'");

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

// Parse: uir.pin %val, 0
//        uir.pin %a, %b, -1
// Operands are comma-separated SSA values, followed by a comma and the
// integer phase offset. All operands and results are `!hir.any` and not
// spelled out in the assembly; the op has one result per operand.
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

  // Resolve operands and add one `!hir.any` result per operand.
  if (parser.resolveOperands(operands, anyTy, result.operands))
    return failure();
  result.addTypes(SmallVector<Type>(operands.size(), anyTy));

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  return success();
}

void PinOp::print(OpAsmPrinter &p) {
  p << ' ';
  llvm::interleaveComma(getInputs(), p, [&](Value v) { p.printOperand(v); });
  p << ", " << getPhaseOffset();
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
  // Climb to the enclosing function, remembering which of its regions holds
  // this op. `uir.signature` is the signature region's return-like exit and
  // may sit under structured control flow at any nesting depth, but only
  // inside the signature region.
  Operation *op = getOperation();
  while (op->getParentOp() && !isa<FuncOp, SplitFuncOp>(op->getParentOp()))
    op = op->getParentOp();
  auto *funcLike = op->getParentOp();
  if (!funcLike)
    return emitOpError() << "must be nested in the signature region of a "
                            "'uir.func' or 'uir.split_func'";

  unsigned numArgs, numResults;
  Region *sigRegion;
  if (auto func = dyn_cast<FuncOp>(funcLike)) {
    numArgs = func.getArgNames().size();
    numResults = func.getResultNames().size();
    sigRegion = &func.getSignature();
  } else {
    auto splitFunc = cast<SplitFuncOp>(funcLike);
    numArgs = splitFunc.getArgNames().size();
    numResults = splitFunc.getResultNames().size();
    sigRegion = &splitFunc.getSignature();
  }
  if (op->getParentRegion() != sigRegion)
    return emitOpError() << "must be nested in the signature region of the "
                            "enclosing function";

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

  // A uir.signature outside the signature region is rejected by its own
  // verifier, so the body needs no walk for it here.

  // Every return in the body provides exactly one value per declared
  // function result, no matter how deeply it is nested.
  auto numResults = getResultPhases().size();
  auto returnWalkResult = getBody().walk([&](ReturnOp op) {
    if (op.getValues().size() != numResults) {
      op.emitOpError() << "provides " << op.getValues().size()
                       << " values but the enclosing function declares "
                       << numResults << " results";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (returnWalkResult.wasInterrupted())
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
//   uir.split_func @name(%arg: phase, ...) -> (result: phase, ...)
//   attributes {...} {
//     <signature region>
//   } impl @sym [phase: @sym, ...]
//
// The `impl @sym` clause is present in the sorted state only; the phase list
// is always printed, and is empty in the declared and sorted states.

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

  // Parse optional attributes.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse signature region with the inline arguments as entry block args.
  auto *region = result.addRegion();
  if (parser.parseRegion(*region, args))
    return failure();

  // Parse the optional implementation reference: `impl @sym`.
  if (succeeded(parser.parseOptionalKeyword("impl"))) {
    FlatSymbolRefAttr implAttr;
    if (parser.parseAttribute(implAttr))
      return failure();
    props.impl = implAttr;
  }

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
  p << ")";

  // Print optional attributes, excluding the properties the syntax spells out
  // itself, before and after the signature region.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getSymVisibilityAttrName(), getArgNamesAttrName(),
       getArgPhasesAttrName(), getResultNamesAttrName(),
       getResultPhasesAttrName(), getImplAttrName(), getPhaseNumbersAttrName(),
       getPhaseFuncsAttrName()});

  // Print signature region without entry block arguments.
  p << ' ';
  p.printRegion(getSignature(), /*printEntryBlockArgs=*/false);

  // Print the implementation reference of the sorted state.
  if (auto impl = getImplAttr()) {
    p << " impl ";
    p.printAttribute(impl);
  }

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
  if (!phaseNumbers.empty())
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

  // The three states are exclusive: an implementation reference means the
  // function has been sorted but not yet split into per-phase functions, so no
  // phase list exists yet; a phase list means splitting has consumed the
  // implementation.
  if (getImplAttr() && !getPhaseNumbers().empty())
    return emitOpError() << "has both an implementation reference and a phase "
                            "list; a split function has no implementation left";

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

LogicalResult
SplitFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  // The phase list is the authoritative mapping callers resolve a
  // `uir.phase_call` through: every entry must name a function that exists, and
  // the phases must be listed in increasing order so a phase maps to at most
  // one function.
  int32_t previous = 0;
  for (auto [index, phase] : llvm::enumerate(getPhaseNumbers())) {
    if (index != 0 && phase <= previous)
      return emitOpError() << "maps phase " << phase << " after phase "
                           << previous
                           << "; phases must be listed in increasing order";
    previous = phase;
    auto entry = cast<FlatSymbolRefAttr>(getPhaseFuncs()[index]);
    if (!symbolTable.lookupNearestSymbolFrom(getOperation(), entry))
      return emitOpError() << "'" << entry.getValue()
                           << "' does not reference a valid function";
  }

  auto impl = getImplAttr();
  if (!impl)
    return success();

  auto *implOp = symbolTable.lookupNearestSymbolFrom(getOperation(), impl);
  if (!implOp)
    return emitOpError() << "'" << impl.getValue()
                         << "' does not reference a valid function";
  auto funcOp = dyn_cast<FuncOp>(implOp);
  if (!funcOp)
    return emitOpError() << "'" << impl.getValue()
                         << "' does not reference a 'uir.func'";

  // The implementation describes the same interface this op publishes: sorting
  // rewrites a function's interior, never its declaration.
  if (funcOp.getArgPhases() != getArgPhases())
    return emitOpError() << "argument phases " << getArgPhases()
                         << " do not match the phases @" << impl.getValue()
                         << " declares for its arguments "
                         << funcOp.getArgPhases();
  if (funcOp.getResultPhases() != getResultPhases())
    return emitOpError() << "result phases " << getResultPhases()
                         << " do not match the phases @" << impl.getValue()
                         << " declares for its results "
                         << funcOp.getResultPhases();

  // Module-ness is part of that interface and belongs to this op alone. An
  // implementation that kept the marker would carry a declaration fact nothing
  // reads, and would lose it silently once splitting retires it.
  if (funcOp.getIsModuleAttr())
    return emitOpError() << "implementation @" << impl.getValue()
                         << " is marked 'isModule'; module-ness belongs to the "
                            "declaration, not the implementation";

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
// PhaseOp
//===----------------------------------------------------------------------===//

// # Custom Parser for PhaseOp
//
// The assembly format is:
//   %r0, %r1 = uir.phase 0 { ... }
//
// All results are `!hir.any` and are not spelled out in the assembly; the
// result count is inferred from the first `uir.phase_return` found in the
// region (all of them agree on the yield count, checked by the verifier).
ParseResult PhaseOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse the phase number.
  int32_t phase;
  if (parser.parseInteger(phase))
    return failure();
  result.addAttribute("phase", builder.getSI32IntegerAttr(phase));

  // Parse optional attributes (before the region).
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse the body region.
  auto *body = result.addRegion();
  if (parser.parseRegion(*body))
    return failure();

  // Infer the result count from the first phase return in the region.
  unsigned numResults = 0;
  body->walk([&](PhaseReturnOp op) {
    numResults = op.getYields().size();
    return WalkResult::interrupt();
  });
  result.addTypes(SmallVector<Type>(numResults, anyTy));

  return success();
}

void PhaseOp::print(OpAsmPrinter &p) {
  p << ' ' << getPhase();
  p.printOptionalAttrDictWithKeyword((*this)->getAttrs(), {"phase"});
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult PhaseOp::verifyRegions() {
  // The block must end in a phase return, or in unreachable when all paths
  // exit early through phase returns nested in structured control flow.
  auto *terminator = getBody().front().getTerminator();
  if (!isa<PhaseReturnOp, UnreachableOp>(terminator))
    return terminator->emitOpError()
           << "cannot terminate a 'uir.phase' body; use 'uir.phase_return' "
              "or 'uir.unreachable'";

  // Function exits inside a phase are phase returns; the unsorted
  // `uir.return` must not appear in phase content, where it would smuggle
  // function results past the phase return contract.
  auto returnWalkResult = getBody().walk([](ReturnOp op) {
    op.emitOpError() << "cannot appear inside a 'uir.phase'";
    return WalkResult::interrupt();
  });
  if (returnWalkResult.wasInterrupted())
    return failure();

  // Every phase return in the region provides this op's results. Which
  // function result values it must carry is part of the sorted-form contract
  // diagnosed by function splitting, not the verifier; the rules are the
  // same in the body and signature regions.
  bool anyPhaseReturn = false;
  auto walkResult = getBody().walk([&](PhaseReturnOp op) {
    anyPhaseReturn = true;
    if (op.getYields().size() != getNumResults()) {
      op.emitOpError() << "yields " << op.getYields().size()
                       << " values but enclosing 'uir.phase' has "
                       << getNumResults() << " results";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  // Results are provided exclusively by phase returns; a phase without any
  // cannot produce results (and could not round-trip, since the printed form
  // infers the result count from the first phase return).
  if (!anyPhaseReturn && getNumResults() != 0)
    return emitOpError() << "has " << getNumResults()
                         << " results but no 'uir.phase_return' to provide "
                            "them";

  return success();
}

//===----------------------------------------------------------------------===//
// PhaseReturnOp
//===----------------------------------------------------------------------===//

LogicalResult PhaseReturnOp::verify() {
  if (!hasAncestorOp(*this, "uir.phase"))
    return emitOpError("must be nested inside a 'uir.phase'");
  return success();
}

//===----------------------------------------------------------------------===//
// PhaseCallOp
//===----------------------------------------------------------------------===//

// # Custom Parser for PhaseCallOp
//
// The assembly format is:
//   uir.phase_call @callee[phase] (%args) context %ctx
//     : (%typeOfArgs) -> (%typeOfResults) context
//
// The `context %ctx` operand and the trailing `context` keyword (marking the
// chained context result) are both optional. All operands and results are
// `!hir.any`; the result count is the type-of-result count plus the context
// result.
ParseResult PhaseCallOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse the callee and phase: `@callee[phase]`.
  StringAttr calleeAttr;
  int32_t phase;
  if (parser.parseSymbolName(calleeAttr) || parser.parseLSquare() ||
      parser.parseInteger(phase) || parser.parseRSquare())
    return failure();
  result.addAttribute("callee", FlatSymbolRefAttr::get(calleeAttr));
  result.addAttribute("calleePhase", builder.getSI32IntegerAttr(phase));

  // Parse the argument list and optional context operand.
  SmallVector<OpAsmParser::UnresolvedOperand> argOperands;
  if (parser.parseOperandList(argOperands, OpAsmParser::Delimiter::Paren))
    return failure();
  bool hasContextIn = false;
  OpAsmParser::UnresolvedOperand contextOperand;
  if (succeeded(parser.parseOptionalKeyword("context"))) {
    if (parser.parseOperand(contextOperand))
      return failure();
    hasContextIn = true;
  }

  // Parse the type operands: `: (%typeOfArgs) -> (%typeOfResults)`.
  SmallVector<OpAsmParser::UnresolvedOperand> typeOfArgs, typeOfResults;
  if (parser.parseColon() ||
      parser.parseOperandList(typeOfArgs, OpAsmParser::Delimiter::Paren) ||
      parser.parseArrow() ||
      parser.parseOperandList(typeOfResults, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse the optional trailing `context` keyword marking the context result.
  bool hasContextOut = succeeded(parser.parseOptionalKeyword("context"));

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve operands in segment order.
  if (parser.resolveOperands(argOperands, anyTy, result.operands) ||
      parser.resolveOperands(typeOfArgs, anyTy, result.operands) ||
      parser.resolveOperands(typeOfResults, anyTy, result.operands))
    return failure();
  if (hasContextIn &&
      parser.resolveOperand(contextOperand, anyTy, result.operands))
    return failure();
  result.addAttribute(
      getOperandSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr({static_cast<int32_t>(argOperands.size()),
                                    static_cast<int32_t>(typeOfArgs.size()),
                                    static_cast<int32_t>(typeOfResults.size()),
                                    hasContextIn ? 1 : 0}));

  // Add result types.
  result.addAttribute(
      getResultSegmentSizeAttr(),
      builder.getDenseI32ArrayAttr(
          {static_cast<int32_t>(typeOfResults.size()), hasContextOut ? 1 : 0}));
  result.addTypes(
      SmallVector<Type>(typeOfResults.size() + hasContextOut, anyTy));

  return success();
}

void PhaseCallOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getCallee());
  p << '[' << getCalleePhase() << "] (";
  llvm::interleaveComma(getArguments(), p, [&](Value v) { p.printOperand(v); });
  p << ')';
  if (auto contextIn = getContextIn()) {
    p << " context ";
    p.printOperand(contextIn);
  }
  p << " : (";
  llvm::interleaveComma(getTypeOfArgs(), p,
                        [&](Value v) { p.printOperand(v); });
  p << ") -> (";
  llvm::interleaveComma(getTypeOfResults(), p,
                        [&](Value v) { p.printOperand(v); });
  p << ')';
  if (getContextOut())
    p << " context";
  p.printOptionalAttrDict((*this)->getAttrs(),
                          {"callee", "calleePhase", getOperandSegmentSizeAttr(),
                           getResultSegmentSizeAttr()});
}

LogicalResult PhaseCallOp::verify() {
  if (!hasAncestorOp(*this, "uir.phase"))
    return emitOpError("must be nested inside a 'uir.phase'");
  if (getArguments().size() != getTypeOfArgs().size())
    return emitOpError() << "has " << getArguments().size() << " arguments but "
                         << getTypeOfArgs().size() << " typeOfArgs";
  if (getResults().size() != getTypeOfResults().size())
    return emitOpError() << "has " << getResults().size() << " results but "
                         << getTypeOfResults().size() << " typeOfResults";
  return success();
}

LogicalResult
PhaseCallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = getCalleeAttr();
  auto *calleeOp = symbolTable.lookupNearestSymbolFrom(getOperation(), callee);
  if (!calleeOp)
    return emitOpError() << "'" << callee.getValue()
                         << "' does not reference a valid function";

  // A phase call names the callee's late-binding interface, never its
  // implementation: sorting snapshots every declaration in the module before it
  // decomposes any call, and already-split library code exposes nothing but its
  // interface, so a `uir.func` callee here would call an implementation
  // directly and bypass the interface.
  auto calleeSplit = dyn_cast<SplitFuncOp>(calleeOp);
  if (!calleeSplit)
    return emitOpError() << "'" << callee.getValue()
                         << "' does not reference a 'uir.split_func'";
  auto argPhases = calleeSplit.getArgPhases();
  auto resultPhases = calleeSplit.getResultPhases();

  // The callee's phase span is the contiguous integer range from its earliest
  // to its latest declared argument or result phase, always including phase 0.
  // Callers thread one entry per phase in the span, so every phase in the
  // range — including gap phases carrying no arguments or results — is a legal
  // entry, derivable from the declaration alone.
  auto [lo, hi] = DeclaredPhases{argPhases, resultPhases}.span();

  auto phase = getCalleePhase();
  if (phase < lo || phase > hi)
    return emitOpError() << "callee @" << callee.getValue() << " has no phase "
                         << phase << " in its declared span [" << lo << ", "
                         << hi << "]";

  // The call's arguments and results are the callee's declared arguments and
  // results at this phase — none for a gap phase inside the span.
  auto expectedArgs = llvm::count(argPhases, phase);
  if (static_cast<int64_t>(getArguments().size()) != expectedArgs)
    return emitOpError() << "has " << getArguments().size()
                         << " arguments but callee @" << callee.getValue()
                         << " declares " << expectedArgs
                         << " arguments at phase " << phase;
  auto expectedResults = llvm::count(resultPhases, phase);
  if (static_cast<int64_t>(getTypeOfResults().size()) != expectedResults)
    return emitOpError() << "has " << getTypeOfResults().size()
                         << " results but callee @" << callee.getValue()
                         << " declares " << expectedResults
                         << " results at phase " << phase;

  // Context values chain consecutive entries across the span.
  bool needsContextIn = phase != lo;
  if (needsContextIn && !getContextIn())
    return emitOpError() << "requires a context operand: phase " << phase
                         << " is not the callee's first span phase";
  if (!needsContextIn && getContextIn())
    return emitOpError() << "cannot have a context operand: phase " << phase
                         << " is the callee's first span phase";
  bool needsContextOut = phase != hi;
  if (needsContextOut && !getContextOut())
    return emitOpError() << "requires a context result: phase " << phase
                         << " is not the callee's last span phase";
  if (!needsContextOut && getContextOut())
    return emitOpError() << "cannot have a context result: phase " << phase
                         << " is the callee's last span phase";

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

// # Custom Parser for CallOp
//
// The assembly format is:
//   uir.call @callee(%args) : (%typeOfArgs) -> (%typeOfResults)
//     [argPhases] -> [resultPhases]
//
// All operands and results are `!hir.any` and not spelled out in the
// assembly; the result count is the type-of-result count.
ParseResult CallOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = builder.getType<hir::AnyType>();

  // Parse the callee.
  StringAttr calleeAttr;
  if (parser.parseSymbolName(calleeAttr))
    return failure();
  result.addAttribute("callee", FlatSymbolRefAttr::get(calleeAttr));

  // Parse the argument list and type operands.
  SmallVector<OpAsmParser::UnresolvedOperand> argOperands, typeOfArgs,
      typeOfResults;
  if (parser.parseOperandList(argOperands, OpAsmParser::Delimiter::Paren) ||
      parser.parseColon() ||
      parser.parseOperandList(typeOfArgs, OpAsmParser::Delimiter::Paren) ||
      parser.parseArrow() ||
      parser.parseOperandList(typeOfResults, OpAsmParser::Delimiter::Paren))
    return failure();

  // Parse the phase lists: `[argPhases] -> [resultPhases]`.
  auto parsePhases = [&](StringRef attrName) -> ParseResult {
    SmallVector<int32_t> phases;
    if (parser.parseLSquare())
      return failure();
    if (failed(parser.parseOptionalRSquare())) {
      do {
        int32_t phase;
        if (parser.parseInteger(phase))
          return failure();
        phases.push_back(phase);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRSquare())
        return failure();
    }
    result.addAttribute(attrName, builder.getDenseI32ArrayAttr(phases));
    return success();
  };
  if (parsePhases("argPhases") || parser.parseArrow() ||
      parsePhases("resultPhases"))
    return failure();

  // Parse optional attributes.
  if (parser.parseOptionalAttrDict(result.attributes))
    return failure();

  // Resolve operands in segment order.
  if (parser.resolveOperands(argOperands, anyTy, result.operands) ||
      parser.resolveOperands(typeOfArgs, anyTy, result.operands) ||
      parser.resolveOperands(typeOfResults, anyTy, result.operands))
    return failure();
  result.addAttribute(getOperandSegmentSizeAttr(),
                      builder.getDenseI32ArrayAttr(
                          {static_cast<int32_t>(argOperands.size()),
                           static_cast<int32_t>(typeOfArgs.size()),
                           static_cast<int32_t>(typeOfResults.size())}));

  // Add result types.
  result.addTypes(SmallVector<Type>(typeOfResults.size(), anyTy));

  return success();
}

void CallOp::print(OpAsmPrinter &p) {
  p << ' ';
  p.printSymbolName(getCallee());
  p << '(';
  llvm::interleaveComma(getArguments(), p, [&](Value v) { p.printOperand(v); });
  p << ") : (";
  llvm::interleaveComma(getTypeOfArgs(), p,
                        [&](Value v) { p.printOperand(v); });
  p << ") -> (";
  llvm::interleaveComma(getTypeOfResults(), p,
                        [&](Value v) { p.printOperand(v); });
  p << ") [";
  llvm::interleaveComma(getArgPhases(), p);
  p << "] -> [";
  llvm::interleaveComma(getResultPhases(), p);
  p << ']';
  p.printOptionalAttrDict(
      (*this)->getAttrs(),
      {"callee", "argPhases", "resultPhases", getOperandSegmentSizeAttr()});
}

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

  // When the callee is a unified function (or an already-split witness), the
  // call's per-argument and per-result phase offsets are the callee's declared
  // phases, so they must line up one-to-one. Call decomposition relies on this
  // to derive the phase entries from the callee declaration alone; without the
  // check a phase present on the call but not the callee would silently drop
  // an argument or result.
  auto declared = getDeclaredPhases(calleeOp);
  if (!declared)
    return success();
  auto calleeArgPhases = declared->argPhases;
  auto calleeResultPhases = declared->resultPhases;

  if (getArgPhases() != calleeArgPhases)
    return emitOpError() << "argument phases " << getArgPhases()
                         << " do not match the phases @" << callee.getValue()
                         << " declares for its arguments " << calleeArgPhases;
  if (getResultPhases() != calleeResultPhases)
    return emitOpError() << "result phases " << getResultPhases()
                         << " do not match the phases @" << callee.getValue()
                         << " declares for its results " << calleeResultPhases;

  return success();
}

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.cpp.inc"
