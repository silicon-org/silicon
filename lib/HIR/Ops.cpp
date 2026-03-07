//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Support/AsmParser.h"
#include "silicon/Support/MLIR.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

// Handle `custom<IntAttr>` parsing.
static ParseResult parseIntAttr(OpAsmParser &parser, base::IntAttr &value) {
  auto result = mlir::FieldParser<DynamicAPInt>::parse(parser);
  if (failed(result))
    return failure();
  value = base::IntAttr::get(parser.getContext(), *result);
  return success();
}

// Handle `custom<IntAttr>` printing.
static void printIntAttr(OpAsmPrinter &printer, Operation *op,
                         base::IntAttr value) {
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
// FuncOp
//===----------------------------------------------------------------------===//

// # Custom Parser for FuncOp
//
// The assembly format is:
//   hir.func [visibility] @name(%arg, ...) -> (result, ...) { <body> }
//
// Argument names use `%` because they become SSA block arguments in the body
// region. All block args have type `!hir.any`. Result names are bare
// identifiers.

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<FuncOp::Properties>();
  auto *ctx = parser.getContext();
  auto builder = OpBuilder(ctx);
  auto anyType = AnyType::get(ctx);

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

  // Parse argument list: (%name [: type], ...).
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Attribute> argNames;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      auto &arg = args.emplace_back();
      arg.type = anyType;
      if (parser.parseArgument(arg))
        return failure();
      argNames.push_back(builder.getStringAttr(arg.ssaName.name.drop_front()));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.argNames = builder.getArrayAttr(argNames);

  // Parse result list: -> (name, ...).
  SmallVector<Attribute> resultNames;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      if (parser.parseKeywordOrString(&name))
        return failure();
      resultNames.push_back(builder.getStringAttr(name));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.resultNames = builder.getArrayAttr(resultNames);

  // Parse optional attributes.
  if (parser.parseOptionalAttrDictWithKeyword(result.attributes))
    return failure();

  // Parse body region with the inline arguments as entry block args.
  auto *region = result.addRegion();
  if (parser.parseRegion(*region, args))
    return failure();

  return success();
}

void FuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  printSymbolVisibility(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());

  // Print argument list. All block args are !hir.any, so omit the type.
  p << '(';
  if (!getBody().empty()) {
    auto args = getBody().front().getArguments();
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      if (i)
        p << ", ";
      p.printRegionArgument(args[i], {}, /*omitType=*/true);
    }
  }
  p << ')';

  // Print result list.
  p << " -> (";
  auto resultNames = getResultNames();
  for (size_t i = 0, e = resultNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(resultNames[i]).getValue();
  }
  p << ") ";

  // Print optional attributes, excluding properties we've already printed.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getSymVisibilityAttrName(),
                            getArgNamesAttrName(), getResultNamesAttrName()});

  // Print body region without entry block arguments.
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  if (getBody().empty())
    return success();

  // All block arguments must have type !hir.any.
  auto anyType = AnyType::get(getContext());
  for (auto arg : getBody().front().getArguments())
    if (arg.getType() != anyType)
      return emitOpError() << "block argument must have type !hir.any, got "
                           << arg.getType();

  if (getArgNames().size() != getBody().front().getNumArguments())
    return emitOpError() << "argNames has " << getArgNames().size()
                         << " entries but body has "
                         << getBody().front().getNumArguments()
                         << " block arguments";

  if (auto returnOp = getReturnOp()) {
    if (getResultNames().size() != returnOp.getValues().size())
      return emitOpError() << "resultNames has " << getResultNames().size()
                           << " entries but return has "
                           << returnOp.getValues().size() << " values";
  }

  return success();
}

ReturnOp FuncOp::getReturnOp() {
  if (getBody().empty())
    return {};
  return dyn_cast<ReturnOp>(getBody().back().getTerminator());
}

void FuncOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  if (&region != &getBody() || region.empty())
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
//   hir.split_func @name(arg: phase, ...) -> (result: phase, ...) {
//     <signature region>
//   } [phase: @sym, ...]
//
// The parser extracts named+phased argument/result lists and a phase-to-symbol
// map, storing them as properties on the op.

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
  auto anyType = AnyType::get(ctx);
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

  // Print phase map, one entry per line.
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

  // Verify the signature terminator's operand counts match.
  auto sigOp = dyn_cast<SignatureOp>(getSignature().back().getTerminator());
  if (!sigOp)
    return success(); // verifyRegions will catch missing terminator

  if (sigOp.getTypeOfArgs().size() != getArgNames().size())
    return emitOpError() << "signature has " << sigOp.getTypeOfArgs().size()
                         << " argument types but function has "
                         << getArgNames().size() << " arguments";

  if (sigOp.getTypeOfResults().size() != getResultNames().size())
    return emitOpError() << "signature has " << sigOp.getTypeOfResults().size()
                         << " result types but function has "
                         << getResultNames().size() << " results";

  return success();
}

LogicalResult SplitFuncOp::verifyRegions() {
  if (!isa<SignatureOp>(getSignature().back().getTerminator()))
    return emitOpError()
           << "requires `hir.signature` terminator in the signature";
  return success();
}

bool SplitFuncOp::canDiscardOnUseEmpty() { return false; }

SignatureOp SplitFuncOp::getSignatureOp() {
  return cast<SignatureOp>(getSignature().back().getTerminator());
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
// MultiphaseFuncOp
//===----------------------------------------------------------------------===//

// # Custom Parser for MultiphaseFuncOp
//
// The assembly format is:
//   hir.multiphase_func @name(last arg, first arg, ...) -> (result, ...) [
//     @sym, ...
//   ]
//
// Each argument is preceded by `first` or `last` to indicate whether it
// originates from the first or last sub-phase.

ParseResult MultiphaseFuncOp::parse(OpAsmParser &parser,
                                    OperationState &result) {
  auto &props = result.getOrAddProperties<MultiphaseFuncOp::Properties>();
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

  // Parse argument list: ((first|last) name, ...).
  SmallVector<Attribute> argNames;
  SmallVector<bool> argIsFirst;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      StringRef keyword;
      std::string name;
      if (parser.parseKeyword(&keyword))
        return failure();
      if (keyword != "first" && keyword != "last")
        return parser.emitError(parser.getCurrentLocation(),
                                "expected 'first' or 'last'");
      if (parser.parseKeywordOrString(&name))
        return failure();
      argNames.push_back(builder.getStringAttr(name));
      argIsFirst.push_back(keyword == "first");
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.argNames = builder.getArrayAttr(argNames);
  props.argIsFirst = builder.getDenseBoolArrayAttr(argIsFirst);

  // Parse result list: -> (name, ...).
  SmallVector<Attribute> resultNames;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      if (parser.parseKeywordOrString(&name))
        return failure();
      resultNames.push_back(builder.getStringAttr(name));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.resultNames = builder.getArrayAttr(resultNames);

  // Parse function list: [@sym, ...].
  SmallVector<Attribute> phaseFuncs;
  if (parser.parseLSquare())
    return failure();
  if (failed(parser.parseOptionalRSquare())) {
    do {
      FlatSymbolRefAttr sym;
      if (parser.parseAttribute(sym))
        return failure();
      phaseFuncs.push_back(sym);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRSquare())
      return failure();
  }
  props.phaseFuncs = builder.getArrayAttr(phaseFuncs);

  return success();
}

void MultiphaseFuncOp::print(OpAsmPrinter &p) {
  p << ' ';
  printSymbolVisibility(p, *this, getSymVisibilityAttr());
  p.printSymbolName(getSymName());

  // Print argument list.
  p << '(';
  auto argNames = getArgNames();
  auto argIsFirst = getArgIsFirst();
  for (size_t i = 0, e = argNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << (argIsFirst[i] ? "first" : "last") << ' '
      << cast<StringAttr>(argNames[i]).getValue();
  }
  p << ')';

  // Print result list.
  p << " -> (";
  auto resultNames = getResultNames();
  for (size_t i = 0, e = resultNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(resultNames[i]).getValue();
  }
  p << ") [";

  // Print function list, one entry per line.
  auto phaseFuncs = getPhaseFuncs();
  for (size_t i = 0, e = phaseFuncs.size(); i < e; ++i) {
    if (i)
      p << ',';
    p.printNewline();
    p << "  ";
    p.printAttribute(phaseFuncs[i]);
  }
  p.printNewline();
  p << ']';
}

LogicalResult MultiphaseFuncOp::verify() {
  if (getArgIsFirst().size() != getArgNames().size())
    return emitOpError() << "argIsFirst has " << getArgIsFirst().size()
                         << " entries but function has " << getArgNames().size()
                         << " arguments";

  if (getPhaseFuncs().empty())
    return emitOpError() << "phaseFuncs must have at least one entry";

  return success();
}

//===----------------------------------------------------------------------===//
// CallOp
//===----------------------------------------------------------------------===//

/// Infer result types for `hir.call`: one `!hir.any` result per element of
/// `typeOfResults`, so the parser can reconstruct the result count without
/// needing an explicit `type($results)` directive in the assembly format.
LogicalResult
CallOp::inferReturnTypes(MLIRContext *ctx, std::optional<Location>,
                         ValueRange operands, DictionaryAttr attrs,
                         OpaqueProperties props, RegionRange,
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
    // type_of(unified_call(..., resultTypes)) -> resultTypes[resultNumber]
    if (auto callOp = dyn_cast<UnifiedCallOp>(result.getOwner()))
      if (result.getResultNumber() < callOp.getTypeOfResults().size())
        return callOp.getTypeOfResults()[result.getResultNumber()];
  }
  return {};
}

LogicalResult TypeOfOp::canonicalize(TypeOfOp op, PatternRewriter &rewriter) {
  // type_of(x) -> x's type operand, if extractable.
  if (auto type = getTypeOf(op.getInput())) {
    rewriter.replaceOp(op, type);
    return success();
  }
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

// # Custom Parser for UnifiedFuncOp
//
// The assembly format is:
//   hir.unified_func [visibility] @name(%arg: phase, ...) -> (result: phase,
//   ...)
//     { <signature> } { <body> }
//
// Argument names use `%` because they become SSA block arguments in both the
// signature and body regions. Result names are bare identifiers.

ParseResult UnifiedFuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<UnifiedFuncOp::Properties>();
  auto *ctx = parser.getContext();
  auto builder = OpBuilder(ctx);
  auto anyType = AnyType::get(ctx);

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

void UnifiedFuncOp::print(OpAsmPrinter &p) {
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
  p << ") ";

  // Print optional attributes, excluding properties we've already printed.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(),
      {getSymNameAttrName(), getSymVisibilityAttrName(), getArgNamesAttrName(),
       getArgPhasesAttrName(), getResultNamesAttrName(),
       getResultPhasesAttrName()});

  // Print signature and body regions without entry block arguments.
  p.printRegion(getSignature(), /*printEntryBlockArgs=*/false);
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult UnifiedFuncOp::verify() {
  // The function's argNames/argPhases and resultNames/resultPhases are
  // authoritative. Everything else is checked against them.
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

  // Check the signature terminator's operand counts against the function's
  // declared arg/result counts.
  auto sigOp =
      dyn_cast<UnifiedSignatureOp>(getSignature().back().getTerminator());
  if (!sigOp)
    return success();

  if (sigOp.getTypeOfArgs().size() != numArgs)
    return emitOpError() << "signature has " << sigOp.getTypeOfArgs().size()
                         << " argument types but function has " << numArgs
                         << " arguments";

  if (sigOp.getTypeOfResults().size() != numResults)
    return emitOpError() << "signature has " << sigOp.getTypeOfResults().size()
                         << " result types but function has " << numResults
                         << " results";

  return success();
}

void UnifiedFuncOp::getAsmBlockArgumentNames(Region &region,
                                             OpAsmSetValueNameFn setNameFn) {
  if ((&region != &getSignature() && &region != &getBody()) || region.empty())
    return;
  auto argNames = getArgNames();
  for (auto [name, arg] : llvm::zip(argNames, region.front().getArguments()))
    setNameFn(arg, cast<StringAttr>(name).getValue());
}

LogicalResult UnifiedFuncOp::verifyRegions() {
  // Make sure there are no signature terminators in the middle of the
  // signature region. Return terminators may appear in non-last blocks of the
  // body region due to explicit `return` expressions.
  for (auto *region : getRegions())
    for (auto &block : llvm::drop_end(*region))
      if (isa<UnifiedSignatureOp>(block.getTerminator()))
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

  // The callee may be a unified_func (before splitting) or a split_func
  // (after the callee has been split but the caller hasn't yet).
  unsigned expectedArgs, expectedResults;
  ArrayRef<int32_t> calleeArgPhases, calleeResultPhases;
  if (auto func = symbolTable.lookupNearestSymbolFrom<UnifiedFuncOp>(
          getOperation(), callee)) {
    auto sigOp = func.getSignatureOp();
    expectedArgs = sigOp.getTypeOfArgs().size();
    expectedResults = sigOp.getTypeOfResults().size();
    calleeArgPhases = func.getArgPhases();
    calleeResultPhases = func.getResultPhases();
  } else if (auto splitFunc = symbolTable.lookupNearestSymbolFrom<SplitFuncOp>(
                 getOperation(), callee)) {
    auto sigOp = splitFunc.getSignatureOp();
    expectedArgs = sigOp.getTypeOfArgs().size();
    expectedResults = sigOp.getTypeOfResults().size();
    calleeArgPhases = splitFunc.getArgPhases();
    calleeResultPhases = splitFunc.getResultPhases();
  } else {
    return emitOpError() << "callee " << callee
                         << " does not reference a valid `hir.unified_func` or "
                            "`hir.split_func`";
  }

  if (getArguments().size() != expectedArgs)
    return emitOpError() << "has " << getArguments().size()
                         << " arguments, but " << callee << " expects "
                         << expectedArgs;

  if (getResults().size() != expectedResults)
    return emitOpError() << "has " << getResults().size() << " results, but "
                         << callee << " expects " << expectedResults;

  if (getArgPhases().size() != getArguments().size())
    return emitOpError() << "argPhases has " << getArgPhases().size()
                         << " entries but call has " << getArguments().size()
                         << " arguments";

  if (getResultPhases().size() != getResults().size())
    return emitOpError() << "resultPhases has " << getResultPhases().size()
                         << " entries but call has " << getResults().size()
                         << " results";

  if (getArgPhases() != calleeArgPhases)
    return emitOpError() << "argPhases do not match callee " << callee;

  if (getResultPhases() != calleeResultPhases)
    return emitOpError() << "resultPhases do not match callee " << callee;

  if (getTypeOfArgs().size() != getArguments().size())
    return emitOpError() << "typeOfArgs has " << getTypeOfArgs().size()
                         << " entries but call has " << getArguments().size()
                         << " arguments";

  if (getTypeOfResults().size() != getResults().size())
    return emitOpError() << "typeOfResults has " << getTypeOfResults().size()
                         << " entries but call has " << getResults().size()
                         << " results";

  return success();
}

//===----------------------------------------------------------------------===//
// MIRConstantOp
//===----------------------------------------------------------------------===//

/// Fold to the constant attribute value.
OpFoldResult MIRConstantOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// OpaqueUnpackOp
//===----------------------------------------------------------------------===//

/// Canonicalize `opaque_unpack(mir_constant #mir.opaque<[e0, e1, ...]>)` by
/// replacing each unpack result with an individual `hir.mir_constant` for the
/// corresponding element.
LogicalResult OpaqueUnpackOp::canonicalize(OpaqueUnpackOp op,
                                           PatternRewriter &rewriter) {
  Value input = op.getInput();

  auto mirConst = input.getDefiningOp<MIRConstantOp>();
  if (!mirConst)
    return failure();

  auto opaqueAttr = dyn_cast<base::OpaqueAttr>(mirConst.getValue());
  if (!opaqueAttr)
    return failure();

  if (opaqueAttr.getElements().size() != op.getResults().size())
    return failure();

  SmallVector<Value> replacements;
  for (auto elem : opaqueAttr.getElements())
    replacements.push_back(
        MIRConstantOp::create(rewriter, op.getLoc(), cast<TypedAttr>(elem))
            .getResult());
  rewriter.replaceOp(op, replacements);
  return success();
}

//===----------------------------------------------------------------------===//
// Type Extraction Helpers
//===----------------------------------------------------------------------===//

Value hir::getTypeOf(Value value) {
  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};

  return TypeSwitch<Operation *, Value>(result.getOwner())
      .Case<AddOp, SubOp, MulOp, DivOp, ModOp, AndOp, OrOp, XorOp, ShlOp, ShrOp,
            EqOp, NeqOp, LtOp, GtOp, GeqOp, LeqOp>(
          [](auto op) { return op.getResultType(); })
      .Case<CoerceTypeOp>([](CoerceTypeOp op) { return op.getTypeOperand(); })
      .Case<LetOp>([](LetOp op) { return op.getType(); })
      .Case<CallOp>([&](CallOp op) {
        return op.getTypeOfResults()[result.getResultNumber()];
      })
      .Case<UnifiedCallOp>([&](UnifiedCallOp op) {
        return op.getTypeOfResults()[result.getResultNumber()];
      })
      .Case<IfOp>([&](IfOp op) -> Value {
        // The type of an if result is the type of the corresponding yield
        // value in the then branch. Both branches must agree.
        auto &thenBlock = op.getThenRegion().front();
        if (auto yieldOp = dyn_cast<YieldOp>(thenBlock.getTerminator()))
          if (result.getResultNumber() < yieldOp.getOperands().size())
            return getTypeOf(yieldOp.getOperands()[result.getResultNumber()]);
        return {};
      })
      .Default([](Operation *) { return Value(); });
}

Value hir::getOrCreateTypeOf(OpBuilder &builder, Location loc, Value value) {
  if (auto type = getTypeOf(value))
    return type;
  if (value.getDefiningOp<ConstantIntOp>())
    return IntTypeOp::create(builder, loc).getResult();
  if (value.getDefiningOp<ConstantUnitOp>())
    return UnitTypeOp::create(builder, loc).getResult();
  return TypeOfOp::create(builder, loc, value).getResult();
}
