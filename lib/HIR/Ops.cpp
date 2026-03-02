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

  // Parse argument list: (name: phase, ...).
  SmallVector<Attribute> argNames;
  SmallVector<int32_t> argPhases;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      int32_t phase;
      if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
          parser.parseInteger(phase))
        return failure();
      argNames.push_back(builder.getStringAttr(name));
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

  // Parse signature region.
  auto *region = result.addRegion();
  if (parser.parseRegion(*region))
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

  // Print argument list.
  p << '(';
  auto argNames = getArgNames();
  auto argPhases = getArgPhases();
  for (size_t i = 0, e = argNames.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(argNames[i]).getValue() << ": " << argPhases[i];
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

  // Print signature region.
  p.printRegion(getSignature());

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

SignatureOp SplitFuncOp::getSignatureOp() {
  return cast<SignatureOp>(getSignature().back().getTerminator());
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
  auto func = symbolTable.lookupNearestSymbolFrom<UnifiedFuncOp>(getOperation(),
                                                                 callee);
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
