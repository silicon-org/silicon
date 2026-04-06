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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
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

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/HIR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// FuncOp
//===----------------------------------------------------------------------===//

// # Custom Parser for FuncOp
//
// The assembly format is:
//   hir.func [visibility] @name(%arg, ...) -> (result, ...)
//     { <signature> } { <body> }
//
// Argument names use `%` because they become SSA block arguments in both the
// signature and body regions. All block args have type `!hir.any`. Result
// names are bare identifiers.

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
  p << ")";

  // Print optional attributes, excluding properties we've already printed.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getSymVisibilityAttrName(),
                            getArgNamesAttrName(), getResultNamesAttrName()});

  // Print signature and body regions without entry block arguments.
  p << ' ';
  p.printRegion(getSignature(), /*printEntryBlockArgs=*/false);
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  auto numArgs = getArgNames().size();

  // All block arguments must have type !hir.any.
  auto anyType = AnyType::get(getContext());
  for (auto arg : getBody().front().getArguments())
    if (arg.getType() != anyType)
      return emitOpError() << "block argument must have type !hir.any, got "
                           << arg.getType();

  // Check signature entry block arg count.
  if (getSignature().front().getNumArguments() != numArgs)
    return emitOpError() << "signature region has "
                         << getSignature().front().getNumArguments()
                         << " block arguments but function has " << numArgs
                         << " arguments";

  // Check body entry block arg count.
  if (getBody().front().getNumArguments() != numArgs)
    return emitOpError() << "argNames has " << numArgs
                         << " entries but body has "
                         << getBody().front().getNumArguments()
                         << " block arguments";

  return success();
}

LogicalResult FuncOp::verifyRegions() {
  // Ensure no SignatureOp in the body region.
  for (auto &block : getBody())
    if (isa<SignatureOp>(block.getTerminator()))
      return block.getTerminator()->emitOpError()
             << "cannot appear in the body";

  // Require SignatureOp terminator in the signature region's last block.
  if (!isa<SignatureOp>(getSignature().back().getTerminator()))
    return emitOpError()
           << "requires `hir.signature` terminator in the signature";

  // Check all block args in signature entry block have type !hir.any.
  auto anyType = AnyType::get(getContext());
  for (auto arg : getSignature().front().getArguments())
    if (arg.getType() != anyType)
      return emitOpError()
             << "signature block argument must have type !hir.any, got "
             << arg.getType();

  return success();
}

//===----------------------------------------------------------------------===//
// consolidateSignatureTerminators
//===----------------------------------------------------------------------===//

void hir::consolidateSignatureTerminators(Region &sig) {
  SmallVector<SignatureOp> sigTerminators;
  for (auto &block : sig)
    if (auto sigOp = dyn_cast<SignatureOp>(block.getTerminator()))
      sigTerminators.push_back(sigOp);
  if (sigTerminators.size() <= 1)
    return;

  auto firstSigOp = sigTerminators.front();
  unsigned numArgTypes = firstSigOp.getTypeOfArgs().size();
  unsigned numResultTypes = firstSigOp.getTypeOfResults().size();

  // Create the exit block with one block arg per typeOfArgs + typeOfResults.
  auto *exitBlock = new Block();
  sig.push_back(exitBlock);
  SmallVector<Type> blockArgTypes;
  SmallVector<Location> blockArgLocs;
  for (unsigned i = 0; i < numArgTypes; ++i) {
    blockArgTypes.push_back(firstSigOp.getTypeOfArgs()[i].getType());
    blockArgLocs.push_back(firstSigOp.getTypeOfArgs()[i].getLoc());
  }
  for (unsigned i = 0; i < numResultTypes; ++i) {
    blockArgTypes.push_back(firstSigOp.getTypeOfResults()[i].getType());
    blockArgLocs.push_back(firstSigOp.getTypeOfResults()[i].getLoc());
  }
  exitBlock->addArguments(blockArgTypes, blockArgLocs);

  // Create the consolidated signature terminator in the exit block.
  OpBuilder exitBuilder(firstSigOp->getContext());
  exitBuilder.setInsertionPointToStart(exitBlock);
  auto exitArgTypes = exitBlock->getArguments().take_front(numArgTypes);
  auto exitResultTypes = exitBlock->getArguments().drop_front(numArgTypes);
  SignatureOp::create(exitBuilder, firstSigOp.getLoc(),
                      SmallVector<Value>(exitArgTypes),
                      SmallVector<Value>(exitResultTypes));

  // Replace each original terminator with a branch to the exit block.
  for (auto sigOp : sigTerminators) {
    OpBuilder builder(firstSigOp->getContext());
    builder.setInsertionPoint(sigOp);
    SmallVector<Value> branchArgs;
    branchArgs.append(sigOp.getTypeOfArgs().begin(),
                      sigOp.getTypeOfArgs().end());
    branchArgs.append(sigOp.getTypeOfResults().begin(),
                      sigOp.getTypeOfResults().end());
    cf::BranchOp::create(builder, sigOp.getLoc(), exitBlock, branchArgs);
    sigOp.erase();
  }
}

//===----------------------------------------------------------------------===//
// SignatureOp
//===----------------------------------------------------------------------===//

/// Verify that the signature terminator's operand counts match the parent
/// function's declared arg/result counts. Works uniformly for FuncOp,
/// UnifiedFuncOp, and SplitFuncOp.
LogicalResult SignatureOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  unsigned numArgs, numResults;

  if (auto func = dyn_cast<FuncOp>(parentOp)) {
    numArgs = func.getArgNames().size();
    numResults = func.getResultNames().size();
  } else if (auto func = dyn_cast<UnifiedFuncOp>(parentOp)) {
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
// YieldOp
//===----------------------------------------------------------------------===//

LogicalResult YieldOp::verify() {
  auto *parentOp = (*this)->getParentOp();
  if (auto exprOp = dyn_cast<ExprOp>(parentOp)) {
    if (getOperands().size() != exprOp.getResults().size())
      return emitOpError() << "has " << getOperands().size()
                           << " operands but parent expr has "
                           << exprOp.getResults().size() << " results";
  } else if (auto replicateOp = dyn_cast<ReplicateOp>(parentOp)) {
    if (getOperands().size() != replicateOp.getResults().size())
      return emitOpError() << "has " << getOperands().size()
                           << " operands but parent replicate has "
                           << replicateOp.getResults().size() << " results";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Belt-and-suspenders check alongside the PredOpTrait.
  if (getTypeOfValues().size() != getValues().size())
    return emitOpError() << "has " << getValues().size() << " values but "
                         << getTypeOfValues().size() << " typeOfValues";

  auto *parentOp = (*this)->getParentOp();

  if (auto funcOp = dyn_cast<FuncOp>(parentOp)) {
    // For FuncOp parents, check values count matches resultNames.
    auto numResults = funcOp.getResultNames().size();
    if (getValues().size() != numResults)
      return emitOpError() << "has " << getValues().size()
                           << " values but parent function has " << numResults
                           << " results";

    // Triple correspondence: values <-> typeOfValues <-> resultNames.
    if (getTypeOfValues().size() != numResults)
      return emitOpError() << "has " << getTypeOfValues().size()
                           << " typeOfValues but parent function has "
                           << numResults << " results";
  }

  // UnifiedFuncOp parents have their return operands populated incrementally
  // by CheckCalls, so counts may not match until that pass completes.
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

  return success();
}

LogicalResult SplitFuncOp::verifyRegions() {
  if (!isa<SignatureOp>(getSignature().back().getTerminator()))
    return emitOpError()
           << "requires `hir.signature` terminator in the signature";
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

/// Verify that each sub-function referenced by the multiphase_func exists and
/// has the expected number of arguments and results. The protocol is:
///
/// - First sub-func: args = `first`-marked args, results = 1 (ctx)
/// - Middle sub-funcs: args = 1 (ctx), results = 1 (ctx)
/// - Last sub-func: args = `last`-marked args + 1 (ctx), results = declared
///   results
/// - Single sub-func (N=1): args = all args, results = declared results
LogicalResult
MultiphaseFuncOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto phaseFuncs = getPhaseFuncs();
  unsigned n = phaseFuncs.size();

  unsigned firstArgs = llvm::count(getArgIsFirst(), true);
  unsigned lastArgs = getArgNames().size() - firstArgs;
  unsigned declaredResults = getResultNames().size();

  for (unsigned i = 0; i < n; ++i) {
    auto sym = cast<FlatSymbolRefAttr>(phaseFuncs[i]);
    auto func =
        symbolTable.lookupNearestSymbolFrom<FuncOp>(getOperation(), sym);
    if (!func)
      // The sub-function may have been lowered to MIR or evaluated already.
      continue;

    unsigned actualArgs = func.getBody().front().getNumArguments();
    unsigned actualResults = func.getResultNames().size();

    // Compute expected arg/result counts based on position.
    unsigned expectedArgs, expectedResults;
    if (n == 1) {
      expectedArgs = firstArgs + lastArgs;
      expectedResults = declaredResults;
    } else if (i == 0) {
      expectedArgs = firstArgs;
      expectedResults = 1; // ctx
    } else if (i == n - 1) {
      expectedArgs = lastArgs + 1; // last args + ctx
      expectedResults = declaredResults;
    } else {
      expectedArgs = 1;    // ctx
      expectedResults = 1; // ctx
    }

    if (actualArgs != expectedArgs)
      return emitOpError() << "sub-function " << sym << " has " << actualArgs
                           << " arguments, expected " << expectedArgs;

    if (actualResults != expectedResults)
      return emitOpError() << "sub-function " << sym << " has " << actualResults
                           << " results, expected " << expectedResults;
  }

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

/// Verify that the call's argument and result counts match the callee's
/// interface. The callee may be an `hir.func` (matched against block args and
/// return values) or an `hir.multiphase_func` (matched against declared args
/// and results).
LogicalResult CallOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  auto callee = getCalleeAttr();
  unsigned expectedArgs, expectedResults;

  if (auto func =
          symbolTable.lookupNearestSymbolFrom<FuncOp>(getOperation(), callee)) {
    expectedArgs = func.getBody().front().getNumArguments();
    expectedResults = func.getResultNames().size();
  } else if (auto mpFunc =
                 symbolTable.lookupNearestSymbolFrom<MultiphaseFuncOp>(
                     getOperation(), callee)) {
    expectedArgs = mpFunc.getArgNames().size();
    expectedResults = mpFunc.getResultNames().size();
  } else {
    // The callee may have been lowered to MIR or evaluated already. In that
    // case we can't check counts — just succeed.
    return success();
  }

  if (getArguments().size() != expectedArgs)
    return emitOpError() << "has " << getArguments().size()
                         << " arguments, but callee " << callee << " expects "
                         << expectedArgs;

  if (getResults().size() != expectedResults)
    return emitOpError() << "has " << getResults().size()
                         << " results, but callee " << callee << " expects "
                         << expectedResults;

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
    // Verify the type value dominates this op. getTypeOf can look through
    // IfOp into child regions, returning values that don't dominate us.
    auto *opBlock = op->getBlock();
    bool dominates = false;
    if (auto *defOp = type.getDefiningOp()) {
      dominates = defOp->getBlock() && opBlock &&
                  defOp->getBlock()->getParent() == opBlock->getParent();
    } else if (auto blockArg = dyn_cast<BlockArgument>(type)) {
      dominates = blockArg.getOwner()->getParent() == opBlock->getParent();
    }
    if (dominates) {
      rewriter.replaceOp(op, type);
      return success();
    }
  }
  // type_of(constant_bool) -> bool_type
  if (op.getInput().getDefiningOp<ConstantBoolOp>()) {
    rewriter.replaceOpWithNewOp<BoolTypeOp>(op);
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
// CoerceTypeOp
//===----------------------------------------------------------------------===//

/// Fold away identity coercions where the input already has the target type.
LogicalResult CoerceTypeOp::canonicalize(CoerceTypeOp op,
                                         PatternRewriter &rewriter) {
  Value inputType = getTypeOf(op.getInput());
  if (inputType && inputType == op.getTypeOperand()) {
    rewriter.replaceOp(op, op.getInput());
    return success();
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// UnifyOp
//===----------------------------------------------------------------------===//

/// Canonicalize `unify %a, %a` → `%a`. This must be a canonicalization
/// pattern rather than a fold, because UnifyOp is not `Pure` (it uses
/// `MemRead` to prevent DCE), and folds on non-pure ops do not erase the op.
LogicalResult UnifyOp::canonicalize(UnifyOp op, PatternRewriter &rewriter) {
  if (!llvm::all_equal(op.getOperands()))
    return failure();
  rewriter.replaceOp(op, op.getLhs());
  return success();
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
  // Make sure there are no signature terminators in the body region.
  for (auto &block : getBody())
    if (isa<SignatureOp>(block.getTerminator()))
      return block.getTerminator()->emitOpError()
             << "cannot appear in the body";

  // Check the signature terminator. The last block must be a
  // `signature` terminator, but non-last blocks may also have them
  // for multi-path type computation (these are consolidated by CheckCalls).
  if (!isa<SignatureOp>(getSignature().back().getTerminator()))
    return emitOpError() << "requires `hir.signature` terminator in "
                            "the signature";

  return success();
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
    expectedArgs = func.getArgNames().size();
    expectedResults = func.getResultNames().size();
    calleeArgPhases = func.getArgPhases();
    calleeResultPhases = func.getResultPhases();
  } else if (auto splitFunc = symbolTable.lookupNearestSymbolFrom<SplitFuncOp>(
                 getOperation(), callee)) {
    expectedArgs = splitFunc.getArgNames().size();
    expectedResults = splitFunc.getResultNames().size();
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
// ConstantBoolOp
//===----------------------------------------------------------------------===//

/// Fold to the constant attribute value.
OpFoldResult ConstantBoolOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// ConstantUnitOp
//===----------------------------------------------------------------------===//

/// Fold to a unit attribute.
OpFoldResult ConstantUnitOp::fold(FoldAdaptor) {
  return base::UnitAttr::get(getContext());
}

//===----------------------------------------------------------------------===//
// MIRConstantOp
//===----------------------------------------------------------------------===//

/// Fold to the constant attribute value.
OpFoldResult MIRConstantOp::fold(FoldAdaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// OpaquePackOp
//===----------------------------------------------------------------------===//

/// Fold `opaque_pack(opaque_unpack(x)#0, ..., opaque_unpack(x)#N)` into `x`
/// when all operands come from the same unpack and correspond to all of its
/// results in order.
LogicalResult OpaquePackOp::canonicalize(OpaquePackOp op,
                                         PatternRewriter &rewriter) {
  auto operands = op.getOperands();
  if (operands.empty())
    return failure();

  // All operands must be results of the same OpaqueUnpackOp.
  auto unpack = operands[0].getDefiningOp<OpaqueUnpackOp>();
  if (!unpack)
    return failure();

  // The number of pack operands must match the number of unpack results.
  if (operands.size() != unpack.getResults().size())
    return failure();

  // Each operand must correspond to the matching unpack result.
  for (auto [i, operand] : llvm::enumerate(operands))
    if (operand != unpack.getResult(i))
      return failure();

  rewriter.replaceOp(op, unpack.getInput());
  return success();
}

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
  // Handle block arguments by looking at predecessor branch operands. We
  // recursively resolve the type of the branch operand, but only accept the
  // result if it dominates the merge block. Types defined inside predecessor
  // branches don't dominate the merge block and must be skipped.
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    auto *block = blockArg.getOwner();
    unsigned argIdx = blockArg.getArgNumber();
    for (auto *pred : block->getPredecessors()) {
      auto *term = pred->getTerminator();
      Value operand;
      if (auto brOp = dyn_cast<mlir::cf::BranchOp>(term))
        operand = brOp.getDestOperands()[argIdx];
      else if (auto condBrOp = dyn_cast<mlir::cf::CondBranchOp>(term)) {
        if (condBrOp.getTrueDest() == block)
          operand = condBrOp.getTrueDestOperands()[argIdx];
        else
          operand = condBrOp.getFalseDestOperands()[argIdx];
      }
      if (!operand)
        continue;
      auto type = getTypeOf(operand);
      if (!type)
        continue;
      // Only accept if the type is not defined inside a predecessor block,
      // since values from predecessor branches don't dominate the merge
      // block.
      if (auto *defOp = type.getDefiningOp()) {
        bool inPred = llvm::any_of(block->getPredecessors(), [&](Block *p) {
          return defOp->getBlock() == p;
        });
        if (inPred)
          continue;
      }
      return type;
    }
    return {};
  }

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return {};

  return TypeSwitch<Operation *, Value>(result.getOwner())
      .Case<AddOp, SubOp, MulOp, DivOp, ModOp, AndOp, OrOp, XorOp, ShlOp, ShrOp,
            EqOp, NeqOp, LtOp, GtOp, GeqOp, LeqOp>(
          [](auto op) { return op.getResultType(); })
      .Case<ConstantIntOp>(
          [](ConstantIntOp op) -> Value { return op.getTypeOperand(); })
      .Case<CoerceTypeOp>([](CoerceTypeOp op) { return op.getTypeOperand(); })
      .Case<CallOp>([&](CallOp op) {
        return op.getTypeOfResults()[result.getResultNumber()];
      })
      .Case<UnifiedCallOp>([&](UnifiedCallOp op) {
        return op.getTypeOfResults()[result.getResultNumber()];
      })
      .Case<arith::SelectOp>(
          [](arith::SelectOp op) { return getTypeOf(op.getTrueValue()); })
      .Default([](Operation *) { return Value(); });
}

Value hir::getOrCreateTypeOf(OpBuilder &builder, Location loc, Value value) {
  if (auto type = getTypeOf(value)) {
    // If the type is an inferrable, return it directly. This lets context
    // (e.g., a callee's uint<8> parameter) constrain the type via
    // unification. InferTypes has fallbacks to assign default types to
    // unconstrained inferrables, and CheckTypes guards against mismatches.
    if (type.getDefiningOp<InferrableOp>())
      return type;

    // Verify the type value is accessible at the current insertion point.
    auto *insertBlock = builder.getInsertionBlock();
    if (auto *defOp = type.getDefiningOp()) {
      if (defOp->getBlock() && insertBlock &&
          defOp->getBlock()->getParent() == insertBlock->getParent())
        return type;
    } else if (auto blockArg = dyn_cast<BlockArgument>(type)) {
      if (blockArg.getOwner()->getParent() == insertBlock->getParent())
        return type;
    }
  }
  if (value.getDefiningOp<ConstantIntOp>())
    return IntTypeOp::create(builder, loc).getResult();
  if (value.getDefiningOp<ConstantBoolOp>())
    return BoolTypeOp::create(builder, loc).getResult();
  if (value.getDefiningOp<ConstantUnitOp>())
    return UnitTypeOp::create(builder, loc).getResult();
  return TypeOfOp::create(builder, loc, value).getResult();
}

//===----------------------------------------------------------------------===//
// ReplicateOp
//===----------------------------------------------------------------------===//

// Parse:
//   hir.replicate %hit in %hits, (%x = %x_init, %y = %y_init) { body }
//   hir.replicate %hit in %hits { body }
ParseResult ReplicateOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &builder = parser.getBuilder();
  auto anyTy = AnyType::get(builder.getContext());

  // Parse "%hit in %hits".
  OpAsmParser::Argument hitArg;
  hitArg.type = anyTy;
  OpAsmParser::UnresolvedOperand hitsOperand;
  if (parser.parseArgument(hitArg) || parser.parseKeyword("in") ||
      parser.parseOperand(hitsOperand) ||
      parser.resolveOperand(hitsOperand, anyTy, result.operands))
    return failure();

  // Parse optional threaded args: ", (%x = %x_init, ...)".
  SmallVector<OpAsmParser::Argument> threadedArgs;
  SmallVector<OpAsmParser::UnresolvedOperand> threadedInits;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseLParen())
      return failure();
    if (failed(parser.parseOptionalRParen())) {
      do {
        OpAsmParser::Argument arg;
        arg.type = anyTy;
        OpAsmParser::UnresolvedOperand init;
        if (parser.parseArgument(arg) || parser.parseEqual() ||
            parser.parseOperand(init))
          return failure();
        threadedArgs.push_back(arg);
        threadedInits.push_back(init);
      } while (succeeded(parser.parseOptionalComma()));
      if (parser.parseRParen())
        return failure();
    }
  }

  // Resolve threaded init operands.
  for (auto &init : threadedInits)
    if (parser.resolveOperand(init, anyTy, result.operands))
      return failure();

  // Results: one per threaded arg.
  result.addTypes(SmallVector<Type>(threadedArgs.size(), anyTy));

  // Parse body region. Block args are: hit, then threaded args.
  SmallVector<OpAsmParser::Argument> bodyArgs;
  bodyArgs.push_back(hitArg);
  bodyArgs.append(threadedArgs.begin(), threadedArgs.end());

  auto *body = result.addRegion();
  if (parser.parseRegion(*body, bodyArgs))
    return failure();

  return success();
}

void ReplicateOp::print(OpAsmPrinter &p) {
  auto &body = getBody().front();
  auto numThreaded = getThreadedInits().size();

  // Print "%hit in %hits".
  p << ' ';
  p.printOperand(body.getArgument(0));
  p << " in ";
  p.printOperand(getHits());

  // Print optional threaded args.
  if (numThreaded > 0) {
    p << ", (";
    for (unsigned i = 0; i < numThreaded; ++i) {
      if (i > 0)
        p << ", ";
      p.printOperand(body.getArgument(1 + i));
      p << " = ";
      p.printOperand(getThreadedInits()[i]);
    }
    p << ')';
  }

  // Print body.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult ReplicateOp::verify() {
  auto &body = getBody().front();
  unsigned expectedBlockArgs = 1 + getThreadedInits().size();
  if (body.getNumArguments() != expectedBlockArgs)
    return emitOpError() << "body has " << body.getNumArguments()
                         << " block arguments but expected "
                         << expectedBlockArgs << " (1 hit + "
                         << getThreadedInits().size() << " threaded)";
  if (getResults().size() != getThreadedInits().size())
    return emitOpError() << "has " << getResults().size() << " results but "
                         << getThreadedInits().size()
                         << " threaded init values";
  return success();
}

void ReplicateOp::getAsmBlockArgumentNames(Region &region,
                                           OpAsmSetValueNameFn setNameFn) {
  // Block arg names were set by the parser; nothing extra needed.
}
