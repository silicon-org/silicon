//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/Base/Types.h"
#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/AsmParser.h"
#include "silicon/Support/MLIR.h"

using namespace mlir;
using namespace silicon;
using namespace base;
using namespace mir;

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/MIR/Ops.cpp.inc"

/// Print an abbreviation of a type that can be used in an assembly name.
static bool getTypeAbbrev(llvm::raw_ostream &os, Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<TypeType, IntType, UnitType, AnyfuncType>([&](auto type) {
        os << type.getMnemonic();
        return true;
      })
      .Case<UIntType>([&](auto type) {
        os << type.getMnemonic() << type.getWidth();
        return true;
      })
      .Default([](auto) { return false; });
}

/// Print an abbreviation of an attribute that can be used in an assembly name.
static bool getAttrAbbrev(llvm::raw_ostream &os, Attribute attr) {
  return TypeSwitch<Attribute, bool>(attr)
      .Case<IntAttr>([&](auto attr) {
        os << 'c' << attr.getValue();
        return true;
      })
      .Case<base::UnitAttr>([&](auto attr) {
        os << "unit";
        return true;
      })
      .Case<base::TypeAttr>(
          [&](auto attr) { return getTypeAbbrev(os, attr.getValue()); })
      .Default([](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

//===----------------------------------------------------------------------===//
// Binary Ops
//
// Each binary op folds when both operands are constant IntAttrs. Arithmetic
// ops use DynamicAPInt's native operators; bitwise ops convert to int64_t
// since DynamicAPInt lacks bitwise operator overloads.
//===----------------------------------------------------------------------===//

/// Helper to extract IntAttr from both operands of a binary op adaptor.
static std::optional<std::pair<llvm::DynamicAPInt, llvm::DynamicAPInt>>
getConstantIntOperands(Attribute lhs, Attribute rhs) {
  auto lhsAttr = dyn_cast_or_null<IntAttr>(lhs);
  auto rhsAttr = dyn_cast_or_null<IntAttr>(rhs);
  if (!lhsAttr || !rhsAttr)
    return std::nullopt;
  return std::make_pair(lhsAttr.getValue(), rhsAttr.getValue());
}

OpFoldResult AddOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return IntAttr::get(getContext(), vals->first + vals->second);
  return {};
}

OpFoldResult SubOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return IntAttr::get(getContext(), vals->first - vals->second);
  return {};
}

OpFoldResult MulOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return IntAttr::get(getContext(), vals->first * vals->second);
  return {};
}

OpFoldResult DivOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return IntAttr::get(getContext(), vals->first / vals->second);
  return {};
}

OpFoldResult ModOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return IntAttr::get(getContext(), mod(vals->first, vals->second));
  return {};
}

OpFoldResult AndOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs())) {
    int64_t result =
        static_cast<int64_t>(vals->first) & static_cast<int64_t>(vals->second);
    return IntAttr::get(getContext(), llvm::DynamicAPInt(result));
  }
  return {};
}

OpFoldResult OrOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs())) {
    int64_t result =
        static_cast<int64_t>(vals->first) | static_cast<int64_t>(vals->second);
    return IntAttr::get(getContext(), llvm::DynamicAPInt(result));
  }
  return {};
}

OpFoldResult XorOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs())) {
    int64_t result =
        static_cast<int64_t>(vals->first) ^ static_cast<int64_t>(vals->second);
    return IntAttr::get(getContext(), llvm::DynamicAPInt(result));
  }
  return {};
}

OpFoldResult ShlOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs())) {
    int64_t result = static_cast<int64_t>(vals->first)
                     << static_cast<int64_t>(vals->second);
    return IntAttr::get(getContext(), llvm::DynamicAPInt(result));
  }
  return {};
}

OpFoldResult ShrOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs())) {
    int64_t result =
        static_cast<int64_t>(vals->first) >> static_cast<int64_t>(vals->second);
    return IntAttr::get(getContext(), llvm::DynamicAPInt(result));
  }
  return {};
}

//===----------------------------------------------------------------------===//
// Comparison Ops
//===----------------------------------------------------------------------===//

OpFoldResult EqOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first == vals->second);
  return {};
}

OpFoldResult NeqOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first != vals->second);
  return {};
}

OpFoldResult LtOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first < vals->second);
  return {};
}

OpFoldResult GtOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first > vals->second);
  return {};
}

OpFoldResult LeqOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first <= vals->second);
  return {};
}

OpFoldResult GeqOp::fold(FoldAdaptor adaptor) {
  if (auto vals = getConstantIntOperands(adaptor.getLhs(), adaptor.getRhs()))
    return base::BoolAttr::get(getContext(), vals->first >= vals->second);
  return {};
}

//===----------------------------------------------------------------------===//
// BoolToI1Op
//===----------------------------------------------------------------------===//

OpFoldResult BoolToI1Op::fold(FoldAdaptor adaptor) {
  if (auto boolAttr = dyn_cast_or_null<base::BoolAttr>(adaptor.getInput()))
    return mlir::BoolAttr::get(getContext(), boolAttr.getValue());
  return {};
}

void ConstantOp::getAsmResultNames(
    llvm::function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream bufferStream(buffer);
  if (!getAttrAbbrev(bufferStream, getValue()))
    return;
  bufferStream << '_';
  if (!getTypeAbbrev(bufferStream, getType()))
    buffer.pop_back();
  setNameFn(getResult(), buffer);
}

//===----------------------------------------------------------------------===//
// FuncOp
//
// The assembly format is:
//   mir.func [visibility] @name(%arg: type, ...) -> (result: type, ...) {
//     <body>
//   }
//
// Argument names use `%` because they become SSA block arguments in the body.
// Result names are bare identifiers, paired with their materialized types.
//===----------------------------------------------------------------------===//

ParseResult FuncOp::parse(OpAsmParser &parser, OperationState &result) {
  auto &props = result.getOrAddProperties<FuncOp::Properties>();
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

  // Parse argument list: (%name: type, ...).
  SmallVector<OpAsmParser::Argument> args;
  SmallVector<Attribute> argNames;
  if (parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      auto &arg = args.emplace_back();
      if (parser.parseArgument(arg) || parser.parseColon() ||
          parser.parseType(arg.type))
        return failure();
      argNames.push_back(builder.getStringAttr(arg.ssaName.name.drop_front()));
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.argNames = builder.getArrayAttr(argNames);

  // Parse result list: -> (name: type, ...).
  SmallVector<Attribute> resultNames;
  SmallVector<Type> resultTypes;
  if (parser.parseArrow() || parser.parseLParen())
    return failure();
  if (failed(parser.parseOptionalRParen())) {
    do {
      std::string name;
      Type type;
      if (parser.parseKeywordOrString(&name) || parser.parseColon() ||
          parser.parseType(type))
        return failure();
      resultNames.push_back(builder.getStringAttr(name));
      resultTypes.push_back(type);
    } while (succeeded(parser.parseOptionalComma()));
    if (parser.parseRParen())
      return failure();
  }
  props.resultNames = builder.getArrayAttr(resultNames);

  // Build the function type from parsed arg types and result types.
  SmallVector<Type> argTypes;
  for (auto &arg : args)
    argTypes.push_back(arg.type);
  props.function_type =
      mlir::TypeAttr::get(FunctionType::get(ctx, argTypes, resultTypes));

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

  // Print argument list with types.
  p << '(';
  if (!getBody().empty()) {
    auto args = getBody().front().getArguments();
    for (size_t i = 0, e = args.size(); i < e; ++i) {
      if (i)
        p << ", ";
      p.printRegionArgument(args[i]);
    }
  }
  p << ')';

  // Print result list with names and types.
  p << " -> (";
  auto resultNames = getResultNames();
  auto resultTypes = getFunctionType().getResults();
  for (size_t i = 0, e = resultTypes.size(); i < e; ++i) {
    if (i)
      p << ", ";
    p << cast<StringAttr>(resultNames[i]).getValue() << ": ";
    p.printType(resultTypes[i]);
  }
  p << ")";

  // Print optional attributes, excluding properties we've already printed.
  p.printOptionalAttrDictWithKeyword(
      (*this)->getAttrs(), {getSymNameAttrName(), getSymVisibilityAttrName(),
                            getFunctionTypeAttrName(), getArgNamesAttrName(),
                            getResultNamesAttrName()});

  // Print body region without entry block arguments.
  p << ' ';
  p.printRegion(getBody(), /*printEntryBlockArgs=*/false);
}

LogicalResult FuncOp::verify() {
  auto funcType = getFunctionType();

  if (getBody().empty())
    return success();

  // Check block argument count matches function type.
  auto &entryBlock = getBody().front();
  if (entryBlock.getNumArguments() != funcType.getNumInputs())
    return emitOpError() << "expects " << funcType.getNumInputs()
                         << " arguments but block has "
                         << entryBlock.getNumArguments() << " arguments";

  // Check block argument types match function type.
  for (auto [i, arg] : llvm::enumerate(entryBlock.getArguments()))
    if (arg.getType() != funcType.getInput(i))
      return emitOpError() << "block argument #" << i << " has type "
                           << arg.getType() << " but function expects "
                           << funcType.getInput(i);

  // Check argNames count.
  if (getArgNames().size() != funcType.getNumInputs())
    return emitOpError() << "argNames has " << getArgNames().size()
                         << " entries but function has "
                         << funcType.getNumInputs() << " arguments";

  // Check resultNames count.
  if (getResultNames().size() != funcType.getNumResults())
    return emitOpError() << "resultNames has " << getResultNames().size()
                         << " entries but function has "
                         << funcType.getNumResults() << " results";

  return success();
}

//===----------------------------------------------------------------------===//
// ReturnOp
//===----------------------------------------------------------------------===//

LogicalResult ReturnOp::verify() {
  // Only check return types when the parent is a FuncOp. The return may also
  // appear inside an if-region or similar, where these checks don't apply.
  auto funcOp = dyn_cast<FuncOp>((*this)->getParentOp());
  if (!funcOp)
    return success();

  auto resultTypes = funcOp.getFunctionType().getResults();

  // Check operand count matches function result count.
  if (getNumOperands() != resultTypes.size())
    return emitOpError() << "has " << getNumOperands()
                         << " operands but enclosing function returns "
                         << resultTypes.size() << " results";

  // Check each operand type matches the corresponding function result type.
  for (auto [i, operand, expected] :
       llvm::enumerate(getOperands(), resultTypes))
    if (operand.getType() != expected)
      return emitOpError() << "operand #" << i << " has type "
                           << operand.getType() << " but enclosing function "
                           << "returns " << expected;

  return success();
}

ReturnOp FuncOp::getReturnOp() {
  if (getBody().empty())
    return {};
  return dyn_cast<ReturnOp>(getBody().back().getTerminator());
}

//===----------------------------------------------------------------------===//
// OpaqueUnpackOp
//===----------------------------------------------------------------------===//

/// Fold `mir.opaque_unpack(mir.opaque_pack(...))` into the pack's operands.
LogicalResult
MIROpaqueUnpackOp::fold(FoldAdaptor adaptor,
                        SmallVectorImpl<OpFoldResult> &foldResults) {
  auto packOp = getInput().getDefiningOp<MIROpaquePackOp>();
  if (!packOp)
    return failure();
  if (packOp.getOperands().size() != getResults().size())
    return failure();
  foldResults.append(packOp.getOperands().begin(), packOp.getOperands().end());
  return success();
}

void FuncOp::getAsmBlockArgumentNames(Region &region,
                                      OpAsmSetValueNameFn setNameFn) {
  if (&region != &getBody() || region.empty())
    return;
  auto argNames = getArgNames();
  for (auto [name, arg] : llvm::zip(argNames, region.front().getArguments()))
    setNameFn(arg, cast<StringAttr>(name).getValue());
}
