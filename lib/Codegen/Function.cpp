//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

LogicalResult Context::convertFnItem(ast::FnItem &item) {
  auto funcOp = funcs.at(&item);

  // Handle the function signature.
  //
  // The signature region has a single block whose arguments carry the function
  // argument values. These block arguments can be referenced by dependent-type
  // expressions (e.g. `uint<b>`). We also store argument names as an attribute
  // on the func op so downstream passes can display meaningful names.
  SmallVector<Type> argTypes;
  SmallVector<Location> argLocs;
  argTypes.reserve(item.args.size());
  argLocs.reserve(item.args.size());
  {
    // Create the entry block in the signature region with one !hir.any block
    // argument per function argument.
    OpBuilder::InsertionGuard guard(builder);
    auto *ctx = module.getContext();
    auto &sigEntry = funcOp.getSignature().emplaceBlock();
    SmallVector<Type> anyTypes(item.args.size(), hir::AnyType::get(ctx));
    SmallVector<Location> sigArgLocs;
    for (auto *arg : item.args)
      sigArgLocs.push_back(arg->loc);
    sigEntry.addArguments(anyTypes, sigArgLocs);
    builder.setInsertionPointToStart(&sigEntry);

    // Bind each block argument so dependent-type expressions like `uint<b>`
    // can refer to the corresponding argument value.
    auto guard2 = BindingsScope(bindings);
    for (auto [arg, blockArg] : llvm::zip(item.args, sigEntry.getArguments())) {
      bindings.insert(arg, blockArg);
      argTypes.push_back(blockArg.getType());
      argLocs.push_back(blockArg.getLoc());
    }

    // Compute the type of each argument.
    SmallVector<Value, 4> typeSSAValues;
    for (auto *arg : item.args) {
      auto type = withinExpr([&] { return convertType(*arg->type); });
      if (!type)
        return failure();
      typeSSAValues.push_back(type);
    }

    // Compute the return type.
    SmallVector<Value, 4> resultTypes;
    if (item.returnType) {
      auto type = withinExpr([&] { return convertType(*item.returnType); });
      if (!type)
        return failure();
      resultTypes.push_back(type);
    } else {
      auto type = hir::UnitTypeOp::create(builder, item.loc);
      resultTypes.push_back(type);
    }

    // Emit the signature terminator with computed argument and result types.
    hir::UnifiedSignatureOp::create(builder, item.loc, typeSSAValues,
                                    resultTypes);
  }

  // Record argument names as an attribute on the func op.
  SmallVector<Attribute> argNameAttrs;
  for (auto *arg : item.args)
    argNameAttrs.push_back(StringAttr::get(module.getContext(), arg->name));
  funcOp.setArgNamesAttr(
      mlir::ArrayAttr::get(module.getContext(), argNameAttrs));

  // Handle the function body.
  {
    // Create the entry block in the body region.
    OpBuilder::InsertionGuard guard(builder);
    auto &entry = funcOp.getBody().emplaceBlock();
    builder.setInsertionPointToStart(&entry);

    // Add arguments to the entry block.
    auto guard2 = BindingsScope(bindings);
    entry.addArguments(argTypes, argLocs);
    for (auto [arg, value] : llvm::zip(item.args, entry.getArguments()))
      bindings.insert(arg, value);

    // Handle the function body.
    auto value = convertExpr(*item.body);
    if (!value)
      return failure();

    // Return the result of the function body.
    auto valueType = hir::getOrCreateTypeOf(builder, item.loc, value);
    hir::UnifiedReturnOp::create(builder, item.loc, ValueRange{value},
                                 ValueRange{valueType});
  }

  return success();
}
