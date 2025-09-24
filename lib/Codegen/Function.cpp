//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Ops.h"

using namespace silicon;
using namespace codegen;

LogicalResult Context::convertFnItem(ast::FnItem &item) {
  auto funcOp = funcs.at(&item);

  // Handle the function signature.
  {
    // Create the entry block in the signature region.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&funcOp.getSignature().emplaceBlock());

    // Handle the function arguments.
    SmallVector<Value, 4> argValues;
    auto guard2 = BindingsScope(bindings);
    for (auto *arg : item.args) {
      // Compute the type of the argument.
      auto type = withinExpr([&] { return convertType(*arg->type); });
      if (!type)
        return failure();

      // Create an op for the argument.
      auto argName = StringAttr::get(module.getContext(), arg->name);
      auto argOp =
          hir::UncheckedArgOp::create(builder, arg->loc, argName, type);
      bindings.insert(arg, argOp);
      argValues.push_back(argOp);
    }

    // Handle the function result.
    SmallVector<Value, 4> resultTypes;
    if (item.returnType) {
      auto type = withinExpr([&] { return convertType(*item.returnType); });
      if (!type)
        return failure();
      resultTypes.push_back(type);
    }

    // Create the signature terminator.
    hir::UncheckedSignatureOp::create(builder, item.loc, argValues,
                                      resultTypes);
  }

  // Handle the function body.
  {
    // Create the entry block in the body region.
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&funcOp.getBody().emplaceBlock());

    // Return the result of the function body.
    hir::UncheckedReturnOp::create(currentBuilder(), item.loc, ValueRange{});
  }

  return success();
}

void Context::increaseConstness() { assert(false && "deprecated"); }
void Context::decreaseConstness() { assert(false && "deprecated"); }
