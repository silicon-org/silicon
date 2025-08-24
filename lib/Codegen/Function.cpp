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

  // Create a const context for the main function body.
  constContexts.clear();
  currentConstness = -1;
  increaseConstness();

  // Handle the function arguments.
  auto guard2 = BindingsScope(bindings);
  for (auto *arg : item.args) {
    // Determine the type of the argument.
    increaseConstness();
    auto type = convertType(*arg->type);
    decreaseConstness();
    if (!type)
      return failure();

    // Add the argument type to the return op's list of arguments at the
    // argument's level of constness.
    unsigned argConstness = getValueConstness(type);
    constContexts[argConstness].returnOp.getArgsMutable().append(type);

    // Add the argument value as a block argument to the next lower level of
    // constness.
    auto &constCx = constContexts[argConstness - 1];
    auto blockArg = constCx.entry.insertArgument(
        constCx.lastArgumentIndex++, hir::getLowerKind(type.getType()),
        arg->loc);
    bindings.insert(arg, blockArg);
  }

  // Handle the function body.
  auto result = convertExpr(*item.body);
  if (!result)
    return failure();

  // Update the function op with the fully populated regions.
  auto newFuncOp = hir::FuncOp::create(
      builder, funcOp.getLoc(), funcOp.getSymNameAttr(), constContexts.size());
  for (unsigned idx = 0; idx < constContexts.size(); ++idx) {
    newFuncOp.getBodies()[idx].takeBody(
        *constContexts[constContexts.size() - idx - 1].entry.getParent());
  }
  funcs[&item] = newFuncOp;
  funcOp.erase();

  return success();
}
