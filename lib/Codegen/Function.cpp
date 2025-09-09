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
  funcOp.getBody().emplaceBlock();

  // Create a const context for the main function body.
  constContexts.clear();
  constContexts.push_back(ConstContext(funcOp));
  currentConstness = 0;

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
    constContexts[argConstness].specializeOp.getTypeOfArgsMutable().append(
        type);

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

  // Return the result of the function body.
  hir::ReturnOp::create(currentBuilder(), item.loc, result);
  return success();
}

ConstContext::ConstContext(hir::FuncOp funcOp)
    : builder(funcOp.getContext()), entry(funcOp.getBody().front()),
      funcOp(funcOp) {
  builder.setInsertionPointToStart(&entry);
}

void Context::increaseConstness() {
  ++currentConstness;
  assert(currentConstness <= constContexts.size());
  if (currentConstness == constContexts.size()) {
    auto baseFuncOp = constContexts.front().funcOp;
    auto prevFuncOp = constContexts.back().funcOp;
    OpBuilder builder(prevFuncOp);
    auto funcOp = hir::FuncOp::create(
        builder, baseFuncOp.getLoc(),
        builder.getStringAttr(baseFuncOp.getSymName() + ".const" +
                              Twine(currentConstness)));
    symbolTable.insert(funcOp);
    funcOp.getBody().emplaceBlock();
    ConstContext cx(funcOp);
    cx.specializeOp = hir::SpecializeFuncOp::create(
        cx.builder, baseFuncOp.getLoc(),
        FlatSymbolRefAttr::get(prevFuncOp.getSymNameAttr()), ValueRange{},
        ValueRange{}, ValueRange{});
    hir::ReturnOp::create(cx.builder, baseFuncOp.getLoc(),
                          ValueRange{cx.specializeOp});
    cx.builder.setInsertionPoint(cx.specializeOp);
    constContexts.push_back(std::move(cx));
  }
}

void Context::decreaseConstness() {
  assert(currentConstness > 0);
  --currentConstness;
}
