//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/Dialect/HIR/HIROps.h"
#include "llvm/ADT/ScopeExit.h"

using namespace silicon;
using namespace codegen;

LogicalResult Context::convertFnItem(ast::FnItem &item) {
  auto funcOp = funcs.at(&item);

  // Ensure we will resume insertion after the function once we're done.
  builder.setInsertionPointAfter(funcOp);
  OpBuilder::InsertionGuard guard(builder);

  // Handle the function arguments.
  auto &sigBlock = funcOp.getSignature().emplaceBlock();
  builder.setInsertionPointToStart(&sigBlock);
  SmallVector<Value> args;
  {
    auto guard = BindingsScope(bindings);
    for (auto *arg : item.args) {
      auto type =
          withinConst(arg->loc, [&] { return convertType(*arg->type); });
      if (!type)
        return failure();
      auto argOp = hir::ArgOp::create(builder, arg->loc, arg->name, type);
      bindings.insert(arg, argOp);
      args.push_back(argOp);
    }
  }
  hir::ArgsOp::create(builder, item.loc, args);

  // Handle the function body.
  auto &bodyBlock = funcOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&bodyBlock);

  auto guard2 = llvm::make_scope_exit([&] { constContexts.clear(); });
  constContexts.push_back(ConstContext{builder, bodyBlock});
  currentConstness = 0;

  auto result = convertExpr(*item.body);
  if (!result)
    return failure();

  // Create terminators in all constness levels.
  for (unsigned idx = 1; idx < constContexts.size(); ++idx)
    hir::NextPhaseOp::create(constContexts[idx].builder, item.body->loc,
                             &constContexts[idx - 1].entry);

  // Return the result value.
  hir::ReturnOp::create(constContexts[0].builder, item.body->loc, result);
  return success();
}
