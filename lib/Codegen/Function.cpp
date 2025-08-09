//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/Dialect/HIR/HIROps.h"

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
  hir::ArgsOp::create(builder, funcOp.getLoc());

  // Handle the function body.
  auto &bodyBlock = funcOp.getBody().emplaceBlock();
  builder.setInsertionPointToStart(&bodyBlock);
  if (!convertExpr(*item.body))
    return failure();
  hir::ReturnOp::create(builder, item.body->loc, {});

  return success();
}
