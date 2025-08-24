//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Ops.h"
#include "silicon/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

/// Handle the int type.
static Value convert(ast::IntType &type, Context &cx) {
  return hir::IntTypeOp::create(cx.currentBuilder(), type.loc);
}

/// Handle const types.
static Value convert(ast::ConstType &type, Context &cx) {
  cx.increaseConstness();
  auto result = cx.convertType(*type.type);
  cx.decreaseConstness();
  return result;
}

/// Handle the uint type.
static Value convert(ast::UIntType &type, Context &cx) {
  auto width = cx.convertExpr(*type.width);
  if (!width)
    return {};
  return hir::UIntTypeOp::create(cx.currentBuilder(), type.loc, width);
}

Value Context::convertType(ast::Type &type) {
  return ast::visit(type, [&](auto &type) { return convert(type, *this); });
}
