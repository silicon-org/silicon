//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/Dialect/HIR/HIRAttributes.h"
#include "silicon/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

/// Handle number literal expressions.
static Value convert(ast::NumLitExpr &expr, Context &cx) {
  return hir::ConstantIntOp::create(
      cx.currentBuilder(), expr.loc,
      hir::IntAttr::get(cx.module.getContext(), DynamicAPInt(expr.value)));
}

/// Handle identifier expressions.
static Value convert(ast::IdentExpr &expr, Context &cx) {
  auto value = cx.bindings.lookup(expr.binding);
  if (!value) {
    emitBug(expr.loc) << "no value generated for identifier";
    return {};
  }

  // Determine the constness level of the value. If the value is less constant
  // than the current context, emit an error.
  unsigned valueConstness = cx.getValueConstness(value);
  if (valueConstness < cx.currentConstness) {
    emitError(expr.loc) << "`" << expr.name << "` is not constant here";
    return {};
  }

  // If the value is more constant than the current context, we need to pass it
  // from one region to another until we reach the current context's region.
  while (valueConstness > cx.currentConstness) {
    auto &frozen = cx.constContexts[valueConstness].forwardedValues[value];
    if (!frozen) {
      cx.constContexts[valueConstness].returnOp.getFreezeMutable().append(
          value);
      frozen = cx.constContexts[valueConstness - 1].entry.addArgument(
          value.getType(), value.getLoc());
    }
    value = frozen;
    --valueConstness;
  }

  return value;
}

/// Handle binary expressions.
static Value convert(ast::BinaryExpr &expr, Context &cx) {
  auto lhs = cx.convertExpr(*expr.lhs);
  if (!lhs)
    return {};
  auto rhs = cx.convertExpr(*expr.rhs);
  if (!rhs)
    return {};
  return hir::BinaryOp::create(cx.currentBuilder(), expr.loc, lhs, rhs);
}

/// Handle block expressions.
static Value convert(ast::BlockExpr &block, Context &cx) {
  // Create a new scope for things like let bindings declared in this scope.
  auto guard = Context::BindingsScope(cx.bindings);

  // Handle the statements in the block.
  for (auto *stmt : block.stmts)
    if (failed(cx.convertStmt(*stmt)))
      return {};

  // Handle the optional result, or create a unit result `()` otherwise.
  if (block.result)
    return cx.convertExpr(*block.result);
  return hir::ConstantUnitOp::create(cx.currentBuilder(), block.loc);
}

/// Handle const expressions.
static Value convert(ast::ConstExpr &expr, Context &cx) {
  cx.increaseConstness();
  auto value = cx.convertExpr(*expr.value);
  cx.decreaseConstness();
  return value;
}

/// Emit an error for unimplemented expressions.
static Value convert(ast::Expr &expr, Context &) {
  emitBug(expr.loc) << "unsupported expression kind `" << expr.getTypeName()
                    << "`";
  return {};
}

Value Context::convertExpr(ast::Expr &expr) {
  return ast::visit(expr, [&](auto &expr) { return convert(expr, *this); });
}
