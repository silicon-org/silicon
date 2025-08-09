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
      cx.builder, expr.loc,
      hir::IntAttr::get(cx.builder.getContext(), DynamicAPInt(expr.value)));
}

/// Handle identifier expressions.
static Value convert(ast::IdentExpr &expr, Context &cx) {
  auto value = cx.bindings.lookup(expr.binding);
  if (!value) {
    emitBug(expr.loc) << "no value generated for identifier";
    return {};
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
  return hir::BinaryOp::create(cx.builder, expr.loc,
                               hir::TypeType::get(cx.builder.getContext()), lhs,
                               rhs);
}

/// Handle block expressions.
static Value convert(ast::BlockExpr &block, Context &cx) {
  if (!block.stmts.empty()) {
    emitBug(block.stmts[0]->loc) << "blocks with statements not implemented";
    return {};
  }
  if (block.result)
    return cx.convertExpr(*block.result);
  return hir::ConstantUnitOp::create(cx.builder, block.loc);
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
