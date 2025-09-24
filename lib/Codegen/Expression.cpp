//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Attributes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Types.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

/// Handle number literal expressions.
static Value convert(ast::NumLitExpr &expr, Context &cx) {
  return hir::ConstantIntOp::create(
      cx.builder, expr.loc,
      hir::IntAttr::get(cx.module.getContext(), DynamicAPInt(expr.value)));
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

/// Handle call expressions.
static Value convert(ast::CallExpr &expr, Context &cx) {
  // We only support calling identifiers.
  auto *identCallee = dyn_cast<ast::IdentExpr>(expr.callee);
  if (!identCallee) {
    emitBug(expr.loc) << "calls to `" << expr.callee->getTypeName()
                      << "` not implemented";
    return {};
  }

  // We only support calls to `FnItem`s for now.
  auto *fnItem = dyn_cast<ast::FnItem *>(identCallee->binding);
  if (!fnItem) {
    emitBug(expr.loc) << "calls to non-fns not implemented";
    return {};
  }

  // Check that we have the right number of arguments.
  if (fnItem->args.size() != expr.args.size()) {
    emitError(expr.loc) << "call to `" << fnItem->name << "` expects "
                        << fnItem->args.size() << " arguments, but got "
                        << expr.args.size();
    return {};
  }

  // Generate the arguments for the call.
  SmallVector<Value> argValues;
  argValues.reserve(expr.args.size());
  for (auto *arg : expr.args) {
    auto argValue = cx.withinExpr([&] { return cx.convertExpr(*arg); });
    if (!argValue)
      return {};
    argValues.push_back(argValue);
  }

  // Create the call op.
  // TODO: Figure out the return type kind and number of results.
  return hir::UncheckedCallOp::create(
             cx.builder, expr.loc, hir::ValueType::get(cx.module.getContext()),
             FlatSymbolRefAttr::get(cx.funcs.lookup(fnItem).getSymNameAttr()),
             argValues)
      .getResult(0);
}

/// Handle binary expressions.
static Value convert(ast::BinaryExpr &expr, Context &cx) {
  auto lhs = cx.convertExpr(*expr.lhs);
  if (!lhs)
    return {};
  auto rhs = cx.convertExpr(*expr.rhs);
  if (!rhs)
    return {};
  return hir::BinaryOp::create(cx.builder, expr.loc, lhs, rhs);
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
  return hir::ConstantUnitOp::create(cx.builder, block.loc);
}

/// Handle const expressions.
static Value convert(ast::ConstExpr &expr, Context &cx) {
  return cx.withinExpr([&] { return cx.convertExpr(*expr.value); });
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
