//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/Dialect/HIR/HIRAttributes.h"
#include "silicon/Dialect/HIR/HIRTypes.h"
#include "silicon/MLIR.h"
#include "silicon/Syntax/AST.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

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

  // Find all constant regions between the identifier and where the value is
  // defined. We'll need to thread the value through these regions.
  SmallVector<hir::ConstOp> constOps;
  auto *region = cx.currentBuilder().getBlock()->getParent();
  auto *targetRegion = value.getParentRegion();
  while (region != targetRegion) {
    if (auto constOp = dyn_cast<hir::ConstOp>(region->getParentOp()))
      constOps.push_back(constOp);
    region = region->getParentRegion();
  }

  // For each layer of constness, feed the value into the const op as an
  // argument, and add the unpacked value as a block argument to the const body.
  for (auto constOp : llvm::reverse(constOps)) {
    auto constType = dyn_cast<hir::ConstType>(value.getType());
    if (!constType) {
      emitError(expr.loc) << "`" << expr.name << "` is not constant here";
      return {};
    }
    auto type = constType.getInnerType();
    constOp.getOperandsMutable().append(value);
    value = constOp.getBody().addArgument(type, expr.loc);
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
  // Increase the current constness level.
  ++cx.currentConstness;
  auto guard = llvm::make_scope_exit([&] { --cx.currentConstness; });

  // If this is the first time we reach this constness level, create the
  // corresponding block and builder.
  assert(cx.currentConstness <= cx.constContexts.size());
  if (cx.currentConstness == cx.constContexts.size()) {
    auto &region = *cx.constContexts.back().entry.getParent();
    region.push_front(new Block);
    cx.constContexts.push_back(
        ConstContext{OpBuilder::atBlockBegin(&region.front()), region.front()});
  }

  // Handle the expression itself.
  return cx.convertExpr(*expr.value);
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
