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

  // Find all constant regions between the identifier and where the value is
  // defined. We'll need to thread the value through these regions.
  SmallVector<hir::ConstOp> constOps;
  auto *region = cx.builder.getBlock()->getParent();
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
  return hir::BinaryOp::create(cx.builder, expr.loc, lhs, rhs);
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
