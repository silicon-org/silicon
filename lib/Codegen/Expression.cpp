//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/Codegen/Context.h"
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
      base::IntAttr::get(cx.module.getContext(), DynamicAPInt(expr.value)));
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

  // Derive argument types from the actual argument values, using concrete type
  // constructors where possible instead of `type_of` ops. Result types are left
  // as inferrable placeholders, since they depend on the callee signature and
  // will be resolved by the CheckCalls pass.
  auto calleeFuncOp = cx.funcs.lookup(fnItem);
  SmallVector<Value> typeOfArgs, typeOfResults;
  for (auto arg : argValues)
    typeOfArgs.push_back(hir::getOrCreateTypeOf(cx.builder, expr.loc, arg));
  // TODO: Figure out the return type kind and number of results.
  typeOfResults.push_back(
      hir::InferrableOp::create(cx.builder, expr.loc).getResult());

  return hir::UnifiedCallOp::create(
             cx.builder, expr.loc, hir::AnyType::get(cx.module.getContext()),
             FlatSymbolRefAttr::get(calleeFuncOp.getSymNameAttr()), argValues,
             typeOfArgs, typeOfResults, calleeFuncOp.getArgPhasesAttr(),
             calleeFuncOp.getResultPhasesAttr())
      .getResult(0);
}

/// Handle binary expressions by dispatching to the appropriate HIR op.
static Value convert(ast::BinaryExpr &expr, Context &cx) {
  auto lhs = cx.convertExpr(*expr.lhs);
  if (!lhs)
    return {};
  auto rhs = cx.convertExpr(*expr.rhs);
  if (!rhs)
    return {};
  auto lhsType = hir::getOrCreateTypeOf(cx.builder, expr.loc, lhs);
  auto rhsType = hir::getOrCreateTypeOf(cx.builder, expr.loc, rhs);
  Value resultType = cx.builder.createOrFold<hir::UnifyOp>(
      expr.loc, hir::AnyType::get(cx.module.getContext()), lhsType, rhsType);

  auto loc = expr.loc;
  switch (expr.op) {
  case ast::BinaryOp::Add:
    return hir::AddOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Sub:
    return hir::SubOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Mul:
    return hir::MulOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Div:
    return hir::DivOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Mod:
    return hir::ModOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::And:
    return hir::AndOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Or:
    return hir::OrOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Xor:
    return hir::XorOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Shl:
    return hir::ShlOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Shr:
    return hir::ShrOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Eq:
    return hir::EqOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Neq:
    return hir::NeqOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Lt:
    return hir::LtOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Gt:
    return hir::GtOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Geq:
    return hir::GeqOp::create(cx.builder, loc, lhs, rhs, resultType);
  case ast::BinaryOp::Leq:
    return hir::LeqOp::create(cx.builder, loc, lhs, rhs, resultType);
  }
  llvm_unreachable("unhandled binary op kind");
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

/// Handle if/else expressions. Create an `hir.if` op with then and else
/// regions. If the else branch is absent, a unit-yielding else block is
/// generated.
static Value convert(ast::IfExpr &expr, Context &cx) {
  auto condition = cx.convertExpr(*expr.condition);
  if (!condition)
    return {};

  // Create the hir.if op with a single !hir.any result.
  auto anyType = hir::AnyType::get(cx.module.getContext());
  auto ifOp =
      hir::IfOp::create(cx.builder, expr.loc, TypeRange{anyType}, condition);

  // Build the then region.
  {
    auto ip = cx.builder.saveInsertionPoint();
    cx.builder.setInsertionPointToStart(&ifOp.getThenRegion().emplaceBlock());
    auto thenValue = cx.convertExpr(*expr.thenExpr);
    if (!thenValue)
      return {};
    hir::YieldOp::create(cx.builder, expr.loc, ValueRange{thenValue});
    cx.builder.restoreInsertionPoint(ip);
  }

  // Build the else region.
  {
    auto ip = cx.builder.saveInsertionPoint();
    cx.builder.setInsertionPointToStart(&ifOp.getElseRegion().emplaceBlock());
    if (expr.elseExpr) {
      auto elseValue = cx.convertExpr(*expr.elseExpr);
      if (!elseValue)
        return {};
      hir::YieldOp::create(cx.builder, expr.loc, ValueRange{elseValue});
    } else {
      auto unitValue = hir::ConstantUnitOp::create(cx.builder, expr.loc);
      hir::YieldOp::create(cx.builder, expr.loc, ValueRange{unitValue});
    }
    cx.builder.restoreInsertionPoint(ip);
  }

  return ifOp.getResult(0);
}

/// Handle const expressions. The phase shift of -1 indicates that the
/// expression should be evaluated one phase earlier than its parent.
static Value convert(ast::ConstExpr &expr, Context &cx) {
  return cx.withinExpr([&] { return cx.convertExpr(*expr.value); },
                       /*phaseShift=*/-1);
}

/// Handle dyn expressions. The phase shift of +1 indicates that the
/// expression should be evaluated one phase later than its parent.
static Value convert(ast::DynExpr &expr, Context &cx) {
  return cx.withinExpr([&] { return cx.convertExpr(*expr.value); },
                       /*phaseShift=*/+1);
}

/// Handle return expressions. Emits a `hir.unified_return` terminator for the
/// current block and creates a new unreachable block to absorb any subsequent
/// code. The new block's argument serves as a placeholder value.
static Value convert(ast::ReturnExpr &expr, Context &cx) {
  // Convert the return value, or create a unit value if absent.
  Value value;
  if (expr.value) {
    value = cx.convertExpr(*expr.value);
    if (!value)
      return {};
  } else {
    value = hir::ConstantUnitOp::create(cx.builder, expr.loc);
  }

  // Emit the return op as a terminator for the current block.
  auto valueType = hir::getOrCreateTypeOf(cx.builder, expr.loc, value);
  hir::UnifiedReturnOp::create(cx.builder, expr.loc, ValueRange{value},
                               ValueRange{valueType});

  // Create a new unreachable block to catch any code after the return. The
  // block argument acts as a placeholder value for the enclosing expression.
  auto *block = cx.builder.getInsertionBlock();
  auto *region = block->getParent();
  auto *deadBlock = new Block();
  region->push_back(deadBlock);
  auto placeholder = deadBlock->addArgument(
      hir::AnyType::get(cx.module.getContext()), expr.loc);
  cx.builder.setInsertionPointToStart(deadBlock);

  return placeholder;
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
