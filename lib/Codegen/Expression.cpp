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
#include "silicon/Support/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

static Value freezeValueAcrossConstness(Value value, unsigned valueConstness,
                                        Context &cx) {
  auto &frozen = cx.constContexts[valueConstness].forwardedValues[value];
  if (!frozen) {
    cx.constContexts[valueConstness].specializeOp.getConstsMutable().append(
        value);
    frozen = cx.constContexts[valueConstness - 1].entry.addArgument(
        value.getType(), value.getLoc());
  }
  return frozen;
}

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
    value = freezeValueAcrossConstness(value, valueConstness, cx);
    --valueConstness;
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
  unsigned callConstness = cx.currentConstness;
  SmallVector<SmallVector<Value>> argsAtConstness;
  {
    auto guard = Context::BindingsScope(cx.bindings);
    for (auto [fnArg, callArg] : llvm::zip(fnItem->args, expr.args)) {
      // Determine the type of the argument.
      cx.increaseConstness();
      auto type = cx.convertType(*fnArg->type);
      cx.decreaseConstness();
      if (!type)
        return {};

      // Handle the argument passed from the call to the function.
      unsigned argConstness = cx.getValueConstness(type) - 1;
      while (cx.currentConstness < argConstness)
        cx.increaseConstness();
      auto argValue = cx.convertExpr(*callArg);
      cx.currentConstness = callConstness;
      if (!argValue)
        return {};

      // Keep track of the argument at its constness level.
      unsigned relativeConstness = argConstness - callConstness;
      if (argsAtConstness.size() <= relativeConstness)
        argsAtConstness.resize(relativeConstness + 1);
      argsAtConstness[relativeConstness].push_back(argValue);
    }
  }

  // Create the call ops at the different constness levels.
  cx.currentConstness = callConstness + argsAtConstness.size();
  auto funcName = StringAttr::get(cx.module.getContext(),
                                  cx.funcs.lookup(fnItem).getSymName() +
                                      ".const" + Twine(cx.currentConstness));
  auto funcType = hir::FuncType::get(cx.module.getContext());
  auto anyfuncTypeOp =
      hir::AnyfuncTypeOp::create(cx.currentBuilder(), expr.loc);
  auto calleeType = hir::FuncTypeOp::create(
      cx.currentBuilder(), expr.loc, ValueRange{}, ValueRange{anyfuncTypeOp});
  auto firstCallee = hir::ConstantFuncOp::create(cx.currentBuilder(), expr.loc,
                                                 funcName, calleeType);
  auto prevCall =
      hir::CallOp::create(cx.currentBuilder(), expr.loc, TypeRange{funcType},
                          firstCallee, ValueRange{});

  for (unsigned idx = 0; idx < argsAtConstness.size(); ++idx) {
    auto callee = freezeValueAcrossConstness(prevCall.getResults()[0],
                                             cx.currentConstness, cx);
    unsigned revIdx = argsAtConstness.size() - idx - 1;
    cx.currentConstness = callConstness + revIdx;
    prevCall =
        hir::CallOp::create(cx.currentBuilder(), expr.loc,
                            revIdx > 0 ? TypeRange{funcType} : TypeRange{},
                            callee, argsAtConstness[revIdx]);
  }

  cx.currentConstness = callConstness;
  return hir::ConstantUnitOp::create(cx.currentBuilder(), expr.loc);
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
