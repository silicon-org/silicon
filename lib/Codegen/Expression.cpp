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
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"

using namespace silicon;
using namespace codegen;

/// Handle boolean literal expressions.
static Value convert(ast::BoolLitExpr &expr, Context &cx) {
  return hir::ConstantBoolOp::create(
      cx.builder, expr.loc,
      base::BoolAttr::get(cx.module.getContext(), expr.value));
}

/// Handle number literal expressions. The type operand starts as an
/// inferrable, allowing context (e.g., assignment to `uint<8>`) to constrain
/// the integer literal's type during type inference.
static Value convert(ast::NumLitExpr &expr, Context &cx) {
  auto typeOperand =
      hir::InferrableOp::create(cx.builder, expr.loc).getResult();
  return hir::ConstantIntOp::create(
      cx.builder, expr.loc,
      base::IntAttr::get(cx.module.getContext(), DynamicAPInt(expr.value)),
      typeOperand);
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

/// Handle unary expressions by lowering to equivalent binary ops.
static Value convert(ast::UnaryExpr &expr, Context &cx) {
  auto arg = cx.convertExpr(*expr.arg);
  if (!arg)
    return {};
  auto argType = hir::getOrCreateTypeOf(cx.builder, expr.loc, arg);
  auto loc = expr.loc;

  switch (expr.op) {
  case ast::UnaryOp::Neg: {
    // Lower `-x` to `0 - x`. The constant inherits the argument's type.
    auto zero = hir::ConstantIntOp::create(
        cx.builder, loc,
        base::IntAttr::get(cx.module.getContext(), DynamicAPInt(0)), argType);
    return hir::SubOp::create(cx.builder, loc, zero, arg, argType);
  }
  case ast::UnaryOp::Not: {
    // Lower `!x` to `x ^ -1` (bitwise NOT). The constant inherits the
    // argument's type.
    auto allOnes = hir::ConstantIntOp::create(
        cx.builder, loc,
        base::IntAttr::get(cx.module.getContext(), DynamicAPInt(-1)), argType);
    return hir::XorOp::create(cx.builder, loc, arg, allOnes, argType);
  }
  default:
    emitBug(expr.loc) << "codegen for unary operator `" << toString(expr.op)
                      << "` not implemented";
    return {};
  }
}

/// Desugar a short-circuiting logical operator into CFG branches.
/// `a && b` → `if a { b } else { false }`
/// `a || b` → `if a { true } else { b }`
static Value convertLogicalOp(ast::BinaryExpr &expr, Context &cx, bool isAnd) {
  auto lhs = cx.convertExpr(*expr.lhs);
  if (!lhs)
    return {};

  auto anyType = hir::AnyType::get(cx.module.getContext());

  // Unify the lhs condition's type with bool. The unify result is used as the
  // type operand of the coerce_type, ensuring it survives canonicalization
  // and that CheckTypes catches type mismatches.
  auto lhsType = hir::getOrCreateTypeOf(cx.builder, expr.loc, lhs);
  auto lhsBoolType = hir::BoolTypeOp::create(cx.builder, expr.loc).getResult();
  Value lhsUnifiedType = cx.builder.createOrFold<hir::UnifyOp>(
      expr.loc, anyType, lhsType, lhsBoolType);
  lhs = hir::CoerceTypeOp::create(cx.builder, expr.loc, lhs, lhsUnifiedType);

  // Convert condition to i1.
  auto i1Cond = hir::CoerceToI1Op::create(cx.builder, expr.loc, lhs);

  // Create the then, else, and merge blocks. The merge block is added to the
  // region last, after building the sub-expressions, to ensure it remains the
  // final block even when nested logical ops add their own blocks.
  auto *currentBlock = cx.builder.getInsertionBlock();
  auto *region = currentBlock->getParent();
  auto *thenBlock = new Block();
  auto *elseBlock = new Block();
  auto *mergeBlock = new Block();
  mergeBlock->addArgument(anyType, expr.loc);
  region->push_back(thenBlock);
  region->push_back(elseBlock);

  // Terminate the current block with a conditional branch.
  mlir::cf::CondBranchOp::create(cx.builder, expr.loc, i1Cond, thenBlock,
                                 elseBlock);

  // Build the then block.
  cx.builder.setInsertionPointToStart(thenBlock);
  Value thenValue;
  if (isAnd) {
    thenValue = cx.convertExpr(*expr.rhs);
  } else {
    thenValue = hir::ConstantBoolOp::create(
        cx.builder, expr.loc,
        base::BoolAttr::get(cx.module.getContext(), true));
  }
  if (!thenValue)
    return {};
  mlir::cf::BranchOp::create(cx.builder, expr.loc, mergeBlock,
                             ValueRange{thenValue});

  // Build the else block.
  cx.builder.setInsertionPointToStart(elseBlock);
  Value elseValue;
  if (isAnd) {
    elseValue = hir::ConstantBoolOp::create(
        cx.builder, expr.loc,
        base::BoolAttr::get(cx.module.getContext(), false));
  } else {
    elseValue = cx.convertExpr(*expr.rhs);
  }
  if (!elseValue)
    return {};
  mlir::cf::BranchOp::create(cx.builder, expr.loc, mergeBlock,
                             ValueRange{elseValue});

  // Add the merge block last and continue there.
  region->push_back(mergeBlock);
  cx.builder.setInsertionPointToStart(mergeBlock);
  return mergeBlock->getArgument(0);
}

/// Handle binary expressions by dispatching to the appropriate HIR op.
static Value convert(ast::BinaryExpr &expr, Context &cx) {
  // Short-circuiting logical operators are desugared into if/else.
  if (expr.op == ast::BinaryOp::LogicalAnd)
    return convertLogicalOp(expr, cx, /*isAnd=*/true);
  if (expr.op == ast::BinaryOp::LogicalOr)
    return convertLogicalOp(expr, cx, /*isAnd=*/false);

  auto lhs = cx.convertExpr(*expr.lhs);
  if (!lhs)
    return {};
  auto rhs = cx.convertExpr(*expr.rhs);
  if (!rhs)
    return {};
  auto lhsType = hir::getOrCreateTypeOf(cx.builder, expr.loc, lhs);
  auto rhsType = hir::getOrCreateTypeOf(cx.builder, expr.loc, rhs);
  Value operandType = cx.builder.createOrFold<hir::UnifyOp>(
      expr.loc, hir::AnyType::get(cx.module.getContext()), lhsType, rhsType);

  auto loc = expr.loc;
  switch (expr.op) {
  case ast::BinaryOp::Add:
    return hir::AddOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Sub:
    return hir::SubOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Mul:
    return hir::MulOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Div:
    return hir::DivOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Mod:
    return hir::ModOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::And:
    return hir::AndOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Or:
    return hir::OrOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Xor:
    return hir::XorOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Shl:
    return hir::ShlOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Shr:
    return hir::ShrOp::create(cx.builder, loc, lhs, rhs, operandType);
  case ast::BinaryOp::Eq:
  case ast::BinaryOp::Neq:
  case ast::BinaryOp::Lt:
  case ast::BinaryOp::Gt:
  case ast::BinaryOp::Geq:
  case ast::BinaryOp::Leq: {
    // Comparisons return bool, not the operand type.
    Value boolType = hir::BoolTypeOp::create(cx.builder, loc);
    switch (expr.op) {
    case ast::BinaryOp::Eq:
      return hir::EqOp::create(cx.builder, loc, lhs, rhs, boolType);
    case ast::BinaryOp::Neq:
      return hir::NeqOp::create(cx.builder, loc, lhs, rhs, boolType);
    case ast::BinaryOp::Lt:
      return hir::LtOp::create(cx.builder, loc, lhs, rhs, boolType);
    case ast::BinaryOp::Gt:
      return hir::GtOp::create(cx.builder, loc, lhs, rhs, boolType);
    case ast::BinaryOp::Geq:
      return hir::GeqOp::create(cx.builder, loc, lhs, rhs, boolType);
    case ast::BinaryOp::Leq:
      return hir::LeqOp::create(cx.builder, loc, lhs, rhs, boolType);
    default:
      llvm_unreachable("unreachable");
    }
  }
  case ast::BinaryOp::LogicalAnd:
  case ast::BinaryOp::LogicalOr:
    llvm_unreachable("handled above");
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

/// Handle if/else expressions by emitting CFG blocks with `cf.cond_br` and
/// `cf.br`. The condition is unified with `bool` and cast to `i1` via
/// `hir.coerce_to_i1`. A merge block with a block argument carries the
/// result value to subsequent code.
static Value convert(ast::IfExpr &expr, Context &cx) {
  auto condition = cx.convertExpr(*expr.condition);
  if (!condition)
    return {};

  auto anyType = hir::AnyType::get(cx.module.getContext());

  // Unify the condition's type with bool. The unify result is used as the
  // type operand of the coerce_type, ensuring it survives canonicalization
  // and that CheckTypes catches type mismatches.
  auto condType = hir::getOrCreateTypeOf(cx.builder, expr.loc, condition);
  auto boolType = hir::BoolTypeOp::create(cx.builder, expr.loc).getResult();
  Value unifiedType = cx.builder.createOrFold<hir::UnifyOp>(expr.loc, anyType,
                                                            condType, boolType);
  condition =
      hir::CoerceTypeOp::create(cx.builder, expr.loc, condition, unifiedType);

  // Convert condition to i1 for the branch.
  auto i1Cond = hir::CoerceToI1Op::create(cx.builder, expr.loc, condition);

  // Create the then, else, and merge blocks. The merge block is added to the
  // region last, after building the sub-expressions, to ensure it remains the
  // final block even when nested if/logical ops add their own blocks.
  auto *currentBlock = cx.builder.getInsertionBlock();
  auto *region = currentBlock->getParent();
  auto *thenBlock = new Block();
  auto *elseBlock = new Block();
  auto *mergeBlock = new Block();
  mergeBlock->addArgument(anyType, expr.loc);
  region->push_back(thenBlock);
  region->push_back(elseBlock);

  // Terminate the current block with a conditional branch.
  mlir::cf::CondBranchOp::create(cx.builder, expr.loc, i1Cond, thenBlock,
                                 elseBlock);

  // Build the then block.
  cx.builder.setInsertionPointToStart(thenBlock);
  auto thenValue = cx.convertExpr(*expr.thenExpr);
  if (!thenValue)
    return {};
  mlir::cf::BranchOp::create(cx.builder, expr.loc, mergeBlock,
                             ValueRange{thenValue});

  // Build the else block.
  cx.builder.setInsertionPointToStart(elseBlock);
  Value elseValue;
  if (expr.elseExpr) {
    elseValue = cx.convertExpr(*expr.elseExpr);
    if (!elseValue)
      return {};
  } else {
    elseValue = hir::ConstantUnitOp::create(cx.builder, expr.loc);
  }
  mlir::cf::BranchOp::create(cx.builder, expr.loc, mergeBlock,
                             ValueRange{elseValue});

  // Add the merge block last and continue there.
  region->push_back(mergeBlock);
  cx.builder.setInsertionPointToStart(mergeBlock);
  return mergeBlock->getArgument(0);
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

/// Handle return expressions. Emits a `hir.return` terminator for the current
/// block and creates a new unreachable block to absorb any subsequent code.
/// The new block's argument serves as a placeholder value.
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
  hir::ReturnOp::create(cx.builder, expr.loc, ValueRange{value},
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
