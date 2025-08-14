//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

/// Handle expression statements.
static LogicalResult convert(ast::ExprStmt &stmt, Context &cx) {
  if (cx.convertExpr(*stmt.expr))
    return success();
  return failure();
}

/// Handle let bindings.
static LogicalResult convert(ast::LetStmt &stmt, Context &cx) {
  // Convert the optional type of the binding.
  Value type;
  if (stmt.type)
    if (!(type = cx.convertType(*stmt.type)))
      return failure();

  // Convert the optional value of the binding.
  Value value;
  if (stmt.value)
    if (!(value = cx.convertExpr(*stmt.value)))
      return failure();

  // Add the value to the bindings table.
  if (!value)
    return emitBug(stmt.loc) << "let bindings without value not implemented";
  cx.bindings.insert(&stmt, value);
  return success();
}

/// Emit an error for unimplemented statements.
static LogicalResult convert(ast::Stmt &stmt, Context &) {
  emitBug(stmt.loc) << "unsupported statement kind `" << stmt.getTypeName()
                    << "`";
  return failure();
}

LogicalResult Context::convertStmt(ast::Stmt &stmt) {
  return ast::visit(stmt, [&](auto &stmt) { return convert(stmt, *this); });
}
