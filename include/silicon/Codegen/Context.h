//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Dialect/HIR/HIROps.h"
#include "silicon/Syntax/AST.h"

namespace silicon {
namespace codegen {

struct Context {
  ModuleOp module;
  OpBuilder builder;
  SymbolTable symbolTable;
  DenseMap<ast::FnItem *, hir::FuncOp> funcs;

  Context(ModuleOp module);
  LogicalResult convertAST(AST &ast);
  LogicalResult convertFnItem(ast::FnItem &item);
  Value convertExpr(ast::Expr &expr);
};

} // namespace codegen
} // namespace silicon
