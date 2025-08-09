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
#include "llvm/ADT/ScopedHashTable.h"

namespace silicon {
namespace codegen {

struct Context {
  ModuleOp module;
  OpBuilder builder;
  SymbolTable symbolTable;

  /// The IR function generated for each AST function. This is populated before
  /// real code generation begins, such that we can always map from an AST
  /// function to an IR function.
  DenseMap<ast::FnItem *, hir::FuncOp> funcs;

  /// The SSA values generated for each binding. Identifiers use this to resolve
  /// their binding to a value in the IR they can return.
  using Bindings = llvm::ScopedHashTable<ast::Binding, Value>;
  using BindingsScope = Bindings::ScopeTy;
  Bindings bindings;

  Context(ModuleOp module);
  LogicalResult convertAST(AST &ast);
  LogicalResult convertFnItem(ast::FnItem &item);
  Value convertExpr(ast::Expr &expr);
  Value convertType(ast::Type &type);

  /// Call `fn` embedded within a `hir::ConstOp`.
  Value withinConst(Location loc, llvm::function_ref<Value()> fn);
};

} // namespace codegen
} // namespace silicon
