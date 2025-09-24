//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/HIR/Ops.h"
#include "silicon/Syntax/AST.h"
#include "llvm/ADT/ScopedHashTable.h"

namespace silicon {
namespace codegen {

struct ConstContext {
  OpBuilder builder;
  Block &entry;
  hir::FuncOp funcOp;
  hir::SpecializeFuncOp specializeOp;
  DenseMap<Value, Value> forwardedValues;
  unsigned lastArgumentIndex = 0;
};

struct Context {
  ModuleOp module;
  OpBuilder builder;
  SymbolTable symbolTable;

  /// The IR function generated for each AST function. This is populated before
  /// real code generation begins, such that we can always map from an AST
  /// function to an IR function.
  DenseMap<ast::FnItem *, hir::UncheckedFuncOp> funcs;

  /// The SSA values generated for each binding. Identifiers use this to resolve
  /// their binding to a value in the IR they can return.
  using Bindings = llvm::ScopedHashTable<ast::Binding, Value>;
  using BindingsScope = Bindings::ScopeTy;
  Bindings bindings;

  unsigned currentConstness;
  SmallVector<ConstContext, 0> constContexts;
  OpBuilder &currentBuilder() { return builder; }
  [[deprecated]]
  void increaseConstness();
  [[deprecated]]
  void decreaseConstness();

  /// Determine the constness level of a value.
  [[deprecated]]
  unsigned getValueConstness(Value value);

  Context(ModuleOp module);
  LogicalResult convertAST(AST &ast);
  LogicalResult convertFnItem(ast::FnItem &item);
  Value convertExpr(ast::Expr &expr);
  Value convertType(ast::Type &type);
  LogicalResult convertStmt(ast::Stmt &stmt);

  /// Execute a function with the builder positioned inside an expression
  /// wrapper op.
  Value withinExpr(llvm::function_ref<Value()> fn);
};

} // namespace codegen
} // namespace silicon
