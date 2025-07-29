//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/LLVM.h"
#include "silicon/MLIR.h"
#include "llvm/Support/Allocator.h"

namespace silicon {
namespace ast {

struct BlockExpr;
struct Expr;
struct FnArg;
struct Item;
struct Stmt;
struct Type;

/// Operator precedences.
///
/// See https://en.cppreference.com/w/c/language/operator_precedence.
enum class Precedence {
  Min,
  Or,     // |
  Xor,    // ^
  And,    // &
  Eq,     // == !=
  Rel,    // < > <= >=
  Shift,  // << >>
  Add,    // + -
  Mul,    // * / %
  Prefix, // - ! & * (unary prefix operators)
  Suffix, // a.b a() a[]
  Max
};

/// A root node in the AST, corresponding to a parsed source file.
struct Root {
  ArrayRef<Item *> items;
};

//===----------------------------------------------------------------------===//
// Items
//===----------------------------------------------------------------------===//

/// The different kinds of items that can appear in the AST.
enum class ItemKind { Fn };

/// Base class for all top-level items.
struct Item {
  const ItemKind kind;
  Location loc;
};

/// A function declaration.
struct FnItem : public Item {
  StringRef name;
  ArrayRef<FnArg *> args;
  Type *returnType; // optional
  BlockExpr *body;
  static bool classof(const Item *item) { return item->kind == ItemKind::Fn; }
};

/// An argument of a function declaration.
struct FnArg {
  Location loc;
  StringRef name;
  Type *type;
};

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// The different kinds of types that can appear in the AST.
enum class TypeKind { Const, Int, UInt };

/// Base class for all types.
struct Type {
  const TypeKind kind;
  Location loc;
};

/// A constant type, which wraps another type.
struct ConstType : public Type {
  Type *type;
  static bool classof(const Type *type) {
    return type->kind == TypeKind::Const;
  }
};

/// An unsigned integer type with a specific width.
struct UIntType : public Type {
  Expr *width;
  static bool classof(const Type *type) { return type->kind == TypeKind::UInt; }
};

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// The different kinds of expressions that can appear in the AST.
enum class ExprKind {
  Ident,
  NumLit,
  Call,
  Unary,
  Binary,
  Block,
  If,
  Return,
  Index,
  Const,
};

/// Base class for all expressions.
struct Expr {
  const ExprKind kind;
  Location loc;
};

/// An identifier expression, which refers to something by name.
struct IdentExpr : public Expr {
  StringRef name;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Ident;
  }
};

/// A number literal expression.
struct NumLitExpr : public Expr {
  APInt value;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::NumLit;
  }
};

/// A call expression.
struct CallExpr : public Expr {
  Expr *callee;
  ArrayRef<Expr *> args;
  static bool classof(const Expr *expr) { return expr->kind == ExprKind::Call; }
};

/// All unary operators.
enum class UnaryOp {
#define AST_UNARY(NAME, TOKEN) NAME,
#include "silicon/Syntax/AST.def"
};

/// All binary operators.
enum class BinaryOp {
#define AST_BINARY(NAME, TOKEN, PREC) NAME,
#include "silicon/Syntax/AST.def"
};

/// Return the precedence of the given binary operator.
Precedence getPrecedence(BinaryOp op);

/// A unary operator expression.
struct UnaryExpr : public Expr {
  UnaryOp op;
  Expr *arg;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Unary;
  }
};

/// A binary operator expression.
struct BinaryExpr : public Expr {
  BinaryOp op;
  Expr *lhs;
  Expr *rhs;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Binary;
  }
};

/// A block expression, which is a sequence of statements and an optional final
/// expression that is used as the block's return value.
struct BlockExpr : public Expr {
  SmallVector<Stmt *> stmts;
  Expr *result; // optional
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Block;
  }
};

/// An if expression.
struct IfExpr : public Expr {
  Expr *condition;
  Expr *thenExpr;
  Expr *elseExpr; // optional
  static bool classof(const Expr *expr) { return expr->kind == ExprKind::If; }
};

/// A return expression.
struct ReturnExpr : public Expr {
  Expr *value; // optional
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Return;
  }
};

/// An index expression.
struct IndexExpr : public Expr {
  Expr *base;
  Expr *index;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Index;
  }
};

/// A constant expression.
struct ConstExpr : public Expr {
  Expr *value;
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Const;
  }
};

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// The different kinds of statements that can appear in the AST.
enum class StmtKind { Empty, Expr, Let };

/// Base class for all statements.
struct Stmt {
  const StmtKind kind;
  Location loc;
};

/// A statement that is simply an expression.
struct ExprStmt : public Stmt {
  Expr *expr;
  static bool classof(const Stmt *stmt) { return stmt->kind == StmtKind::Expr; }
};

/// A let binding.
struct LetStmt : public Stmt {
  StringRef name;
  Type *type;  // optional
  Expr *value; // optional
  static bool classof(const Stmt *stmt) { return stmt->kind == StmtKind::Let; }
};

} // namespace ast

//===----------------------------------------------------------------------===//
// AST Container
//===----------------------------------------------------------------------===//

/// A container that holds an entire AST and owns the memory for all its nodes.
struct AST {
  /// The root nodes of the AST.
  SmallVector<ast::Root *> roots;

  /// Move a single node into the memory owned by the AST and return a reference
  /// to it. The reference remains valid for as long as the AST is alive.
  template <typename T>
  T *create(T &&node) {
    return new (allocator) T(std::forward<T>(node));
  }

  /// Move an array of nodes into the memory owned by the AST and return an
  /// ArrayRef to it. The ArrayRef remains valid for as long as the AST is
  /// alive.
  template <class Container, typename T = typename std::remove_reference<
                                 Container>::type::value_type>
  ArrayRef<T> array(Container &&container) {
    auto num = llvm::range_size(container);
    T *data = allocator.Allocate<T>(num);
    ArrayRef a(data, num);
    for (auto &&element : container)
      new (data++) T(element);
    return a;
  }

private:
  /// A bump pointer allocator that allocates memory for AST nodes.
  llvm::BumpPtrAllocator allocator;
};

} // namespace silicon
