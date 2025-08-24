//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Support/LLVM.h"
#include "silicon/Support/MLIR.h"
#include "llvm/ADT/APInt.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/raw_ostream.h"

/// Define a type name constant and method, and a walk method that traverses all
/// fields and children.
#define AST_VISIT_DEF(type)                                                    \
  static constexpr StringRef typeName = #type;                                 \
  StringRef getTypeName() const { return typeName; }                           \
  template <typename V, typename... Args>                                      \
  void walk(V &&visitor, Args &&..._args)

/// Declare visitor-related methods for base classes of AST nodes, to be
/// implemented later on in the file once all AST nodes are defined.
#define AST_VISIT_DECL()                                                       \
  StringRef getTypeName() const;                                               \
  template <typename V, typename... Args>                                      \
  void walk(V &&visitor, Args &&...args)

/// Visit a field. Call this inside AST_VISIT_DEF.
#define AST_VISIT(field)                                                       \
  visitor.visitField(field, #field, std::forward<Args>(_args)...)

/// Visit an optional field. If the field is null, it is not visited. Call this
/// inside AST_VISIT_DEF.
#define AST_VISIT_OPTIONAL(field)                                              \
  if (field) {                                                                 \
    AST_VISIT(field);                                                          \
  }

namespace silicon {
namespace ast {

struct BlockExpr;
struct Expr;
struct FnArg;
struct FnItem;
struct Item;
struct LetStmt;
struct Stmt;
struct Type;

/// A reference to an AST node produced by name resolution.
using Binding = PointerUnion<FnItem *, FnArg *, LetStmt *>;

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
  AST_VISIT_DEF(Root) { AST_VISIT(items); }
};

/// Call `fn` with the given `node` and optional `args`.
template <typename Callable, typename T, typename... Args>
decltype(auto) visit(T &node, Callable &&fn, Args &&...args) {
  return fn(node, std::forward<Args>(args)...);
}

//===----------------------------------------------------------------------===//
// Items
//===----------------------------------------------------------------------===//

/// The different kinds of items that can appear in the AST.
enum class ItemKind {
#define AST_ITEM(NAME) NAME,
#include "silicon/Syntax/AST.def"
};

/// Base class for all top-level items.
struct Item {
  const ItemKind kind;
  Location loc;
  AST_VISIT_DECL();
};

/// A function declaration.
struct FnItem : public Item {
  StringRef name;
  ArrayRef<FnArg *> args;
  Type *returnType; // optional
  BlockExpr *body;

  AST_VISIT_DEF(FnItem) {
    AST_VISIT(name);
    AST_VISIT(args);
    AST_VISIT_OPTIONAL(returnType);
    AST_VISIT(body);
  }
  static bool classof(const Item *item) { return item->kind == ItemKind::Fn; }
};

/// An argument of a function declaration.
struct FnArg {
  Location loc;
  StringRef name;
  Type *type;

  AST_VISIT_DEF(FnArg) {
    AST_VISIT(name);
    AST_VISIT(type);
  }
};

/// Call `fn` with the concrete subclass of `item` and optional `args`.
template <typename Callable, typename... Args>
decltype(auto) visit(ast::Item &item, Callable &&fn, Args &&...args) {
  switch (item.kind) {
#define AST_ITEM(NAME)                                                         \
  case ItemKind::NAME:                                                         \
    return fn(static_cast<NAME##Item &>(item), std::forward<Args>(args)...);
#include "silicon/Syntax/AST.def"
  }
}

/// Walk implementation for items.
template <typename V, typename... Args>
void Item::walk(V &&visitor, Args &&...args) {
  return visit(*this, [&](auto &node) {
    node.walk(visitor, std::forward<Args>(args)...);
  });
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

/// The different kinds of types that can appear in the AST.
enum class TypeKind {
#define AST_TYPE(NAME) NAME,
#include "silicon/Syntax/AST.def"
};

/// Base class for all types.
struct Type {
  const TypeKind kind;
  Location loc;
  AST_VISIT_DECL();
};

/// A constant type, which wraps another type.
struct ConstType : public Type {
  Type *type;
  AST_VISIT_DEF(ConstType) { AST_VISIT(type); }
  static bool classof(const Type *type) {
    return type->kind == TypeKind::Const;
  }
};

/// A generic integer type.
struct IntType : public Type {
  AST_VISIT_DEF(IntType) {}
  static bool classof(const Type *type) { return type->kind == TypeKind::Int; }
};

/// An unsigned integer type with a specific width.
struct UIntType : public Type {
  Expr *width;
  AST_VISIT_DEF(UIntType) { AST_VISIT(width); }
  static bool classof(const Type *type) { return type->kind == TypeKind::UInt; }
};

/// Call `fn` with the concrete subclass of `type` and optional `args`.
template <typename Callable, typename... Args>
decltype(auto) visit(ast::Type &type, Callable &&fn, Args &&...args) {
  switch (type.kind) {
#define AST_TYPE(NAME)                                                         \
  case TypeKind::NAME:                                                         \
    return fn(static_cast<NAME##Type &>(type), std::forward<Args>(args)...);
#include "silicon/Syntax/AST.def"
  }
}

/// Walk implementation for types.
template <typename V, typename... Args>
void Type::walk(V &&visitor, Args &&...args) {
  return visit(*this, [&](auto &node) {
    node.walk(visitor, std::forward<Args>(args)...);
  });
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// The different kinds of expressions that can appear in the AST.
enum class ExprKind {
#define AST_EXPR(NAME) NAME,
#include "silicon/Syntax/AST.def"
};

/// Base class for all expressions.
struct Expr {
  const ExprKind kind;
  Location loc;
  AST_VISIT_DECL();
};

/// An identifier expression, which refers to something by name.
struct IdentExpr : public Expr {
  StringRef name;
  Binding binding;
  AST_VISIT_DEF(IdentExpr) {
    AST_VISIT(name);
    AST_VISIT_OPTIONAL(binding);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Ident;
  }
};

/// A number literal expression.
struct NumLitExpr : public Expr {
  APInt value;
  AST_VISIT_DEF(NumLitExpr) { AST_VISIT(value); }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::NumLit;
  }
};

/// A call expression.
struct CallExpr : public Expr {
  Expr *callee;
  ArrayRef<Expr *> args;
  AST_VISIT_DEF(CallExpr) {
    AST_VISIT(callee);
    AST_VISIT(args);
  }
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

/// Convert a unary operator to a human-readable string.
StringRef toString(UnaryOp op);
/// Convert a binary operator to a human-readable string.
StringRef toString(BinaryOp op);

/// Return the precedence of the given binary operator.
Precedence getPrecedence(BinaryOp op);

/// A unary operator expression.
struct UnaryExpr : public Expr {
  UnaryOp op;
  Expr *arg;
  AST_VISIT_DEF(UnaryExpr) {
    AST_VISIT(op);
    AST_VISIT(arg);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Unary;
  }
};

/// A binary operator expression.
struct BinaryExpr : public Expr {
  BinaryOp op;
  Expr *lhs;
  Expr *rhs;
  AST_VISIT_DEF(BinaryExpr) {
    AST_VISIT(op);
    AST_VISIT(lhs);
    AST_VISIT(rhs);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Binary;
  }
};

/// A block expression, which is a sequence of statements and an optional
/// final expression that is used as the block's return value.
struct BlockExpr : public Expr {
  ArrayRef<Stmt *> stmts;
  Expr *result; // optional
  AST_VISIT_DEF(BlockExpr) {
    AST_VISIT(stmts);
    AST_VISIT_OPTIONAL(result);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Block;
  }
};

/// An if expression.
struct IfExpr : public Expr {
  Expr *condition;
  Expr *thenExpr;
  Expr *elseExpr; // optional
  AST_VISIT_DEF(IfExpr) {
    AST_VISIT(condition);
    AST_VISIT(thenExpr);
    AST_VISIT_OPTIONAL(elseExpr);
  }
  static bool classof(const Expr *expr) { return expr->kind == ExprKind::If; }
};

/// A return expression.
struct ReturnExpr : public Expr {
  Expr *value; // optional
  AST_VISIT_DEF(ReturnExpr) { AST_VISIT_OPTIONAL(value); }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Return;
  }
};

/// An index expression.
struct IndexExpr : public Expr {
  Expr *base;
  Expr *index;
  AST_VISIT_DEF(IndexExpr) {
    AST_VISIT(base);
    AST_VISIT(index);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Index;
  }
};

/// A slice expression.
struct SliceExpr : public Expr {
  Expr *base;
  Expr *index;
  Expr *length;
  AST_VISIT_DEF(SliceExpr) {
    AST_VISIT(base);
    AST_VISIT(index);
    AST_VISIT(length);
  }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Slice;
  }
};

/// A constant expression.
struct ConstExpr : public Expr {
  Expr *value;
  AST_VISIT_DEF(ConstExpr) { AST_VISIT(value); }
  static bool classof(const Expr *expr) {
    return expr->kind == ExprKind::Const;
  }
};

/// Call `fn` with the concrete subclass of `expr` and optional `args`.
template <typename Callable, typename... Args>
decltype(auto) visit(ast::Expr &expr, Callable &&fn, Args &&...args) {
  switch (expr.kind) {
#define AST_EXPR(NAME)                                                         \
  case ExprKind::NAME:                                                         \
    return fn(static_cast<NAME##Expr &>(expr), std::forward<Args>(args)...);
#include "silicon/Syntax/AST.def"
  }
}

/// Walk implementation for expressions.
template <typename V, typename... Args>
void Expr::walk(V &&visitor, Args &&...args) {
  return visit(*this, [&](auto &node) {
    node.walk(visitor, std::forward<Args>(args)...);
  });
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

/// The different kinds of statements that can appear in the AST.
enum class StmtKind {
#define AST_STMT(NAME) NAME,
#include "silicon/Syntax/AST.def"
};

/// Base class for all statements.
struct Stmt {
  const StmtKind kind;
  Location loc;
  AST_VISIT_DECL();
};

/// An empty statement. This is used to capture stray semicolons.
struct EmptyStmt : public Stmt {
  AST_VISIT_DEF(EmptyStmt) {}
  static bool classof(const Stmt *stmt) {
    return stmt->kind == StmtKind::Empty;
  }
};

/// A statement that is simply an expression.
struct ExprStmt : public Stmt {
  Expr *expr;
  AST_VISIT_DEF(ExprStmt) { AST_VISIT(expr); }
  static bool classof(const Stmt *stmt) { return stmt->kind == StmtKind::Expr; }
};

/// A let binding.
struct LetStmt : public Stmt {
  StringRef name;
  Type *type;  // optional
  Expr *value; // optional
  AST_VISIT_DEF(LetStmt) {
    AST_VISIT(name);
    AST_VISIT_OPTIONAL(type);
    AST_VISIT_OPTIONAL(value);
  }
  static bool classof(const Stmt *stmt) { return stmt->kind == StmtKind::Let; }
};

/// Call `fn` with the concrete subclass of `stmt` and optional `args`.
template <typename Callable, typename... Args>
decltype(auto) visit(ast::Stmt &stmt, Callable &&fn, Args &&...args) {
  switch (stmt.kind) {
#define AST_STMT(NAME)                                                         \
  case StmtKind::NAME:                                                         \
    return fn(static_cast<NAME##Stmt &>(stmt), std::forward<Args>(args)...);
#include "silicon/Syntax/AST.def"
  }
}

/// Walk implementation for statements.
template <typename V, typename... Args>
void Stmt::walk(V &&visitor, Args &&...args) {
  return visit(*this, [&](auto &node) {
    node.walk(visitor, std::forward<Args>(args)...);
  });
}

//===----------------------------------------------------------------------===//
// Visitor
//===----------------------------------------------------------------------===//

template <typename Derived>
struct Visitor {
  /// Called by the AST nodes for each of their member fields.
  template <typename T, typename... Args>
  void visitField(T &value, StringRef field, Args &&...args) {
    derived().enterField(field, std::forward<Args>(args)...);
    derived().visit(value, std::forward<Args>(args)...);
    derived().leaveField(field, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  decltype(auto) visit(T &node, Args &&...args) {
    return derived().visitNode(node, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  decltype(auto) visit(T *node, Args &&...args) {
    return derived().visit(*node, std::forward<Args>(args)...);
  }

  /// Dispatch concrete items.
  template <typename... Args>
  decltype(auto) visit(ast::Item &node, Args &&...args) {
    return ast::visit(node, [&](auto &node) {
      return derived().visitNode(node, std::forward<Args>(args)...);
    });
  }

  /// Dispatch concrete types.
  template <typename... Args>
  decltype(auto) visit(ast::Type &node, Args &&...args) {
    return ast::visit(node, [&](auto &node) {
      return derived().visitNode(node, std::forward<Args>(args)...);
    });
  }

  /// Dispatch concrete expressions.
  template <typename... Args>
  decltype(auto) visit(ast::Expr &node, Args &&...args) {
    return ast::visit(node, [&](auto &node) {
      return derived().visitNode(node, std::forward<Args>(args)...);
    });
  }

  /// Dispatch concrete statements.
  template <typename... Args>
  decltype(auto) visit(ast::Stmt &node, Args &&...args) {
    return ast::visit(node, [&](auto &node) {
      return derived().visitNode(node, std::forward<Args>(args)...);
    });
  }

  template <typename... Args>
  inline void visit(StringRef, Args &&...) {}
  template <typename... Args>
  inline void visit(Location, Args &&...) {}
  template <typename... Args>
  inline void visit(APInt &, Args &&...) {}
  template <typename... Args>
  inline void visit(UnaryOp, Args &&...) {}
  template <typename... Args>
  inline void visit(BinaryOp, Args &&...) {}
  template <typename... Args>
  inline void visit(Binding, Args &&...) {}

  template <typename T, typename... Args>
  void visit(ArrayRef<T> nodes, Args &&...args) {
    for (auto &node : nodes)
      derived().visit(node, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  void visitNode(T &node, Args &&...args) {
    derived().preVisitNode(node, std::forward<Args>(args)...);
    node.walk(derived(), std::forward<Args>(args)...);
    derived().postVisitNode(node, std::forward<Args>(args)...);
  }

  template <typename T, typename... Args>
  inline void preVisitNode(T &node, Args &&...args) {}
  template <typename T, typename... Args>
  inline void postVisitNode(T &node, Args &&...args) {}

  template <typename... Args>
  inline void enterField(StringRef field, Args &&...args) {}
  template <typename... Args>
  inline void leaveField(StringRef field, Args &&...args) {}

private:
  inline Derived &derived() { return static_cast<Derived &>(*this); }
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

  /// Visit the nodes in the AST.
  template <typename V, typename... Args>
  void walk(V &&visitor, Args &&...args) {
    for (auto *root : roots)
      visitor.visit(root, std::forward<Args>(args)...);
  }

  /// Print the AST to the given output stream.
  void print(llvm::raw_ostream &os);

private:
  /// A bump pointer allocator that allocates memory for AST nodes.
  llvm::BumpPtrAllocator allocator;
};

} // namespace silicon

#undef AST_VISIT_DEF
#undef AST_VISIT
