//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/LLVM.h"
#include "silicon/Syntax/AST.h"
#include "silicon/Syntax/Names.h"
#include "llvm/ADT/ScopeExit.h"

#define DEBUG_TYPE "names"

using namespace silicon;

/// Get the location of a binding.
static Location getLoc(ast::Binding binding) {
  return TypeSwitch<ast::Binding, Location>(binding)
      .Case<ast::FnItem *, ast::FnArg *, ast::LetStmt *>(
          [](auto *node) { return node->loc; });
}

/// Print a binding as `<type>(<address>)` for debugging purposes.
static llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     ast::Binding binding) {
  TypeSwitch<ast::Binding>(binding)
      .Case<ast::FnItem *, ast::FnArg *, ast::LetStmt *>([&](auto *node) {
        os << node->getTypeName() << "(" << static_cast<void *>(node) << ")";
      });
  return os;
}

namespace {
struct Declare {};
struct Resolve {};

struct Scope {
  SmallDenseMap<StringRef, ast::Binding> names;
};

/// A visitor that walks the AST and resolves names.
struct Resolver : public ast::Visitor<Resolver> {
  using Visitor::visit;
  using Visitor::visitNode;
  SmallVector<Scope> scopes;
  bool hasError = false;

  /// Declare a name in the current scope. If `redeclare` is true, it is
  /// allowed to redeclare an existing name, otherwise an error is emitted.
  void declareName(StringRef name, ast::Binding node, bool redeclare) {
    auto &slot = scopes.back().names[name];
    if (slot && !redeclare) {
      auto nodeLoc = getLoc(node);
      auto slotLoc = getLoc(slot);
      auto d = mlir::emitError(nodeLoc)
               << "name `" << name << "` already defined";
      d.attachNote(slotLoc)
          << "previous definition of `" << name << "` was here";
      hasError = true;
      return;
    }
    LLVM_DEBUG(dbgs().indent(scopes.size() * 2)
               << (slot ? "Red" : "D") << "eclaring `" << name << "` as "
               << node << " " << getLoc(node) << "\n");
    slot = node;
  }

  /// Resolve a name in the current scope or one of its parents. Reports an
  /// error for the given location if the name cannot be found.
  ast::Binding resolveName(StringRef name, Location loc) {
    for (auto &scope : llvm::reverse(scopes)) {
      if (auto node = scope.names.lookup(name)) {
        LLVM_DEBUG(dbgs().indent(scopes.size() * 2)
                   << "Resolved `" << name << "` to " << node << " " << loc
                   << "\n");
        return node;
      }
    }
    mlir::emitError(loc) << "unknown name `" << name << "`";
    hasError = true;
    return {};
  }

  /// Create a new scope and return a scope exit guard that will pop it when
  /// it goes out of scope (pun intended).
  auto makeScope() {
    LLVM_DEBUG(dbgs().indent(scopes.size() * 2) << "Subscope\n");
    scopes.emplace_back();
    return llvm::make_scope_exit([this] { scopes.pop_back(); });
  }

  // Do not descend into child nodes when declaring names. We only want to
  // collect order-agnostic items in this phase.
  template <typename T>
  void visitNode(T &node, Declare) {}

  // Declare names for order-agnostic items.
  void visitNode(ast::FnItem &item, Declare) {
    declareName(item.name, &item, false);
  }

  /// Standard handling for AST nodes that create a subscope for everything they
  /// contain.
  template <typename T>
  void resolveScope(T &node) {
    auto guard = makeScope();
    node.walk(*this, Declare{});
    if (hasError)
      return;
    node.walk(*this, Resolve{});
  }
  void visitNode(ast::Root &node, Resolve) { resolveScope(node); }
  void visitNode(ast::FnItem &node, Resolve) { resolveScope(node); }

  // Declare and resolve names.
  void visitNode(ast::FnArg &node, Resolve) {
    node.walk(*this, Resolve{});
    declareName(node.name, &node, false);
  }
  void visitNode(ast::LetStmt &node, Resolve) {
    node.walk(*this, Resolve{});
    declareName(node.name, &node, true);
  }
  void visitNode(ast::IdentExpr &node, Resolve) {
    node.binding = resolveName(node.name, node.loc);
  }
};
} // namespace

LogicalResult silicon::resolveNames(AST &ast) {
  Resolver resolver;
  ast.walk(resolver, Resolve{});
  return failure(resolver.hasError);
}
