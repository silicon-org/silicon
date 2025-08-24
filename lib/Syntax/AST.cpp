//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/AST.h"

using namespace silicon;

ast::Precedence ast::getPrecedence(BinaryOp op) {
  switch (op) {
#define AST_BINARY(NAME, TOKEN, PREC)                                          \
  case BinaryOp::NAME:                                                         \
    return Precedence::PREC;
#include "silicon/Syntax/AST.def"
  };
}

//===----------------------------------------------------------------------===//
// Type Names
//===----------------------------------------------------------------------===//

StringRef ast::Item::getTypeName() const {
  switch (kind) {
#define AST_ITEM(NAME)                                                         \
  case ItemKind::NAME:                                                         \
    return static_cast<const NAME##Item *>(this)->typeName;
#include "silicon/Syntax/AST.def"
  }
}

StringRef ast::Type::getTypeName() const {
  switch (kind) {
#define AST_TYPE(NAME)                                                         \
  case TypeKind::NAME:                                                         \
    return static_cast<const NAME##Type *>(this)->typeName;
#include "silicon/Syntax/AST.def"
  }
}

StringRef ast::Expr::getTypeName() const {
  switch (kind) {
#define AST_EXPR(NAME)                                                         \
  case ExprKind::NAME:                                                         \
    return static_cast<const NAME##Expr *>(this)->typeName;
#include "silicon/Syntax/AST.def"
  }
}

StringRef ast::Stmt::getTypeName() const {
  switch (kind) {
#define AST_STMT(NAME)                                                         \
  case StmtKind::NAME:                                                         \
    return static_cast<const NAME##Stmt *>(this)->typeName;
#include "silicon/Syntax/AST.def"
  }
}

//===----------------------------------------------------------------------===//
// Printing
//===----------------------------------------------------------------------===//

StringRef ast::toString(UnaryOp op) {
  switch (op) {
#define AST_UNARY(NAME, TOKEN)                                                 \
  case UnaryOp::NAME:                                                          \
    return #NAME;
#include "silicon/Syntax/AST.def"
  }
}

StringRef ast::toString(BinaryOp op) {
  switch (op) {
#define AST_BINARY(NAME, TOKEN, PREC)                                          \
  case BinaryOp::NAME:                                                         \
    return #NAME;
#include "silicon/Syntax/AST.def"
  }
}

namespace {
/// A visitor that prints the AST to an output stream in a tree-like format.
struct ASTPrinter : public ast::Visitor<ASTPrinter> {
  llvm::raw_ostream &os;
  SmallString<32> indentBuffer;
  DenseMap<void *, unsigned> nodeIds;
  using Visitor::visit;
  ASTPrinter(llvm::raw_ostream &os) : os(os) {}

  unsigned getNodeId(void *ptr) {
    return nodeIds.insert({ptr, nodeIds.size()}).first->second;
  }

  void enterField(StringRef field) { os << indentBuffer << field << ": "; }

  /// Print arrays as indented list with a vertical bar for each element.
  template <typename T>
  void visit(ArrayRef<T> nodes) {
    if (nodes.empty())
      os << "<empty>";
    os << "\n";
    for (unsigned i = 0; i < nodes.size(); ++i) {
      os << indentBuffer;
      if (i + 1 == nodes.size()) {
        os << "`-";
        indentBuffer += "  ";
      } else {
        os << "|-";
        indentBuffer += "| ";
      }
      visit(nodes[i]);
      indentBuffer.pop_back_n(2);
    }
  }

  // Print basic values.
  void visit(StringRef string) { os << "\"" << string << "\"\n"; }
  void visit(APInt &value) { os << value << "\n"; }
  void visit(ast::UnaryOp &op) { os << ast::toString(op) << "\n"; }
  void visit(ast::BinaryOp &op) { os << ast::toString(op) << "\n"; }
  void visit(ast::Binding &binding) {
    TypeSwitch<ast::Binding>(binding)
        .Case<ast::FnItem *, ast::FnArg *, ast::LetStmt *>([&](auto *node) {
          os << node->getTypeName() << "(@"
             << getNodeId(static_cast<void *>(node)) << ")\n";
        });
  }

  /// Print the type name of the AST node and indent for its children.
  template <typename T>
  void preVisitNode(T &node) {
    os << node.getTypeName() << " @" << getNodeId(static_cast<void *>(&node))
       << "\n";
    indentBuffer += "  ";
  }

  /// Unindent after visiting the children of the AST node.
  template <typename T>
  void postVisitNode(T &node) {
    indentBuffer.pop_back_n(2);
  }
};
} // namespace

void AST::print(llvm::raw_ostream &os) {
  ASTPrinter printer(os);
  walk(printer);
}
