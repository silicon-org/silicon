//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Syntax/AST.h"
#include "silicon/Syntax/Lexer.h"

namespace silicon {

/// A parser that converts a stream of input tokens into a syntax tree.
struct Parser {
  using ExprOrStmt = PointerUnion<ast::Expr *, ast::Stmt *>;
  Parser(Lexer &lexer, AST &ast);

  ast::Root *parseRoot();
  ast::Item *parseItem();
  ast::FnItem *parseFnItem(Token kw);
  ast::FnArg *parseFnArg();
  ast::Type *parseType();
  ast::Expr *parseExpr(ast::Precedence minPrec = ast::Precedence::Min);
  ast::Expr *parsePrimaryExpr();
  ast::NumLitExpr *parseNumberLiteral(Token lit);
  ast::BlockExpr *parseBlockExpr();
  ExprOrStmt parseStmtOrExpr();

  /// The lexer that produces the input tokens.
  Lexer &lexer;
  /// The AST into which nodes are allocated.
  AST &ast;

private:
  /// The current token being parsed.
  Token token;

  /// Get the location of the current token.
  Location loc() const { return loc(token); }
  /// Get the location of a token.
  Location loc(Token token) const { return lexer.getLoc(token); }
  /// Get the location of a substring of the source text.
  Location loc(StringRef substring) const { return lexer.getLoc(substring); }

  /// Return the current token and advance to the next one.
  Token consume();

  /// Consume the current token if it matches the given kind, otherwise return
  /// an end-of-file token.
  Token consumeIf(TokenKind kind);

  /// Require the current token to be of the given kind, otherwise emit an error
  /// and return an end-of-file token.
  [[nodiscard]] Token require(TokenKind kind, const Twine &msg = {});

  /// Check whether the current token is of the given kind.
  bool isa(TokenKind kind) { return token.kind == kind; }
  /// Check whether the current token is of one of the given kinds.
  template <typename... Kinds>
  bool isa(TokenKind kind, Kinds... kinds) {
    return isa(kind) || isa(kinds...);
  }

  /// Check whether we have not reached the end of the input and the current
  /// token is not of the given kind. This is useful to have a while loop
  /// iterate until a closing delimiter is reached, or an end-of-file token is
  /// encountered which can then be handled separately.
  bool notAtDelimiter(TokenKind kind) { return token && token.kind != kind; }
};

} // namespace silicon
