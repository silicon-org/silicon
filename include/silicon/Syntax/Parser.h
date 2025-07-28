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
  Parser(Lexer &lexer, AST &ast);

  ast::Root *parseRoot();
  ast::Item *parseItem();
  ast::FnItem *parseFnItem(Token kw);
  ast::FnArg *parseFnArg();
  ast::Type *parseType();
  ast::Expr *parseExpr();
  ast::Expr *parsePrimaryExpr();

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

  /// Check whether we have not reached the end of the input and the current
  /// token is not of the given kind. This is useful to have a while loop
  /// iterate until a closing delimiter is reached, or an end-of-file token is
  /// encountered which can then be handled separately.
  bool notAtDelimiter(TokenKind kind) { return token && token.kind != kind; }
};

} // namespace silicon
