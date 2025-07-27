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

  /// The lexer that produces the input tokens.
  Lexer &lexer;
  /// The AST into which nodes are allocated.
  AST &ast;

private:
  /// The current token being parsed.
  Token token;

  /// Get the location of the current token.
  inline Location loc() const { return loc(token); }
  /// Get the location of a token.
  inline Location loc(Token token) const { return lexer.getLoc(token); }
};

} // namespace silicon
