//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/Parser.h"
#include "silicon/Syntax/Tokens.h"

using namespace silicon;

//===----------------------------------------------------------------------===//
// Parser Infrastructure
//===----------------------------------------------------------------------===//

Parser::Parser(Lexer &lexer, AST &ast) : lexer(lexer), ast(ast) {
  token = lexer.next();
}

Token Parser::consume() {
  auto consumedToken = token;
  token = lexer.next();
  return consumedToken;
}

Token Parser::consumeIf(TokenKind kind) {
  if (isa(kind))
    return consume();
  return {token.spelling.substr(0, 0), TokenKind::Eof};
}

Token Parser::require(TokenKind kind, const Twine &msg) {
  if (isa(kind))
    return consume();
  auto d = mlir::emitError(loc(), "expected ");
  if (msg.isTriviallyEmpty())
    d << symbolizeTokenKind(kind);
  else
    d << msg;
  d << ", found " << token;
  return {token.spelling.substr(0, 0), TokenKind::Eof};
}

//===----------------------------------------------------------------------===//
// Grammar
//===----------------------------------------------------------------------===//

ast::Root *Parser::parseRoot() {
  SmallVector<ast::Item *> items;
  while (token) {
    auto *item = parseItem();
    if (!item)
      return {};
    items.push_back(item);
  }
  return ast.create<ast::Root>({ast.array(items)});
}

ast::Item *Parser::parseItem() {
  // Parse function definitions.
  if (auto kw = consumeIf(TokenKind::Kw_fn))
    return parseFnItem(kw);

  // If we get here we didn't find a keyword that starts an item.
  mlir::emitError(loc()) << "expected item, found " << token;
  return {};
}

ast::FnItem *Parser::parseFnItem(Token kw) {
  // Parse the function name.
  auto name = require(TokenKind::Ident, "function name");
  if (!name)
    return {};

  // Parse the function arguments.
  SmallVector<ast::FnArg *> args;
  if (!require(TokenKind::LParen))
    return {};
  while (notAtDelimiter(TokenKind::RParen)) {
    auto *arg = parseFnArg();
    if (!arg)
      return {};
    args.push_back(arg);
    if (!consumeIf(TokenKind::Comma))
      break;
  }
  if (!require(TokenKind::RParen))
    return {};

  return ast.create<ast::FnItem>(
      {{ast::ItemKind::Fn, loc(name)}, name.spelling});
}

ast::FnArg *Parser::parseFnArg() {
  // Parse the argument name.
  auto name = require(TokenKind::Ident, "argument name");
  if (!name)
    return {};

  // Parse the argument type.
  if (!require(TokenKind::Colon))
    return {};
  auto *type = parseType();
  if (!type)
    return {};

  return ast.create<ast::FnArg>({loc(name), name.spelling, type});
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
ast::Type *Parser::parseType() {
  // Parse the `const` type.
  if (auto kw = consumeIf(TokenKind::Kw_const)) {
    auto *type = parseType();
    if (!type)
      return {};
    return ast.create<ast::ConstType>({{ast::TypeKind::Const, loc(kw)}, type});
  }

  // Parse the `int` type.
  if (auto kw = consumeIf(TokenKind::Kw_int))
    return ast.create<ast::Type>({ast::TypeKind::Int, loc(kw)});

  // Parse the `uint` type.
  if (auto kw = consumeIf(TokenKind::Kw_uint)) {
    if (!require(TokenKind::Lt))
      return {};
    if (!require(TokenKind::Gt))
      return {};
    return ast.create<ast::Type>({ast::TypeKind::UInt, loc(kw)});
  }

  // If we get here we didn't find a keyword that starts a type.
  mlir::emitError(loc()) << "expected type, found " << token;
  return {};
}
