//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/AST.h"
#include "silicon/Syntax/Parser.h"
#include "silicon/Syntax/Tokens.h"
#include "llvm/ADT/StringExtras.h"

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

  // Only emit an error if the lexer has not already emitted one.
  if (!token.isError()) {
    auto d = mlir::emitError(loc(), "expected ");
    if (msg.isTriviallyEmpty())
      d << symbolizeTokenKind(kind);
    else
      d << msg;
    d << ", found " << token;
  }

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
  if (!token.isError())
    mlir::emitError(loc()) << "expected item, found " << token;
  return {};
}

//===----------------------------------------------------------------------===//
// Functions
//===----------------------------------------------------------------------===//

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

  // Parse the optional function return type.
  ast::Type *returnType = {};
  if (consumeIf(TokenKind::Arrow)) {
    returnType = parseType();
    if (!returnType)
      return {};
  }

  // Parse the function body.
  auto *body = parseBlockExpr();
  if (!body)
    return {};

  return ast.create<ast::FnItem>({{ast::ItemKind::Fn, loc(name)},
                                  name.spelling,
                                  ast.array(args),
                                  returnType,
                                  body});
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
    // Use relational precedence to parse the width expression to not consume
    // the closing `>`.
    auto *width = parseExpr(ast::Precedence::Rel);
    if (!width)
      return {};
    if (!require(TokenKind::Gt))
      return {};
    return ast.create<ast::UIntType>({{ast::TypeKind::UInt, loc(kw)}, width});
  }

  // If we get here we didn't find a keyword that starts a type.
  if (!token.isError())
    mlir::emitError(loc()) << "expected type, found " << token;
  return {};
}

//===----------------------------------------------------------------------===//
// Expressions
//===----------------------------------------------------------------------===//

/// Check if an expression requires a semicolon when used as a statement.
static bool requiresSemicolon(ast::Expr *expr) {
  return !isa<ast::BlockExpr, ast::IfExpr>(expr);
}

/// Check whether the given token kind is a unary operator and return the
/// corresponding AST unary operator.
static std::optional<ast::UnaryOp> getUnaryOp(TokenKind kind) {
  switch (kind) {
#define AST_UNARY(NAME, TOKEN)                                                 \
  case TokenKind::TOKEN:                                                       \
    return ast::UnaryOp::NAME;
#include "silicon/Syntax/AST.def"
  default:
    return {};
  }
}

/// Check whether the given token kind is a binary operator and return the
/// corresponding AST binary operator.
static std::optional<ast::BinaryOp> getBinaryOp(TokenKind kind) {
  switch (kind) {
#define AST_BINARY(NAME, TOKEN, PREC)                                          \
  case TokenKind::TOKEN:                                                       \
    return ast::BinaryOp::NAME;
#include "silicon/Syntax/AST.def"
  default:
    return {};
  }
}

// NOLINTNEXTLINE(misc-no-recursion)
ast::Expr *Parser::parseExpr(ast::Precedence minPrec) {
  // Parse a primary expression first. This will form the LHS of any operators
  // that may follow.
  auto *lhs = parsePrimaryExpr();
  if (!lhs)
    return {};

  // Parse any operators that may follow the primary expression.
  while (token) {
    // Parse function calls.
    if (isa(TokenKind::LParen) && ast::Precedence::Suffix > minPrec) {
      auto lparen = consume();
      SmallVector<ast::Expr *> args;
      while (notAtDelimiter(TokenKind::RParen)) {
        auto *arg = parseExpr();
        if (!arg)
          return {};
        args.push_back(arg);
        if (!consumeIf(TokenKind::Comma))
          break;
      }
      if (!require(TokenKind::RParen))
        return {};
      lhs = ast.create<ast::CallExpr>(
          {{ast::ExprKind::Call, loc(lparen)}, lhs, ast.array(args)});
      continue;
    }

    // Parse indexing expressions.
    if (isa(TokenKind::LBrack) && ast::Precedence::Suffix > minPrec) {
      auto lbrack = consume();
      auto *index = parseExpr();
      if (!index)
        return {};
      if (!require(TokenKind::RBrack))
        return {};
      lhs = ast.create<ast::IndexExpr>(
          {{ast::ExprKind::Index, loc(lbrack)}, lhs, index});
      continue;
    }

    // Parse binary operators.
    if (auto op = getBinaryOp(token.kind)) {
      auto prec = ast::getPrecedence(*op);
      if (prec > minPrec) {
        auto opToken = consume();
        auto *rhs = parseExpr(prec);
        if (!rhs)
          return {};
        lhs = ast.create<ast::BinaryExpr>(
            {{ast::ExprKind::Binary, loc(opToken)}, *op, lhs, rhs});
        continue;
      }
    }

    // If we get here we didn't recognize the token as part of an expression.
    break;
  }

  return lhs;
}

// NOLINTNEXTLINE(misc-no-recursion)
ast::Expr *Parser::parsePrimaryExpr() {
  // Parse identifiers.
  if (auto ident = consumeIf(TokenKind::Ident))
    return ast.create<ast::IdentExpr>(
        {{ast::ExprKind::Ident, loc(ident)}, ident.spelling});

  // Parse number literals.
  if (auto lit = consumeIf(TokenKind::NumLit))
    return parseNumberLiteral(lit);

  // Parse block expressions.
  if (isa(TokenKind::LCurly))
    return parseBlockExpr();

  // Parse if expressions.
  if (auto kw = consumeIf(TokenKind::Kw_if)) {
    auto *condition = parseExpr(ast::Precedence::Min);
    if (!condition)
      return {};
    auto *thenExpr = parseBlockExpr();
    if (!thenExpr)
      return {};
    ast::Expr *elseExpr = {};
    if (consumeIf(TokenKind::Kw_else)) {
      if (isa(TokenKind::Kw_if))
        elseExpr = parseExpr(ast::Precedence::Min);
      else
        elseExpr = parseBlockExpr();
      if (!elseExpr)
        return {};
    }
    return ast.create<ast::IfExpr>(
        {{ast::ExprKind::If, loc(kw)}, condition, thenExpr, elseExpr});
  }

  // Parse return expressions.
  if (auto kw = consumeIf(TokenKind::Kw_return)) {
    ast::Expr *value = {};
    if (!isa(TokenKind::Semicolon, TokenKind::Comma, TokenKind::RParen,
             TokenKind::RBrack, TokenKind::RCurly)) {
      value = parseExpr(ast::Precedence::Min);
      if (!value)
        return {};
    }
    return ast.create<ast::ReturnExpr>(
        {{ast::ExprKind::Return, loc(kw)}, value});
  }

  // Parse constant expressions.
  if (auto kw = consumeIf(TokenKind::Kw_const)) {
    auto *value = parseBlockExpr();
    if (!value)
      return {};
    return ast.create<ast::ConstExpr>({{ast::ExprKind::Const, loc(kw)}, value});
  }

  // Parse unary operators.
  if (auto op = getUnaryOp(token.kind)) {
    auto opToken = consume();
    auto *arg = parseExpr(ast::Precedence::Prefix);
    if (!arg)
      return {};
    return ast.create<ast::UnaryExpr>(
        {{ast::ExprKind::Unary, loc(opToken)}, *op, arg});
  }

  // If we get here we didn't find anything that looks like an expression.
  if (!token.isError())
    mlir::emitError(loc()) << "expected expression, found " << token;
  return {};
}

/// Check whether the given character is a valid digit for the given base.
static bool isValidDigitForBase(char c, int base) {
  if (c >= '0' && c <= '9')
    return (c - '0') < base;
  c = llvm::toLower(c);
  if (c >= 'a' && c <= 'f')
    return (c - 'a' + 10) < base;
  return false;
}

ast::NumLitExpr *Parser::parseNumberLiteral(Token lit) {
  StringRef spelling = lit.spelling;

  // Determine the base.
  unsigned base = 10;
  if (spelling.consume_front("0b"))
    base = 2;
  else if (spelling.consume_front("0o"))
    base = 8;
  else if (spelling.consume_front("0x"))
    base = 16;

  // Filter out `_` and check for invalid digits for the given base.
  SmallString<32> digits;
  digits.reserve(spelling.size());
  for (unsigned i = 0, e = spelling.size(); i != e; ++i) {
    if (spelling[i] == '_')
      continue;
    if (!isValidDigitForBase(spelling[i], base)) {
      mlir::emitError(loc(spelling.substr(i)))
          << "`" << spelling[i] << "` is not a valid base-" << base << " digit";
      return {};
    }
    digits.push_back(spelling[i]);
  }

  // Parse the integer. At this point we know that all characters in `digits`
  // are valid digits for the given base.
  APInt value;
  auto hasRemainder = digits.str().getAsInteger(base, value);
  assert(!hasRemainder);

  // Resize the APInt to the smallest possible width that can hold the value.
  value = value.zextOrTrunc(value.getActiveBits());

  return ast.create<ast::NumLitExpr>(
      {{ast::ExprKind::NumLit, loc(lit)}, value});
}

// NOLINTNEXTLINE(misc-no-recursion)
ast::BlockExpr *Parser::parseBlockExpr() {
  auto lcurly = require(TokenKind::LCurly);
  if (!lcurly)
    return {};

  SmallVector<ast::Stmt *> stmts;
  ast::Expr *result = {};

  // Parse the statements in the block. The very last statement in the block may
  // be an expression without a terminating semicolon, in which case that
  // expressions becomes the block's return value.
  while (notAtDelimiter(TokenKind::RCurly)) {
    // Parse an expression or statement.
    auto stmtOrExpr = parseStmtOrExpr();
    if (!stmtOrExpr)
      return {};

    // If this is an expression, this may be the final expression in the block
    // which defines the block's return value.
    if (auto *expr = dyn_cast<ast::Expr *>(stmtOrExpr)) {
      // If the next token is a closing curly brace, use the expression as the
      // block's return value and break out of the loop.
      if (isa(TokenKind::RCurly)) {
        result = expr;
        break;
      }

      // If the expression requires a semicolon when used as a statement, which
      // is true for almost all expressions, consume that semicolon.
      if (requiresSemicolon(expr))
        if (!require(TokenKind::Semicolon))
          return {};

      // Wrap the expression in a statement so we can add it to the block.
      stmts.push_back(
          ast.create<ast::ExprStmt>({{ast::StmtKind::Expr, expr->loc}, expr}));
      continue;
    }

    // Since this was not an expression, it must be a statement.
    stmts.push_back(cast<ast::Stmt *>(stmtOrExpr));
  }

  // Consume the closing curly brace.
  if (!require(TokenKind::RCurly))
    return {};

  return ast.create<ast::BlockExpr>(
      {{ast::ExprKind::Block, loc(lcurly)}, stmts, result});
}

//===----------------------------------------------------------------------===//
// Statements
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(misc-no-recursion)
Parser::ExprOrStmt Parser::parseStmtOrExpr() {
  // Ignore stray semicolons.
  if (auto token = consumeIf(TokenKind::Semicolon))
    return ast.create<ast::Stmt>({ast::StmtKind::Empty, loc(token)});

  // Parse let bindings.
  if (auto kw = consumeIf(TokenKind::Kw_let)) {
    // Parse the variable name.
    auto name = require(TokenKind::Ident, "variable name");
    if (!name)
      return {};

    // Parse the optional type of the variable.
    ast::Type *type = {};
    if (consumeIf(TokenKind::Colon)) {
      type = parseType();
      if (!type)
        return {};
    }

    // Parse the optional value of the variable.
    ast::Expr *value = {};
    if (consumeIf(TokenKind::Assign)) {
      value = parseExpr(ast::Precedence::Min);
      if (!value)
        return {};
    }

    if (!require(TokenKind::Semicolon))
      return {};
    return ast.create<ast::LetStmt>(
        {{ast::StmtKind::Let, loc(kw)}, name.spelling, type, value});
  }

  // Otherwise this is an expression.
  return parseExpr();
}
