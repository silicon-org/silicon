//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/Parser.h"

using namespace silicon;

Parser::Parser(Lexer &lexer, AST &ast) : lexer(lexer), ast(ast) {
  token = lexer.next();
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

  return &ast.create<ast::Root>({ast.array(items)});
}

ast::Item *Parser::parseItem() {
  mlir::emitError(loc()) << "item parsing not implemented";
  return {};
}
