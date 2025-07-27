//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/Tokens.h"

using namespace silicon;

StringRef silicon::symbolizeTokenKind(TokenKind kind) {
  switch (kind) {
  case TokenKind::Eof:
    return "end of file";
  case TokenKind::Error:
    return "<error>";
  case TokenKind::Ident:
    return "identifier";

#define TOK_KEYWORD(IDENT)                                                     \
  case TokenKind::Kw_##IDENT:                                                  \
    return "keyword `" #IDENT "`";

#define TOK_SYMBOL(NAME, SPELLING)                                             \
  case TokenKind::NAME:                                                        \
    return "`" SPELLING "`";

#include "silicon/Syntax/Tokens.def"
  }
  llvm_unreachable("should handle all token kinds");
  return "<unknown>";
}

bool silicon::shouldPrintSpelling(TokenKind kind) {
  switch (kind) {
  case TokenKind::Ident:
    return true;
  default:
    return false;
  }
}
