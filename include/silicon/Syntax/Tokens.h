//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/Support/LLVM.h"

namespace silicon {

/// The different kinds of tokens that may appear in the source text.
enum class TokenKind {
  /// The end of the input file.
  Eof,
  /// An error encountered while parsing the token.
  Error,
#define TOK_ANY(NAME) NAME,
#include "silicon/Syntax/Tokens.def"
};

/// Get a string representation of a token.
StringRef symbolizeTokenKind(TokenKind kind);

/// Whether the token's spelling should be printed alongside its kind. This is
/// used for things like identifiers and number literals, where we want to
/// actually display the exact spelling in the source file.
bool shouldPrintSpelling(TokenKind kind);

/// A token in the source text.
struct Token {
  /// The exact fragment of the source text corresponding to the token.
  StringRef spelling;
  /// The token kind.
  TokenKind kind;

  inline bool isEof() const { return kind == TokenKind::Eof; }
  inline bool isError() const { return kind == TokenKind::Error; }

  /// Check whether the token represents the end of the input file.
  inline explicit operator bool() const { return !isEof(); }
};

/// Allow `Token` to be printed.
template <typename T>
static T &operator<<(T &os, const Token &token) {
  os << symbolizeTokenKind(token.kind);
  if (shouldPrintSpelling(token.kind))
    os << " `" << token.spelling << "`";
  return os;
}

} // namespace silicon
