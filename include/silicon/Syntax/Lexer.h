//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Support/MLIR.h"
#include "silicon/Syntax/Tokens.h"
#include "llvm/ADT/StringMap.h"

namespace silicon {

/// A lexer that splits an input file into tokens.
struct Lexer {
  Lexer(MLIRContext *context, SourceMgr &sourceMgr);

  /// Return the next token in the input.
  Token next();

  /// Determine the location of a substring in the source text.
  Location getLoc(StringRef substring);
  /// Determine the location of a token in the source text.
  Location getLoc(Token token) { return getLoc(token.spelling); }

  /// The MLIR context into which we intern strings and locations.
  MLIRContext *context;
  /// The source manager passed that holds the source text.
  SourceMgr &sourceMgr;
  /// The name of the source file. This name is used when creating locations.
  StringAttr filename;

private:
  /// The remaining text to be tokenized.
  StringRef text;
  /// A reference to a statically-allocated lookup table for keywords. This
  /// avoids having to recheck whether the keyword table has been constructed
  /// whenever we parse a keyword.
  llvm::StringMap<TokenKind> &keywordTable;
};

} // namespace silicon
