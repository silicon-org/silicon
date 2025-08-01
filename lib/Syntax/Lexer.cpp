//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/Lexer.h"
#include "silicon/Syntax/Tokens.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/ManagedStatic.h"

using namespace silicon;

//===----------------------------------------------------------------------===//
// Keyword Lookup Table
//===----------------------------------------------------------------------===//

namespace {
struct KeywordTableCreator {
  static void *call() {
    auto table = std::make_unique<llvm::StringMap<TokenKind>>();
#define TOK_KEYWORD(IDENT) table->insert({#IDENT, TokenKind::Kw_##IDENT});
#include "silicon/Syntax/Tokens.def"
    return table.release();
  }
};
llvm::ManagedStatic<llvm::StringMap<TokenKind>, KeywordTableCreator>
    staticKeywordTable;
} // namespace

//===----------------------------------------------------------------------===//
// Lexer
//===----------------------------------------------------------------------===//

Lexer::Lexer(MLIRContext *context, SourceMgr &sourceMgr)
    : context(context), sourceMgr(sourceMgr),
      keywordTable(*staticKeywordTable) {
  auto *buffer = sourceMgr.getMemoryBuffer(sourceMgr.getMainFileID());

  /// Guess the filename from the main file in the source manager. Once we
  /// support multiple files, we'll have to be a bit more clever about this.
  filename = StringAttr::get(context, buffer->getBufferIdentifier());

  /// Start out with the entire source file as the remaining text.
  text = buffer->getBuffer();
}

Location Lexer::getLoc(StringRef substring) {
  auto loc = SMLoc::getFromPointer(substring.data());
  auto lineCol = sourceMgr.getLineAndColumn(loc, sourceMgr.getMainFileID());
  return FileLineColRange::get(filename, lineCol.first, lineCol.second,
                               lineCol.second + substring.size());
}

//===----------------------------------------------------------------------===//
// Tokenization
//===----------------------------------------------------------------------===//

/// Check whether `text` starts with a valid 1-character symbol token.
static TokenKind matchSymbol1(StringRef text) {
  if (text.size() < 1)
    return TokenKind::Eof;
  switch (text[0]) {
#define TOK_SYMBOL1(NAME, SPELLING)                                            \
  case SPELLING[0]:                                                            \
    return TokenKind::NAME;
#include "silicon/Syntax/Tokens.def"
  default:
    return TokenKind::Eof;
  }
}

/// Check whether `text` starts with a valid 2-character symbol token.
static TokenKind matchSymbol2(StringRef text) {
  if (text.size() < 2)
    return TokenKind::Eof;
  switch (text[0] | text[1] << 8) {
#define TOK_SYMBOL2(NAME, SPELLING)                                            \
  case (SPELLING)[0] | (SPELLING)[1] << 8:                                     \
    return TokenKind::NAME;
#include "silicon/Syntax/Tokens.def"
  default:
    return TokenKind::Eof;
  }
}

/// Check whether a character is a valid start of an identifier.
static bool isIdentStart(char c) { return llvm::isAlpha(c) || c == '_'; }

/// Check whether a character is valid in an identifier.
static bool isIdent(char c) { return isIdentStart(c) || llvm::isDigit(c); }

Token Lexer::next() {
  // Ignore things in front of the next token.
  while (!text.empty()) {
    // Skip whitespace.
    while (!text.empty() && llvm::isSpace(text[0]))
      text = text.drop_front();

    // Skip single-line comments.
    if (text.consume_front("//")) {
      while (!text.empty() && text[0] != '\n')
        text = text.drop_front();
      continue;
    }

    // Skip multi-line comments.
    if (text.starts_with("/*")) {
      auto commentStart = text.substr(0, 2);
      text = text.drop_front(2);
      while (!text.empty() && !text.starts_with("*/"))
        text = text.drop_front();
      if (!text.consume_front("*/")) {
        mlir::emitError(getLoc(commentStart), "unclosed comment; missing `*/`");
        return {text, TokenKind::Error};
      }
      continue;
    }

    // Nothing left to ignore.
    break;
  }

  // If there's nothing left in the input, create an end-of-file token.
  if (text.empty())
    return {text, TokenKind::Eof};

  // Snapshot the input before we do any lexing. This will allow us to mutate
  // the `text` in place.
  auto initialText = text;

  // Parse symbols.
  if (auto kind = matchSymbol2(text.data()); kind != TokenKind::Eof) {
    text = text.drop_front(2);
    return {initialText.take_front(2), kind};
  }
  if (auto kind = matchSymbol1(text.data()); kind != TokenKind::Eof) {
    text = text.drop_front(1);
    return {initialText.take_front(1), kind};
  }

  // Parse identifiers.
  if (isIdentStart(text[0])) {
    auto ident = text.take_while(isIdent);
    text = text.drop_front(ident.size());
    auto kind = TokenKind::Ident;
    if (auto it = keywordTable.find(ident); it != keywordTable.end())
      kind = it->second;
    return {ident, kind};
  }

  // Parse number literals.
  if (llvm::isDigit(text[0])) {
    auto digits = text.take_while(isIdent);
    text = text.drop_front(digits.size());
    return {digits, TokenKind::NumLit};
  }

  // If we get here we didn't recognize what's in the input text. Emit an error
  // diagnostic and produce an error token.
  auto chr = initialText.substr(0, 1);
  mlir::emitError(getLoc(chr)) << "unknown character `" << chr << "`";
  return {initialText.substr(0, 0), TokenKind::Error};
}
