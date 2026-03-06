//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Support/AsmParser.h"

using namespace silicon;

ParseResult silicon::parseSymbolVisibility(OpAsmParser &parser,
                                           StringAttr &attr) {
  StringRef visibility;
  if (failed(parser.parseOptionalKeyword(&visibility))) {
    attr = {};
    return success();
  }
  if (visibility == "public" || visibility == "private" ||
      visibility == "nested") {
    attr = parser.getBuilder().getStringAttr(visibility);
    return success();
  }
  return parser.emitError(parser.getCurrentLocation())
         << "expected 'public', 'private', or 'nested' visibility";
}

void silicon::printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                    StringAttr attr) {
  if (attr)
    p << attr.getValue() << " ";
}
