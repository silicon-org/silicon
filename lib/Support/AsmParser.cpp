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
  auto loc = parser.getCurrentLocation();
  if (failed(parser.parseOptionalKeyword(&visibility)) ||
      visibility == "public") {
    attr = {};
    return success();
  }
  if (visibility == "private" || visibility == "nested") {
    attr = parser.getBuilder().getStringAttr(visibility);
    return success();
  }
  return parser.emitError(loc)
         << "expected 'public', 'private', or 'nested' visibility";
}

void silicon::printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                                    StringAttr attr) {
  if (attr && attr != "public")
    p << attr.getValue() << " ";
}
