//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/Support/MLIR.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DynamicAPInt.h"

/// A parser for `DynamicAPInt` fields.
template <>
struct mlir::FieldParser<llvm::DynamicAPInt> {
  static FailureOr<DynamicAPInt> parse(AsmParser &parser) {
    APInt value;
    if (parser.parseInteger(value))
      return failure();
    return DynamicAPInt(value);
  }
};

namespace silicon {

// Implementation for `custom<SymbolVisibility>` assembly syntax. Parser and
// prints a symbol's visibility as `public`, `private`, or `nested`.
ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr);
void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                           StringAttr symVisibilityAttr);

} // namespace silicon
