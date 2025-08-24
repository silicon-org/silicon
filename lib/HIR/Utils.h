//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DynamicAPInt.h"

template <>
struct mlir::FieldParser<llvm::DynamicAPInt> {
  static FailureOr<DynamicAPInt> parse(AsmParser &parser) {
    APInt value;
    if (parser.parseInteger(value))
      return failure();
    // HACK: At the time of writing there is no way to construct a DynamicAPInt
    // from an APInt. However, the data layout of DynamicAPInt is the same as
    // APInt, so we can just reinterpret_cast it.
    return *reinterpret_cast<DynamicAPInt *>(&value);
  }
};
