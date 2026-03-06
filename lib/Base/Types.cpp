//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Dialect.h"
#include "silicon/Base/Types.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace base;

void BaseDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "silicon/Base/Types.cpp.inc"
      >();
}

// Pull in the generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "silicon/Base/Types.cpp.inc"
