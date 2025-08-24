//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Types.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace mir;

void MIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "silicon/MIR/Types.cpp.inc"
      >();
}

// Pull in the generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "silicon/MIR/Types.cpp.inc"
