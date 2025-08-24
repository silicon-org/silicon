//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Dialect.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace mir;

void MIRDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "silicon/MIR/Ops.cpp.inc"
      >();
}

// Pull in the generated dialect definition.
#include "silicon/MIR/Dialect.cpp.inc"
