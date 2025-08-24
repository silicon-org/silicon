//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Attributes.h"
#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Types.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace hir;

void HIRDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "silicon/HIR/Ops.cpp.inc"
      >();
}

// Pull in the generated dialect definition.
#include "silicon/HIR/Dialect.cpp.inc"
