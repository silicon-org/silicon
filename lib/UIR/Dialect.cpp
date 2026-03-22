//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/UIR/Dialect.h"
#include "silicon/UIR/Ops.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace uir;

void UIRDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "silicon/UIR/Ops.cpp.inc"
      >();
}

// Pull in the generated dialect definition.
#include "silicon/UIR/Dialect.cpp.inc"
