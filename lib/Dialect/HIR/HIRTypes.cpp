//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIRDialect.h"
#include "silicon/Dialect/HIR/HIRTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace silicon;
using namespace hir;

void HIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "silicon/Dialect/HIR/HIRTypes.cpp.inc"
      >();
}

// Pull in the generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "silicon/Dialect/HIR/HIRTypes.cpp.inc"
