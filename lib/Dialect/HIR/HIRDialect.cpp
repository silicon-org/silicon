//===- HIRDialect.cpp - High-level IR dialect definition ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIRDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "silicon/Dialect/HIR/HIRAttributes.h"
#include "silicon/Dialect/HIR/HIROps.h"
#include "silicon/Dialect/HIR/HIRTypes.h"

using namespace silicon;
using namespace hir;

void HIRDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "silicon/Dialect/HIR/HIROps.cpp.inc"
      >();
}

// Pull in the generated dialect definition.
#include "silicon/Dialect/HIR/HIRDialect.cpp.inc"
