//===- HIRDialect.cpp - High-level IR dialect definition ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIRDialect.h"

using namespace silicon;
using namespace hir;

void HIRDialect::initialize() {
  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "silicon/Dialect/HIR/HIROps.cpp.inc"
      >();
}

// Pull in the generated dialect definition.
#include "silicon/Dialect/HIR/HIRDialect.cpp.inc"
