//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Utils.h"
#include "silicon/Dialect/HIR/HIRAttributes.h"
#include "silicon/Dialect/HIR/HIRDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace silicon;
using namespace hir;

using llvm::DynamicAPInt;

void HIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "silicon/Dialect/HIR/HIRAttributes.cpp.inc"
      >();
}

// Pull in the generated attribute definitions.
#define GET_ATTRDEF_CLASSES
#include "silicon/Dialect/HIR/HIRAttributes.cpp.inc"
