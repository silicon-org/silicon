//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/Base/Dialect.h"
#include "silicon/Base/Types.h"
#include "silicon/Support/AsmParser.h"
#include "silicon/Support/MLIR.h"

using namespace silicon;
using namespace base;

void BaseDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "silicon/Base/Attributes.cpp.inc"
      >();
}

// Pull in the generated attribute definitions.
#define GET_ATTRDEF_CLASSES
#include "silicon/Base/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

Type TypeAttr::getType() const { return TypeType::get(getContext()); }

//===----------------------------------------------------------------------===//
// IntAttr
//===----------------------------------------------------------------------===//

Type IntAttr::getType() const { return IntType::get(getContext()); }

//===----------------------------------------------------------------------===//
// UnitAttr
//===----------------------------------------------------------------------===//

Type UnitAttr::getType() const { return UnitType::get(getContext()); }

//===----------------------------------------------------------------------===//
// OpaqueAttr
//===----------------------------------------------------------------------===//

Type OpaqueAttr::getType() const { return OpaqueType::get(getContext()); }
