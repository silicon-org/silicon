//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Types.h"
#include "silicon/Support/AsmParser.h"
#include "silicon/Support/MLIR.h"

using namespace silicon;
using namespace mir;

void MIRDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "silicon/MIR/Attributes.cpp.inc"
      >();
}

// Pull in the generated attribute definitions.
#define GET_ATTRDEF_CLASSES
#include "silicon/MIR/Attributes.cpp.inc"

//===----------------------------------------------------------------------===//
// TypeAttr
//===----------------------------------------------------------------------===//

Type TypeAttr::getType() const { return TypeType::get(getContext()); }

//===----------------------------------------------------------------------===//
// IntAttr
//===----------------------------------------------------------------------===//

Type IntAttr::getType() const { return IntType::get(getContext()); }

//===----------------------------------------------------------------------===//
// FuncAttr
//===----------------------------------------------------------------------===//

Type FuncAttr::getType() const {
  return SpecializedFuncType::get(getContext());
}

//===----------------------------------------------------------------------===//
// SpecializedFuncAttr
//===----------------------------------------------------------------------===//

Type SpecializedFuncAttr::getType() const {
  return SpecializedFuncType::get(getContext());
}
