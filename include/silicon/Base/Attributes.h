//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DynamicAPInt.h"

#define GET_ATTRDEF_CLASSES
#include "silicon/Base/Attributes.h.inc"

namespace silicon::base {

/// Extract the integer value from an IntAttr or UIntAttr.
inline llvm::DynamicAPInt getIntValue(mlir::Attribute attr) {
  if (auto intAttr = mlir::dyn_cast<IntAttr>(attr))
    return intAttr.getValue();
  return mlir::cast<UIntAttr>(attr).getValue();
}

} // namespace silicon::base
