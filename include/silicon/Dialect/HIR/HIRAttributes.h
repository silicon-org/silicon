//===- HIRAttributes.h - High-level IR attributes -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "llvm/ADT/DynamicAPInt.h"

#define GET_ATTRDEF_CLASSES
#include "silicon/Dialect/HIR/HIRAttributes.h.inc"
