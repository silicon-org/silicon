//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/UIR/Ops.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.cpp.inc"
