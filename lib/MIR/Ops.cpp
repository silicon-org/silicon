//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Ops.h"

using namespace mlir;
using namespace silicon;
using namespace mir;

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/MIR/Ops.cpp.inc"

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }
