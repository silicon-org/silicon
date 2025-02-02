//===- HIROps.cpp - High-level IR operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIROps.h"

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/Dialect/HIR/HIROps.cpp.inc"
