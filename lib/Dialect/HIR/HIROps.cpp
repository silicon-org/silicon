//===- HIROps.cpp - High-level IR operations ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Dialect/HIR/HIROps.h"
#include "Utils.h"
#include "silicon/Dialect/HIR/HIRAttributes.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

// Handle `custom<IntAttr>` parsing.
static ParseResult parseIntAttr(OpAsmParser &parser, IntAttr &value) {
  auto result = mlir::FieldParser<DynamicAPInt>::parse(parser);
  if (failed(result))
    return failure();
  value = IntAttr::get(parser.getContext(), *result);
  return success();
}

// Handle `custom<IntAttr>` printing.
static void printIntAttr(OpAsmPrinter &printer, Operation *op, IntAttr value) {
  printer << value.getValue();
}

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/Dialect/HIR/HIROps.cpp.inc"
