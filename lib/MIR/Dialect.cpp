//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Dialect.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/DialectImplementation.h"

using namespace silicon;
using namespace mir;

void MIRDialect::initialize() {
  // Register types and attributes.
  registerTypes();
  registerAttributes();

  // Register operations.
  addOperations<
#define GET_OP_LIST
#include "silicon/MIR/Ops.cpp.inc"
      >();
}

Operation *MIRDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  if (auto typedValue = dyn_cast<mlir::TypedAttr>(value);
      typedValue && type == typedValue.getType())
    return ConstantOp::create(builder, loc, type, typedValue);
  return nullptr;
}

// Pull in the generated dialect definition.
#include "silicon/MIR/Dialect.cpp.inc"
