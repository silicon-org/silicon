//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/HIR/Attributes.h"
#include "silicon/HIR/Types.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/HIR/Ops.h.inc"

namespace silicon {
namespace hir {

/// Try to extract the type of a value as an SSA value by inspecting its
/// defining op. Returns nullptr if the type cannot be determined without
/// creating a new op.
mlir::Value getTypeOf(mlir::Value value);

/// Get the type of a value as an SSA value.
/// First tries to extract it from the defining op's type operand; falls back
/// to creating a `hir.type_of` op.
mlir::Value getOrCreateTypeOf(mlir::OpBuilder &builder, mlir::Location loc,
                              mlir::Value value);

} // namespace hir
} // namespace silicon
