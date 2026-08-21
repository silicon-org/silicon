//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/HIR/Types.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.h.inc"

namespace silicon {
namespace uir {

/// The declared argument and result phases of a UIR function symbol.
struct DeclaredPhases {
  llvm::ArrayRef<int32_t> argPhases;
  llvm::ArrayRef<int32_t> resultPhases;

  /// The declared phase span: the contiguous range from the earliest to the
  /// latest declared argument or result phase, always including phase 0.
  std::pair<int32_t, int32_t> span() const;
};

/// Get the declared phases of a `uir.func` or `uir.split_func`, or `nullopt`
/// for any other operation.
///
/// Both ops declare a phase for every argument and result, and `uir.call`
/// targets them interchangeably: a callee may already be split when its callers
/// are verified or sorted.
std::optional<DeclaredPhases> getDeclaredPhases(mlir::Operation *op);

} // namespace uir
} // namespace silicon
