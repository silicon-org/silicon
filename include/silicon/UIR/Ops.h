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

namespace silicon {
namespace uir {

/// The phase span covered by a set of declared argument and result phases: the
/// contiguous range from the earliest to the latest of them, always including
/// phase 0.
std::pair<int32_t, int32_t>
computePhaseSpan(llvm::ArrayRef<int32_t> argPhases,
                 llvm::ArrayRef<int32_t> resultPhases);

} // namespace uir
} // namespace silicon

// Pull in the generated op interface declarations.
#include "silicon/UIR/Interfaces.h.inc"

// Pull in the generated op definitions.
#define GET_OP_CLASSES
#include "silicon/UIR/Ops.h.inc"

namespace silicon {
namespace uir {

/// The single `uir.signature` a normalized signature region exits through.
///
/// Returns null if the region does not have that shape -- if it spreads across
/// several blocks, does not end in `uir.signature`, or holds more than one of
/// them. Such a region declares its types on no single path, so there is no one
/// computation a consumer could copy.
SignatureOp getSignatureExit(mlir::Region &signature);

} // namespace uir
} // namespace silicon
