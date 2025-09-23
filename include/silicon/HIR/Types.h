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

#define GET_TYPEDEF_CLASSES
#include "silicon/HIR/Types.h.inc"

namespace silicon {
namespace hir {

/// Return the next higher HIR type kind after `type`.
mlir::Type getHigherKind(mlir::Type type);

/// Return the next lower HIR type kind after `type`.
mlir::Type getLowerKind(mlir::Type type);

} // namespace hir
} // namespace silicon
