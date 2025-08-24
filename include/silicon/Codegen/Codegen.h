//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Support/MLIR.h"
#include "silicon/Syntax/AST.h"

namespace silicon {

/// Convert an AST to MLIR.
OwningOpRef<ModuleOp> convertToIR(MLIRContext *context, AST &ast);

} // namespace silicon
