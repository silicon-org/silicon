//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/MLIR.h"
#include "silicon/Syntax/AST.h"

namespace silicon {

/// Convert an AST to MLIR.
OwningOpRef<ModuleOp> convertToIR(MLIRContext *context, AST &ast);

} // namespace silicon
