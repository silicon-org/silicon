//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_CONVERSION_PASSES_TD
#define SILICON_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

def HIRToMIRPass : Pass<"lower-hir-to-mir", "hir::FuncOp">;

#endif // SILICON_CONVERSION_PASSES_TD
