//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_TRANSFORMS_PASSES_TD
#define SILICON_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def InterpretPass : Pass<"interpret", "mlir::ModuleOp"> {
  let dependentDialects = [
    "hir::HIRDialect",
    "mir::MIRDialect",
  ];
}

def SpecializeFuncsPass : Pass<"specialize-funcs", "mlir::ModuleOp"> {
  let dependentDialects = [
    "hir::HIRDialect",
    "mir::MIRDialect",
  ];
}

#endif // SILICON_TRANSFORMS_PASSES_TD
