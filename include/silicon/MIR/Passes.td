//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_PASSES_TD
#define SILICON_MIR_PASSES_TD

include "silicon/MIR/Dialect.td"
include "mlir/Pass/PassBase.td"

def InterpretPass : Pass<"interpret", "mlir::ModuleOp"> {
  let summary = "Interpret zero-argument MIR functions";
  let description = [{
    Evaluates zero-argument MIR functions by walking their operations and
    computing constant results. Produces `mir.evaluated_func` ops with the
    constant result attributes. The original `mir.func` is erased.

    Chaining and specialization are handled separately by the
    SpecializeFuncs pass.
  }];
  let dependentDialects = [
    "mir::MIRDialect",
  ];
}

#endif // SILICON_MIR_PASSES_TD
