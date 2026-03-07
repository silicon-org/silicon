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

def HIRToMIRPass : Pass<"lower-hir-to-mir", "mlir::ModuleOp"> {
  let dependentDialects = ["mir::MIRDialect"];
}

def MIRToCIRCTPass : Pass<"lower-mir-to-circt", "mlir::ModuleOp"> {
  let summary = "Lower MIR module functions to CIRCT hw/comb dialects";
  let description = [{
    Converts `mir.func` ops marked `isModule` and their transitive callees
    to `hw.module` ops with `comb` combinational logic. All remaining
    Silicon dialect ops (HIR, MIR, Base) are erased by the conversion.

    Type mapping:
    - `!si.uint<N>` becomes `iN`
    - `!si.int` becomes `i64` (temporary until bitwidth inference exists)
    - `FunctionType` is converted recursively
    - Standard `IntegerType` passes through
  }];
  let dependentDialects = [
    "circt::hw::HWDialect",
    "circt::comb::CombDialect"
  ];
}

#endif // SILICON_CONVERSION_PASSES_TD
