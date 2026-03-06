//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_PASSES_TD
#define SILICON_HIR_PASSES_TD

include "silicon/HIR/Dialect.td"
include "mlir/Pass/PassBase.td"

def InferTypesPass : Pass<"infer-types">;
def EvalConstsPass : Pass<"eval-consts", "mlir::func::FuncOp">;

def CheckCallsPass : Pass<"check-calls", "mlir::ModuleOp"> {
  let summary = "Move function signature constraints to call sites";
}

def SplitPhasesPass : Pass<"split-phases", "mlir::ModuleOp"> {
  let summary = "Split functions into multiple phases based on constness";
}

def SpecializeFuncsPass : Pass<"specialize-funcs", "mlir::ModuleOp"> {
  let summary = "Specialize functions with evaluated compile-time results";
  let description = [{
    Performs two kinds of specialization:

    1. **Multiphase chaining**: when the first sub-function of a
       `multiphase_func` has been evaluated (`mir.evaluated_func` exists),
       chain its results into the next sub-function by replacing the opaque
       context arg with `hir.mir_constant` ops. Dissolve the
       `multiphase_func` when only one sub-function remains.

    2. **Transitive call-site specialization**: after chaining, walk the
       specialized function's body for `hir.call` ops that pass
       `hir.mir_constant` opaque values as arguments. Clone the callee,
       inline the constant, expand the `opaque_unpack`, and update the call.
  }];
  let dependentDialects = [
    "silicon::mir::MIRDialect",
  ];
}

#endif // SILICON_HIR_PASSES_TD
