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

#endif // SILICON_HIR_PASSES_TD
