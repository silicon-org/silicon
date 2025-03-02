//===- HIRPasses.td - High-level IR transformation passes -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIRPASSES_TD
#define SILICON_DIALECT_HIR_HIRPASSES_TD

include "silicon/Dialect/HIR/HIRDialect.td"
include "mlir/Pass/PassBase.td"

def InferTypesPass : Pass<"infer-types">;
def EvalConstsPass : Pass<"eval-consts", "mlir::func::FuncOp">;

#endif // SILICON_DIALECT_HIR_HIRPASSES_TD
