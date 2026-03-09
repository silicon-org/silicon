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
  let summary = "Lower ready HIR functions to MIR";
  let description = [{
    Converts `hir.func` ops to `mir.func` ops using MLIR dialect
    conversion. Only functions whose type operands are all resolvable
    to concrete MIR types are lowered; functions with unresolved
    types (e.g., containing `hir.opaque_unpack`, calls to
    `hir.multiphase_func`, or non-constant type operands) are
    skipped and left for a later iteration of the PhaseEvalLoop.

    The pass performs the following:

    1. **Readiness check (`shouldLower`):** walks each `hir.func`
       to verify that all type operands across `coerce_type`, `call`,
       `return`, `uint_type`, `func_type`, and `opaque_unpack` ops
       can be resolved to concrete types.

    2. **Type resolution:** determines argument types from
       `hir.coerce_type` ops and result types from `hir.return`'s
       type-of-values operands, recursively resolving through type
       constructors, `hir.unify`, and constant-like ops.

    3. **Signature conversion:** uses MLIR's `TypeConverter` and
       `SignatureConversion` to retype block arguments from `!hir.any`
       to concrete MIR types. The framework inserts
       `unrealized_conversion_cast` ops that body op patterns resolve
       through their adaptors.

    4. **Op conversion:** lowers HIR ops to their MIR equivalents:
       - Type constructors (`bool_type`, `int_type`, `uint_type`,
         etc.) become `mir.constant` ops with `TypeAttr` values.
       - Constants (`constant_int`, `constant_bool`, `constant_unit`)
         become `mir.constant` ops with the corresponding attrs.
       - Binary and comparison ops map to their MIR counterparts.
       - `hir.call` becomes `mir.call` with resolved result types.
       - `hir.return` becomes `mir.return`.
       - `hir.coerce_type` and `hir.unify` are folded away.
       - `hir.opaque_pack`/`hir.opaque_unpack` map to MIR equivalents.

    **Expects:** `hir.func` ops with type constraints resolved by
    InferTypes and signatures inlined by CheckCalls.

    **Produces:** `mir.func` ops with concrete types, ready for
    interpretation or further lowering.
  }];
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
