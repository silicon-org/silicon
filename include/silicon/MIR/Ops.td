//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_OPS_TD
#define SILICON_MIR_OPS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/MIR/Attributes.td"
include "silicon/MIR/Dialect.td"
include "silicon/MIR/Types.td"

// Base class for the operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  Op<MIRDialect, mnemonic, traits>;

def ConstantOp : MIROp<"constant", [
  AllTypesMatch<["value", "result"]>,
  ConstantLike,
  Pure,
]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

#endif // SILICON_MIR_OPS_TD
