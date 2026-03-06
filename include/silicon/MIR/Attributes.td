//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_ATTRIBUTES_TD
#define SILICON_MIR_ATTRIBUTES_TD

include "silicon/MIR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

class MIRAttrDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Attribute"
> : AttrDef<MIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def FuncAttr : MIRAttrDef<"Func", [
  TypedAttrInterface
]> {
  let mnemonic = "func";
  let parameters = (ins
    AttributeSelfTypeParameter<"", "mlir::FunctionType">:$type,
    "mlir::FlatSymbolRefAttr":$func
  );
  let assemblyFormat = "`<` $func `>`";
}

#endif // SILICON_MIR_ATTRIBUTES_TD
