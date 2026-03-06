//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_BASE_ATTRIBUTES_TD
#define SILICON_BASE_ATTRIBUTES_TD

include "silicon/Base/Dialect.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinAttributeInterfaces.td"

def DynamicAPIntParameter : AttrOrTypeParameter<"llvm::DynamicAPInt", ""> {
  let comparator = "$_lhs == $_rhs";
}

class BaseAttrDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Attribute"
> : AttrDef<BaseDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def BaseTypeAttr : BaseAttrDef<"Type", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "type";
  let parameters = (ins "mlir::Type":$value);
  let assemblyFormat = "`<` $value `>`";
}

def IntAttr : BaseAttrDef<"Int", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "int";
  let parameters = (ins DynamicAPIntParameter:$value);
  let assemblyFormat = "`<` $value `>`";
}

def BaseUnitAttr : BaseAttrDef<"Unit", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "unit";
  let assemblyFormat = "";
}

def OpaqueAttr : BaseAttrDef<"Opaque", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "opaque";
  let summary = "opaque bundle of values passed between phases";
  let parameters = (ins
    OptionalArrayRefParameter<"mlir::Attribute">:$elements
  );
  let assemblyFormat = [{
    `<` `[` (`]`) : ($elements^ `]`)? `>`
  }];
}

#endif // SILICON_BASE_ATTRIBUTES_TD
