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

def DynamicAPIntParameter : AttrOrTypeParameter<"llvm::DynamicAPInt", ""> {
  let comparator = "$_lhs == $_rhs";
}

class MIRAttrDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Attribute"
> : AttrDef<MIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def MIRTypeAttr : MIRAttrDef<"Type", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "type";
  let parameters = (ins "mlir::Type":$value);
  let assemblyFormat = "`<` $value `>`";
}

def IntAttr : MIRAttrDef<"Int", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "int";
  let parameters = (ins DynamicAPIntParameter:$value);
  let assemblyFormat = "`<` $value `>`";
}

def FuncAttr : MIRAttrDef<"Func", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "func";
  let parameters = (ins "mlir::FlatSymbolRefAttr":$func);
  let assemblyFormat = "`<` $func `>`";
}

def SpecializedFuncAttr : MIRAttrDef<"SpecializedFunc", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "specialized_func";
  let parameters = (ins
    "mlir::FlatSymbolRefAttr":$func,
    OptionalArrayRefParameter<"mlir::Type">:$args,
    OptionalArrayRefParameter<"mlir::Type">:$results,
    OptionalArrayRefParameter<"mlir::Attribute">:$consts
  );
  // The quirky syntax is necessary since `ArrayRefParameters` don't parse
  // properly if they are empty.
  let assemblyFormat = [{
    `<` $func `,`
    `[` (`]`) : ($args^ `]`)? `,`
    `[` (`]`) : ($results^ `]`)? `,`
    `[` (`]`) : ($consts^ `]`)? `>`
  }];
}

#endif // SILICON_MIR_ATTRIBUTES_TD
