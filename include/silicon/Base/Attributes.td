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
  let summary = "a constant type value";
  let description = [{
    An attribute that wraps a type as a constant value. This is used to
    represent the result of compile-time type computations, for example
    when the interpreter evaluates a function that returns a type. The
    attribute has `si.type` as its type.

    Example: `#si.type<!si.uint<8>>` represents the constant type value
    `!si.uint<8>`.
  }];
  let parameters = (ins "mlir::Type":$value);
  let assemblyFormat = "`<` $value `>`";
}

def IntAttr : BaseAttrDef<"Int", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "int";
  let summary = "a constant integer value";
  let description = [{
    An attribute representing a constant arbitrary-precision integer
    value. This is the constant form of `si.int` and is used to carry
    integer results across compilation phases. The attribute has
    `si.int` as its type.

    Example: `#si.int<42>`.
  }];
  let parameters = (ins DynamicAPIntParameter:$value);
  let assemblyFormat = "`<` $value `>`";
}

def BaseUnitAttr : BaseAttrDef<"Unit", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "unit";
  let summary = "a constant unit value";
  let description = [{
    An attribute representing the unit value, the single inhabitant of
    `si.unit`. Used as the result of functions that return no meaningful
    value.
  }];
  let assemblyFormat = "";
}

def SiBoolAttr : BaseAttrDef<"Bool", [
  DeclareAttrInterfaceMethods<TypedAttrInterface>
]> {
  let mnemonic = "bool";
  let summary = "a constant boolean value";
  let description = [{
    An attribute representing a constant boolean value. This is the
    constant form of `si.bool` and has `si.bool` as its type.

    Example: `#si.bool<true>`.
  }];
  let parameters = (ins "bool":$value);
  let assemblyFormat = "`<` $value `>`";
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
