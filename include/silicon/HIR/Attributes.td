//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_ATTRIBUTES_TD
#define SILICON_HIR_ATTRIBUTES_TD

include "silicon/HIR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

def DynamicAPIntParameter : AttrOrTypeParameter<"llvm::DynamicAPInt", ""> {
  let comparator = "$_lhs == $_rhs";
}

class HIRAttrDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Attribute"
> : AttrDef<HIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def IntAttr : HIRAttrDef<"Int"> {
  let mnemonic = "int";
  let parameters = (ins DynamicAPIntParameter:$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // SILICON_HIR_ATTRIBUTES_TD
