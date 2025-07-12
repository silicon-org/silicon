//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIRATTRIBUTES_TD
#define SILICON_DIALECT_HIR_HIRATTRIBUTES_TD

include "silicon/Dialect/HIR/HIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

class HIRAttrDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Attribute"
> : AttrDef<HIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def IntAttr : HIRAttrDef<"Int"> {
  let mnemonic = "int";
  let parameters = (ins "llvm::DynamicAPInt":$value);
  let assemblyFormat = "`<` $value `>`";
}

#endif // SILICON_DIALECT_HIR_HIRATTRIBUTES_TD
