//===- HIRTypes.td - High-level IR types ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIRTYPES_TD
#define SILICON_DIALECT_HIR_HIRTYPES_TD

include "silicon/Dialect/HIR/HIRDialect.td"
include "mlir/IR/AttrTypeBase.td"

class HIRTypeDef<string name, list<Trait> traits = [],
                     string baseCppClass = "mlir::Type">
    : TypeDef<HIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def TypeType : HIRTypeDef<"Type"> {
  let mnemonic = "type";
}

#endif // SILICON_DIALECT_HIR_HIRTYPES_TD
