//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_TYPES_TD
#define SILICON_HIR_TYPES_TD

include "silicon/HIR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class HIRTypeDef<string name, list<Trait> traits = [],
                     string baseCppClass = "mlir::Type">
    : TypeDef<HIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def AnyHIRType : HIRTypeDef<"Any"> {
  let mnemonic = "any";
}

#endif // SILICON_HIR_TYPES_TD
