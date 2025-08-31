//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_TYPES_TD
#define SILICON_MIR_TYPES_TD

include "silicon/MIR/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class MIRTypeDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Type"
> : TypeDef<MIRDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def TypeType : MIRTypeDef<"Type"> {
  let mnemonic = "type";
  let summary = "a value representing a type";
}

def IntType : MIRTypeDef<"Int"> {
  let mnemonic = "int";
  let summary = "generic integer type";
}

def SpecializedFuncType : MIRTypeDef<"SpecializedFunc"> {
  let mnemonic = "specialized_func";
  let summary = "specialized function";
}

#endif // SILICON_MIR_TYPES_TD
