//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_BASE_TYPES_TD
#define SILICON_BASE_TYPES_TD

include "silicon/Base/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class BaseTypeDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Type"
> : TypeDef<BaseDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def TypeType : BaseTypeDef<"Type"> {
  let mnemonic = "type";
  let summary = "a value representing a type";
}

def IntType : BaseTypeDef<"Int"> {
  let mnemonic = "int";
  let summary = "generic integer type";
}

def UIntType : BaseTypeDef<"UInt"> {
  let mnemonic = "uint";
  let summary = "sized unsigned integer type";
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
}

def UnitType : BaseTypeDef<"Unit"> {
  let mnemonic = "unit";
  let summary = "unit type";
}

def AnyfuncType : BaseTypeDef<"Anyfunc"> {
  let mnemonic = "anyfunc";
  let summary = "any function type";
}

def OpaqueType : BaseTypeDef<"Opaque"> {
  let mnemonic = "opaque";
  let summary = "opaque value bundle passed between phases";
}

#endif // SILICON_BASE_TYPES_TD
