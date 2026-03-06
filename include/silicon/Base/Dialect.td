//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_BASE_DIALECT_TD
#define SILICON_BASE_DIALECT_TD

include "mlir/IR/DialectBase.td"

def BaseDialect : Dialect {
  let name = "si";
  let cppNamespace = "silicon::base";

  let summary = "Shared Silicon types and attributes";
  let description = [{
    The Base dialect (`si`) defines types and attributes shared between the
    HIR and MIR dialects, such as integer types, unit types, type-valued
    attributes, and opaque phase-transfer bundles. Both HIR and MIR depend
    on this dialect instead of on each other.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // SILICON_BASE_DIALECT_TD
