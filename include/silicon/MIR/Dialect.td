//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_DIALECT_TD
#define SILICON_MIR_DIALECT_TD

include "mlir/IR/DialectBase.td"

def MIRDialect : Dialect {
  let name = "mir";
  let cppNamespace = "silicon::mir";

  let summary = "Typed mid-level Silicon IR";
  let description = [{
    The MIR dialect defines operations and types to represent a Silicon design
    with all types materialized as MLIR types.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // SILICON_MIR_DIALECT_TD
