//===- HIRDialect.td - High-level IR dialect definition -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIRDIALECT_TD
#define SILICON_DIALECT_HIR_HIRDIALECT_TD

include "mlir/IR/DialectBase.td"

def HIRDialect : Dialect {
  let name = "hir";
  let cppNamespace = "silicon::hir";

  let summary = "Untyped high-level Silicon IR";
  let description = [{
    The HIR dialect defines operations and types to represent a Silicon design
    during constant evaluation and type checking.
  }];

  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    void registerTypes();
    void registerAttributes();
  }];
}

#endif // SILICON_DIALECT_HIR_HIRDIALECT_TD
