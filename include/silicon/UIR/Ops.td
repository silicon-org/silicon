//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_OPS_TD
#define SILICON_UIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/HIR/Types.td"
include "silicon/UIR/Dialect.td"

// Base class for the operations in this dialect.
class UIROp<string mnemonic, list<Trait> traits = []> :
  Op<UIRDialect, mnemonic, traits>;

#endif // SILICON_UIR_OPS_TD
