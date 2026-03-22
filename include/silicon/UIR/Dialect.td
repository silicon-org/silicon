//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_DIALECT_TD
#define SILICON_UIR_DIALECT_TD

include "mlir/IR/DialectBase.td"

def UIRDialect : Dialect {
  let name = "uir";
  let cppNamespace = "silicon::uir";

  let summary = "Unified IR for structured, pre-lowering Silicon programs";
  let description = [{
    The UIR dialect captures the high-level, structured representation of
    Silicon programs before phase splitting and CFG flattening. It holds
    unified (cross-phase) functions, structured control flow with nested
    regions, expression grouping for phase inference, and witness ops that
    bridge the unified and split representations.

    All UIR ops are lowered away by two passes:
    - **SplitPhases** consumes unified function ops and phase grouping ops.
    - **FlattenCF** consumes structured control flow ops and their terminators.

    See `docs/design/unified-dialect.md` for the full design.
  }];

  // UIR ops reference !hir.any in their tablegen constraints, but don't
  // programmatically create HIR types or Base attributes at runtime.
  // No dependentDialects needed.
}

#endif // SILICON_UIR_DIALECT_TD
