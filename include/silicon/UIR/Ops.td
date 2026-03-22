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

//===----------------------------------------------------------------------===//
// Region Terminators
//===----------------------------------------------------------------------===//

def YieldOp : UIROp<"yield", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "yield values from a region";
  let description = [{
    Normal region completion. Returns values from a structured CF region
    or expression region to the parent op. Each value has a corresponding
    type operand, typically the result of an `hir.unify` between the
    parent op's result type and the yielded value's type.

    For void regions (e.g., loop body continuing to next iteration), the
    op carries no operands.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$values,
    Variadic<HIRAnyType>:$typeOfValues
  );
  let assemblyFormat = [{
    ($values^ `:` $typeOfValues)?
    attr-dict
  }];
}

def BreakOp : UIROp<"break", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "exit enclosing loop with values";
  let description = [{
    Exits the nearest enclosing `uir.loop`, providing the loop's result
    values. Must be transitively nested inside a `uir.loop` region
    (possibly inside `uir.if` etc.). Each value has a corresponding type
    operand, unified with the enclosing loop's result type operands.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$values,
    Variadic<HIRAnyType>:$typeOfValues
  );
  let assemblyFormat = [{
    ($values^ `:` $typeOfValues)?
    attr-dict
  }];
  let hasVerifier = 1;
}

def ContinueOp : UIROp<"continue", [
  Pure,
  Terminator,
]> {
  let summary = "restart enclosing loop iteration";
  let description = [{
    Restarts the nearest enclosing `uir.loop` iteration. Must be
    transitively nested inside a `uir.loop` region.
  }];
  let assemblyFormat = [{ attr-dict }];
  let hasVerifier = 1;
}

def ReturnOp : UIROp<"return", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "return from enclosing function";
  let description = [{
    Returns from the enclosing function. This is the structured early-exit
    counterpart to `hir.return`. Unlike `hir.return` (which terminates the
    function body's block in flat CFG), `uir.return` can appear inside
    structured CF regions at any nesting depth.

    Carries the same operand structure as `hir.return`: return values and
    their type operands.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$values,
    Variadic<HIRAnyType>:$typeOfValues
  );
  let assemblyFormat = [{
    ($values^)?
    `->` `(` $typeOfValues `)`
    attr-dict
  }];
}

def UnreachableOp : UIROp<"unreachable", [
  Pure,
  Terminator,
]> {
  let summary = "marks unreachable control flow";
  let description = [{
    Marks a point in the block that control flow never reaches. Emitted
    by codegen after structured CF ops where all regions exit early (e.g.,
    a `uir.if` where both branches use `uir.return`).

    This avoids the need for dummy return values or placeholder ops after
    exhaustive early exits.
  }];
  let assemblyFormat = [{ attr-dict }];
}

#endif // SILICON_UIR_OPS_TD
