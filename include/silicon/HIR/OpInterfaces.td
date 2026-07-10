//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_OPINTERFACES_TD
#define SILICON_HIR_OPINTERFACES_TD

include "mlir/IR/OpBase.td"

def TypeOperandsOpInterface : OpInterface<"TypeOperandsOpInterface"> {
  let cppNamespace = "::silicon::hir";
  let description = [{
    An op implementing this interface carries *type operands*: operands that
    describe the type of a value instead of carrying data. Types are ordinary
    values computed one phase before the values they describe, so phase
    analysis subjects type operands to the "-1 rule": a type operand must be
    available one phase earlier than the op itself.

    Ops declare their type operands by implementing this interface; passes
    treat all operands of ops without it as plain value operands. Any new op
    with type operands must implement this interface, otherwise phase
    analysis silently applies value-operand rules to its type operands.
  }];
  let methods = [
    InterfaceMethod<[{
      Check whether the given operand of this operation is a type operand.
    }], "bool", "isTypeOperand", (ins "::mlir::OpOperand &":$operand)>,
  ];
}

#endif // SILICON_HIR_OPINTERFACES_TD
