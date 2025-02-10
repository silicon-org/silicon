//===- HIROps.td - High-level IR operations -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIROPS_TD
#define SILICON_DIALECT_HIR_HIROPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/Dialect/HIR/HIRAttributes.td"
include "silicon/Dialect/HIR/HIRDialect.td"
include "silicon/Dialect/HIR/HIRTypes.td"

// Base class for the operations in this dialect.
class HIROp<string mnemonic, list<Trait> traits = []> :
  Op<HIRDialect, mnemonic, traits>;

def ConstantIntOp : HIROp<"constant_int", [Pure, ConstantLike]> {
  let arguments = (ins IntAttr:$value);
  let results = (outs IntType:$result);
  let assemblyFormat = "custom<IntAttr>($value) attr-dict";
}

def InferrableTypeOp : HIROp<"inferrable_type", []> {
  let results = (outs TypeType:$result);
  let assemblyFormat = "attr-dict";
}

def IntTypeOp : HIROp<"int_type", [Pure, ConstantLike]> {
  let results = (outs TypeType:$result);
  let assemblyFormat = "attr-dict";
}

def RefTypeOp : HIROp<"ref_type", [Pure]> {
  let arguments = (ins TypeType:$innerType);
  let results = (outs TypeType:$result);
  let assemblyFormat = "$innerType attr-dict";
}

def LetOp : HIROp<"let", []> {
  let arguments = (ins StrAttr:$name, TypeType:$type);
  let results = (outs TypeType:$result);
  let assemblyFormat = "$name `:` $type attr-dict";
}

def UnifyTypeOp : HIROp<"unify_type", []> {
  let arguments = (ins TypeType:$lhs, TypeType:$rhs);
  let results = (outs TypeType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def StoreOp : HIROp<"store", []> {
  let arguments = (ins TypeType:$target, TypeType:$value, TypeType:$valueType);
  let assemblyFormat = "$target `,` $value `:` $valueType attr-dict";
}

#endif // SILICON_DIALECT_HIR_HIROPS_TD
