//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_DIALECT_HIR_HIROPS_TD
#define SILICON_DIALECT_HIR_HIROPS_TD

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
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

def ConstantUnitOp : HIROp<"constant_unit", [Pure, ConstantLike]> {
  let results = (outs TypeType:$result);
  let assemblyFormat = "attr-dict";
}

def InferrableOp : HIROp<"inferrable", []> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
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

def UnifyOp : HIROp<"unify", [
  SameOperandsAndResultType,
  Commutative,
]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

def StoreOp : HIROp<"store", []> {
  let arguments = (ins TypeType:$target, TypeType:$value, TypeType:$valueType);
  let assemblyFormat = "$target `,` $value `:` $valueType attr-dict";
}

def ConstBranchOp : HIROp<"const_br", [
  DeclareOpInterfaceMethods<BranchOpInterface>,
  Pure,
  Terminator
]> {
  let arguments = (ins Variadic<AnyType>:$destOperands);
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = [{
    $dest (`(` $destOperands^ `:` type($destOperands) `)`)?
    attr-dict
  }];
}

def ConstCondBranchOp : HIROp<"const_cond_br", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<BranchOpInterface>,
  Pure,
  Terminator
]> {
  let arguments = (ins
    I1:$condition,
    Variadic<AnyType>:$trueOperands,
    Variadic<AnyType>:$falseOperands
  );
  let successors = (successor
    AnySuccessor:$trueDest,
    AnySuccessor:$falseDest
  );
  let assemblyFormat = [{
    $condition `,`
    $trueDest (`(` $trueOperands^ `:` type($trueOperands) `)`)? `,`
    $falseDest (`(` $falseOperands^ `:` type($falseOperands) `)`)?
    attr-dict
  }];
}

def ConstWrapOp : HIROp<"const_wrap", [
  Pure,
  TypesMatchWith<
    "result must be !hir.const of input type",
    "input", "result",
    [{ ConstType::get($_self.getContext(), $_self) }]
  >
]> {
  let arguments = (ins AnyType:$input);
  let results = (outs ConstType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ConstUnwrapOp : HIROp<"const_unwrap", [
  Pure,
  TypesMatchWith<
    "input must be !hir.const of result type",
    "input", "result",
    [{ cast<ConstType>($_self).getInnerType() }]
  >
]> {
  let arguments = (ins ConstType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def FuncOp : HIROp<"func", [
  IsolatedFromAbove
]> {
  let arguments = (ins SymbolNameAttr:$sym_name);
  let regions = (region AnyRegion:$signature, AnyRegion:$body);
  let assemblyFormat = "$sym_name attr-dict-with-keyword $signature $body";
}

def ArgsOp : HIROp<"args", [
  Pure,
  Terminator,
]> {
  let assemblyFormat = "attr-dict";
}

def ReturnOp : HIROp<"return", [
  HasParent<"FuncOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
}

#endif // SILICON_DIALECT_HIR_HIROPS_TD
