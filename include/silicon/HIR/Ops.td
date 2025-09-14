//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_OPS_TD
#define SILICON_HIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/HIR/Attributes.td"
include "silicon/HIR/Dialect.td"
include "silicon/HIR/Types.td"

// Base class for the operations in this dialect.
class HIROp<string mnemonic, list<Trait> traits = []> :
  Op<HIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ConstantIntOp : HIROp<"constant_int", [Pure]> {
  let arguments = (ins IntAttr:$value);
  let results = (outs ValueType:$result);
  let assemblyFormat = "custom<IntAttr>($value) attr-dict";
}

def ConstantUnitOp : HIROp<"constant_unit", [Pure]> {
  let results = (outs ValueType:$result);
  let assemblyFormat = "attr-dict";
}

def ConstantFuncOp : HIROp<"constant_func", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$value);
  let results = (outs FuncType:$result);
  let assemblyFormat = "$value attr-dict";
}

//===----------------------------------------------------------------------===//
// Type Constructors
//===----------------------------------------------------------------------===//

def IntTypeOp : HIROp<"int_type", [Pure]> {
  let results = (outs TypeType:$result);
  let assemblyFormat = "attr-dict";
}

def UIntTypeOp : HIROp<"uint_type", [Pure]> {
  let arguments = (ins ValueType:$width);
  let results = (outs TypeType:$result);
  let assemblyFormat = "$width attr-dict";
}

def RefTypeOp : HIROp<"ref_type", [Pure]> {
  let arguments = (ins TypeType:$innerType);
  let results = (outs TypeType:$result);
  let assemblyFormat = "$innerType attr-dict";
}

def FuncTypeOp : HIROp<"func_type", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let arguments = (ins
    Variadic<AnyType>:$typeOfArgs,
    Variadic<AnyType>:$typeOfResults
  );
  let results = (outs TypeType:$result);
  let assemblyFormat = [{
    ` `
    `(` ($typeOfArgs^ `:` type($typeOfArgs))? `)` `->`
    `(` ($typeOfResults^ `:` type($typeOfResults))? `)`
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def InferrableOp : HIROp<"inferrable", []> {
  let results = (outs AnyType:$result);
  let assemblyFormat = "attr-dict `:` type($result)";
}

def BinaryOp : HIROp<"binary", [
  Pure,
  SameOperandsAndResultType,
]> {
  let arguments = (ins ValueType:$lhs, ValueType:$rhs);
  let results = (outs ValueType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
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

def ConstOp : HIROp<"const", [
  RecursiveMemoryEffects,
]> {
  let arguments = (ins Variadic<ConstType>:$operands);
  let results = (outs Variadic<ConstType>:$results);
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? (`->` type($results)^)?
    attr-dict-with-keyword $body
  }];
}

def YieldOp : HIROp<"yield", [
  HasParent<"ConstOp">,
  Pure,
  Terminator,
]> {
  let arguments = (ins Variadic<AnyType>:$values);
  let assemblyFormat = "($values^ `:` type($values))? attr-dict";
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
  IsolatedFromAbove,
  Symbol,
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region AnyRegion:$body);
  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) ``
    $sym_name attr-dict-with-keyword $body
  }];
}

def ArgOp : HIROp<"arg", [
  HasParent<"FuncOp">,
  Pure,
  TypesMatchWith<
    "input must be const type of result",
    "type", "result",
    [{ getLowerKind(cast<ConstType>($_self).getInnerType()) }]
  >,
]> {
  let arguments = (ins StrAttr:$name, ConstType:$type);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$name `:` $type `:` type($type) attr-dict";
}

def ArgsOp : HIROp<"args", [
  HasParent<"FuncOp">,
  Pure,
  Terminator,
]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = "($operands^ `:` type($operands))? attr-dict";
}

def ReturnOp : HIROp<"return", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"FuncOp">,
]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
}

def NextPhaseOp : HIROp<"next_phase", [
  HasParent<"FuncOp">,
  Pure,
  Terminator,
]> {
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
}

def CallOp : HIROp<"call"> {
  let arguments = (ins
    FuncType:$callee,
    TypeType:$calleeType,
    Variadic<AnyType>:$arguments
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` attr-dict
    `:` $calleeType `,` functional-type($arguments, $results)
  }];
}

def SpecializeFuncOp : HIROp<"specialize_func", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<TypeType>:$typeOfArgs,
    Variadic<TypeType>:$typeOfResults,
    Variadic<AnyType>:$consts
  );
  let results = (outs FuncType);
  let assemblyFormat = [{
    $func `(` $typeOfArgs `)` `->` `(` $typeOfResults `)`
    (`,` $consts^ `:` type($consts))?
    attr-dict
  }];
}

#endif // SILICON_HIR_OPS_TD
