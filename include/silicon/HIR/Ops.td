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
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "custom<IntAttr>($value) attr-dict";
}

def ConstantUnitOp : HIROp<"constant_unit", [Pure]> {
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def ConstantFuncOp : HIROp<"constant_func", [Pure]> {
  let arguments = (ins FlatSymbolRefAttr:$value, AnyHIRType:$funcType);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$value `:` $funcType attr-dict";
}

//===----------------------------------------------------------------------===//
// Type Constructors
//===----------------------------------------------------------------------===//

def UnitTypeOp : HIROp<"unit_type", [Pure]> {
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def IntTypeOp : HIROp<"int_type", [Pure]> {
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def UIntTypeOp : HIROp<"uint_type", [Pure]> {
  let arguments = (ins AnyHIRType:$width);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$width attr-dict";
}

def AnyfuncTypeOp : HIROp<"anyfunc_type", [Pure]> {
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def RefTypeOp : HIROp<"ref_type", [Pure]> {
  let arguments = (ins AnyHIRType:$innerType);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$innerType attr-dict";
}

def ConstTypeOp : HIROp<"const_type", [Pure]> {
  let arguments = (ins AnyHIRType:$innerType);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$innerType attr-dict";
}

def FuncTypeOp : HIROp<"func_type", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let arguments = (ins
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults
  );
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{
    ` ` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Lexical Grouping
//===----------------------------------------------------------------------===//

def ExprOp : HIROp<"expr", [
  RecursiveMemoryEffects,
]> {
  let summary = "Region to execute as a single expression";
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    (`:` type($results)^)? attr-dict-with-keyword $body
  }];
}

def YieldOp : HIROp<"yield", [
  HasParent<"ExprOp">,
  Pure,
  Terminator,
]> {
  let summary = "Yield result values from an expression op";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Value Typing
//===----------------------------------------------------------------------===//

def TypeOfOp : HIROp<"type_of", [Pure]> {
  let summary = "Determine the type of a value";
  let arguments = (ins AnyHIRType:$input);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{
    $input attr-dict
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def CoerceTypeOp : HIROp<"coerce_type", [Pure]> {
  let summary = "Coerce a value to a given type";
  let arguments = (ins AnyHIRType:$input, AnyHIRType:$typeOperand);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{
    $input `,` $typeOperand attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

def InferrableOp : HIROp<"inferrable", []> {
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def BinaryOp : HIROp<"binary", [
  Pure,
  SameOperandsAndResultType,
]> {
  let arguments = (ins AnyHIRType:$lhs, AnyHIRType:$rhs);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
}

def LetOp : HIROp<"let", []> {
  let arguments = (ins StrAttr:$name, AnyHIRType:$type);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$name `:` $type attr-dict";
}

def UnifyOp : HIROp<"unify", [
  SameOperandsAndResultType,
  Commutative,
]> {
  let arguments = (ins AnyHIRType:$lhs, AnyHIRType:$rhs);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def StoreOp : HIROp<"store", []> {
  let arguments = (ins AnyHIRType:$target, AnyHIRType:$value, AnyHIRType:$valueType);
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

def ConstWrapOp : HIROp<"const_wrap", [Pure]> {
  let arguments = (ins AnyHIRType:$input);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$input attr-dict";
}

def ConstUnwrapOp : HIROp<"const_unwrap", [Pure]> {
  let arguments = (ins AnyHIRType:$input);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$input attr-dict";
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

def ReturnOp : HIROp<"return", [
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"FuncOp">,
]> {
  let arguments = (ins Variadic<AnyHIRType>:$operands);
  let assemblyFormat = [{
    ($operands^)? attr-dict
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

def CallOp : HIROp<"call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  RangedTypesMatchWith<"arguments match argument types", "arguments", "typeOfArgs", "$_self">,
  RangedTypesMatchWith<"results match result types", "results", "typeOfResults", "$_self">,
]> {
  let summary = "A direct call to a named function";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyHIRType>:$arguments,
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults
  );
  let results = (outs Variadic<AnyHIRType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` `:` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)`
    attr-dict
  }];
}

def SpecializeFuncOp : HIROp<"specialize_func", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults,
    Variadic<AnyHIRType>:$consts
  );
  let results = (outs AnyHIRType);
  let assemblyFormat = [{
    $func `(` $typeOfArgs `)` `->` `(` $typeOfResults `)`
    (`,` $consts^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Constless Functions and Calls
//===----------------------------------------------------------------------===//

def UncheckedFuncOp : HIROp<"unchecked_func", [
  IsolatedFromAbove,
  Symbol,
]> {
  let summary = "A function not yet split into constness levels";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility
  );
  let regions = (region
    MinSizedRegion<1>:$signature,
    MinSizedRegion<1>:$body
  );
  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility) ``
    $sym_name attr-dict-with-keyword $signature $body
  }];
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    UncheckedSignatureOp getSignatureOp();
    UncheckedReturnOp getReturnOp();
  }];
}

def UncheckedArgOp : HIROp<"unchecked_arg", [
  HasParent<"UncheckedFuncOp">,
  Pure,
]> {
  let summary = "A function argument or result before const inference";
  let arguments = (ins StrAttr:$name, AnyHIRType:$typeOperand, I32Attr:$constness);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{
    $name `,` $typeOperand `,` $constness attr-dict
  }];
}

def UncheckedSignatureOp : HIROp<"unchecked_signature", [
  AttrSizedOperandSegments,
  HasParent<"UncheckedFuncOp">,
  Pure,
  Terminator,
]> {
  let summary = "Terminator for the signature region of an unchecked function";
  let arguments = (ins
    Variadic<AnyHIRType>:$argValues,
    Variadic<AnyHIRType>:$typeOfResults
  );
  let assemblyFormat = [{
    ` ` `(` $argValues `)` `->` `(` $typeOfResults `)` attr-dict
  }];
}

def UncheckedReturnOp : HIROp<"unchecked_return", [
  HasParent<"UncheckedFuncOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for the body region of an unchecked function";
  let arguments = (ins Variadic<AnyHIRType>:$operands);
  let assemblyFormat = [{
    ($operands^)? attr-dict
  }];
}

def UncheckedCallOp : HIROp<"unchecked_call", [
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments
  );
  let summary = "A typeless call not yet split into constness levels";
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` `:` functional-type($arguments, results)
    attr-dict
  }];
}

def CheckedCallOp : HIROp<"checked_call", [
  AttrSizedOperandSegments,
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyHIRType>:$arguments,
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults,
    DenseI32ArrayAttr:$constnessOfArgs,
    DenseI32ArrayAttr:$constnessOfResults
  );
  let summary = "A typed call not yet split into constness levels";
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` `:`
    `(` $typeOfArgs `)` `->`
    `(` $typeOfResults `)`
    `->` `(` type($results) `)`
    $constnessOfArgs
    $constnessOfResults
    attr-dict
  }];
}

#endif // SILICON_HIR_OPS_TD
