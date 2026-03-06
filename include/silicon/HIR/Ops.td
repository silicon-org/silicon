//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_HIR_OPS_TD
#define SILICON_HIR_OPS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
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

def TypeTypeOp : HIROp<"type_type", [Pure]> {
  let summary = "Construct the metatype 'Type'";
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

class HIRBinaryOp<string mnemonic>
    : HIROp<mnemonic, [Pure]> {
  let arguments = (ins AnyHIRType:$lhs, AnyHIRType:$rhs, AnyHIRType:$resultType);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` $resultType attr-dict";
}

// Arithmetic
def AddOp : HIRBinaryOp<"add">;
def SubOp : HIRBinaryOp<"sub">;
def MulOp : HIRBinaryOp<"mul">;
def DivOp : HIRBinaryOp<"div">;
def ModOp : HIRBinaryOp<"mod">;

// Bitwise
def AndOp : HIRBinaryOp<"and">;
def OrOp  : HIRBinaryOp<"or">;
def XorOp : HIRBinaryOp<"xor">;
def ShlOp : HIRBinaryOp<"shl">;
def ShrOp : HIRBinaryOp<"shr">;

// Comparison
def EqOp  : HIRBinaryOp<"eq">;
def NeqOp : HIRBinaryOp<"neq">;
def LtOp  : HIRBinaryOp<"lt">;
def GtOp  : HIRBinaryOp<"gt">;
def GeqOp : HIRBinaryOp<"geq">;
def LeqOp : HIRBinaryOp<"leq">;

def LetOp : HIROp<"let", []> {
  let arguments = (ins StrAttr:$name, AnyHIRType:$type);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "$name `:` $type attr-dict";
}

def UnifyOp : HIROp<"unify", [
  SameOperandsAndResultType,
  Commutative,
  Pure,
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

def FuncOp : HIROp<"func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$argNames,
    ArrayAttr:$resultNames
  );
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ReturnOp getReturnOp();
  }];
}

def ReturnOp : HIROp<"return", [
  AttrSizedOperandSegments,
  PredOpTrait<"values and typeOfValues must have the same size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"FuncOp">,
]> {
  let arguments = (ins
    Variadic<AnyHIRType>:$values,
    Variadic<AnyHIRType>:$typeOfValues
  );
  let assemblyFormat = [{
    (`(` $values^ `)` `:` `(` $typeOfValues `)`)? attr-dict
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

//===----------------------------------------------------------------------===//
// Unified Functions and Calls
//===----------------------------------------------------------------------===//

def UnifiedFuncOp : HIROp<"unified_func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "A function combining all phases before splitting";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    DenseI32ArrayAttr:$argPhases,
    DenseI32ArrayAttr:$resultPhases,
    ArrayAttr:$argNames,
    ArrayAttr:$resultNames
  );
  let regions = (region
    MinSizedRegion<1>:$signature,
    MinSizedRegion<1>:$body
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    UnifiedSignatureOp getSignatureOp();
    UnifiedReturnOp getReturnOp();
  }];
}

def UnifiedSignatureOp : HIROp<"unified_signature", [
  AttrSizedOperandSegments,
  HasParent<"UnifiedFuncOp">,
  Pure,
  Terminator,
]> {
  let summary = "Terminator for the signature region of a unified function";
  let arguments = (ins
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults
  );
  let assemblyFormat = [{
    ` ` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)` attr-dict
  }];
}

def UnifiedReturnOp : HIROp<"unified_return", [
  AttrSizedOperandSegments,
  PredOpTrait<"values and typeOfValues must have the same size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  HasParent<"UnifiedFuncOp">,
  Pure,
  ReturnLike,
  Terminator,
]> {
  let summary = "Terminator for the body region of a unified function";
  let arguments = (ins
    Variadic<AnyHIRType>:$values,
    Variadic<AnyHIRType>:$typeOfValues
  );
  let assemblyFormat = [{
    (`(` $values^ `)` `:` `(` $typeOfValues `)`)? attr-dict
  }];
}

def UnifiedCallOp : HIROp<"unified_call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "A call not yet split into phases";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments,
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults,
    DenseI32ArrayAttr:$argPhases,
    DenseI32ArrayAttr:$resultPhases
  );
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $callee `(` $arguments `)` `:`
    `(` $typeOfArgs `)` `->` `(` $typeOfResults `)`
    functional-type($arguments, results)
    $argPhases `->` $resultPhases
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Split Functions and Phase Management
//===----------------------------------------------------------------------===//

def SignatureOp : HIROp<"signature", [
  AttrSizedOperandSegments,
  HasParent<"SplitFuncOp">,
  Pure,
  Terminator,
]> {
  let summary = "Terminator for the signature region of a split function";
  let arguments = (ins
    Variadic<AnyHIRType>:$typeOfArgs,
    Variadic<AnyHIRType>:$typeOfResults
  );
  let assemblyFormat = [{
    ` ` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)` attr-dict
  }];
}

def SplitFuncOp : HIROp<"split_func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "Records how a unified function was split into per-phase functions";
  let description = [{
    A `hir.split_func` records the result of splitting a `hir.unified_func`
    into separate per-phase functions. It preserves the original function's
    signature (for library callers) and maps phase numbers to the generated
    per-phase `hir.func` ops.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$argNames,
    DenseI32ArrayAttr:$argPhases,
    ArrayAttr:$resultNames,
    DenseI32ArrayAttr:$resultPhases,
    DenseI32ArrayAttr:$phaseNumbers,
    ArrayAttr:$phaseFuncs
  );
  let regions = (region MinSizedRegion<1>:$signature);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
  let extraClassDeclaration = [{
    SignatureOp getSignatureOp();
  }];
}

def MultiphaseFuncOp : HIROp<"multiphase_func", [
  Symbol,
]> {
  let summary = "A chain of sub-functions evaluated iteratively, one phase at a time";
  let description = [{
    A `hir.multiphase_func` represents a single phase of a split function that
    itself spans multiple internal sub-phases. The sub-functions are evaluated
    in order. Arguments are marked `first` or `last` to indicate whether they
    are available from the first or last sub-phase.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$argNames,
    DenseBoolArrayAttr:$argIsFirst,
    ArrayAttr:$resultNames,
    ArrayAttr:$phaseFuncs
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Opaque Bundles
//===----------------------------------------------------------------------===//

def OpaqueTypeOp : HIROp<"opaque_type", [Pure]> {
  let summary = "Type constructor for opaque bundles";
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = "attr-dict";
}

def OpaquePackOp : HIROp<"opaque_pack", [Pure]> {
  let summary = "Pack multiple values into a single opaque bundle";
  let arguments = (ins Variadic<AnyHIRType>:$operands);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{ `(` $operands `)` attr-dict }];
}

def OpaqueUnpackOp : HIROp<"opaque_unpack", [Pure]> {
  let summary = "Unpack an opaque bundle into individual values";
  let arguments = (ins AnyHIRType:$input);
  let results = (outs Variadic<AnyHIRType>:$results);
  let assemblyFormat = [{ $input attr-dict `:` type($results) }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// MIR Constant Injection
//===----------------------------------------------------------------------===//

def MIRConstantOp : HIROp<"mir_constant", [ConstantLike, Pure]> {
  let summary = "materialize an MIR attribute as an HIR value";
  let description = [{
    Injects a concrete MIR attribute (produced by interpretation) into an HIR
    function during specialization. Lowered to `mir.constant` by HIR-to-MIR.
  }];
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyHIRType:$result);
  let assemblyFormat = [{
    $value attr-dict
  }];
  let hasFolder = 1;
}

#endif // SILICON_HIR_OPS_TD
