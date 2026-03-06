//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_MIR_OPS_TD
#define SILICON_MIR_OPS_TD

include "mlir/IR/BuiltinAttributeInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/MIR/Attributes.td"
include "silicon/MIR/Dialect.td"
include "silicon/MIR/Types.td"

// Base class for the operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  Op<MIRDialect, mnemonic, traits>;

def ConstantOp : MIROp<"constant", [
  AllTypesMatch<["value", "result"]>,
  ConstantLike,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>,
  Pure,
]> {
  let arguments = (ins TypedAttrInterface:$value);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def SpecializeFuncOp : MIROp<"specialize_func", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let arguments = (ins
    FlatSymbolRefAttr:$func,
    Variadic<TypeType>:$typeOfArgs,
    Variadic<TypeType>:$typeOfResults,
    Variadic<AnyType>:$consts
  );
  let results = (outs SpecializedFuncType);
  let assemblyFormat = [{
    $func `(` $typeOfArgs `)` `->` `(` $typeOfResults `)`
    (`,` $consts^ `:` type($consts))?
    attr-dict
  }];
  let hasFolder = 1;
  let extraClassDeclaration = [{
    SpecializedFuncAttr interpret(FoldAdaptor adaptor);
  }];
}

class MIRBinaryOp<string mnemonic>
    : MIROp<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
}

class MIRCmpOp<string mnemonic>
    : MIROp<mnemonic, [Pure, SameTypeOperands]> {
  let arguments = (ins AnyType:$lhs, AnyType:$rhs);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `->` type($result)";
}

// Arithmetic
def AddOp : MIRBinaryOp<"add">;
def SubOp : MIRBinaryOp<"sub">;
def MulOp : MIRBinaryOp<"mul">;
def DivOp : MIRBinaryOp<"div">;
def ModOp : MIRBinaryOp<"mod">;

// Bitwise
def AndOp : MIRBinaryOp<"and">;
def OrOp  : MIRBinaryOp<"or">;
def XorOp : MIRBinaryOp<"xor">;
def ShlOp : MIRBinaryOp<"shl">;
def ShrOp : MIRBinaryOp<"shr">;

// Comparison
def EqOp  : MIRCmpOp<"eq">;
def NeqOp : MIRCmpOp<"neq">;
def LtOp  : MIRCmpOp<"lt">;
def GtOp  : MIRCmpOp<"gt">;
def GeqOp : MIRCmpOp<"geq">;
def LeqOp : MIRCmpOp<"leq">;

def FuncOp : MIROp<"func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "A function with materialized MIR types";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<FunctionType>:$function_type,
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

def ReturnOp : MIROp<"return", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
}

def CallOp : MIROp<"call"> {
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments
  );
  let results = (outs Variadic<AnyType>);
  let assemblyFormat = [{
    $callee `(` $arguments `)` attr-dict
    `:` functional-type($arguments, results)
  }];
}

//===----------------------------------------------------------------------===//
// Evaluated Functions
//===----------------------------------------------------------------------===//

def EvaluatedFuncOp : MIROp<"evaluated_func", [Symbol]> {
  let summary = "a fully-evaluated compile-time function";
  let description = [{
    Represents the result of interpreting a zero-argument MIR function. The
    `results` array contains the constant attribute values that the function
    returned. The original `mir.func` is erased after producing this op.

    The specialization pass uses this to chain evaluated results into the
    next sub-function of a `hir.multiphase_func`.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$results
  );
  let assemblyFormat = [{
    custom<SymbolVisibility>($sym_visibility)
    $sym_name $results attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Opaque Bundles
//===----------------------------------------------------------------------===//

def MIROpaquePackOp : MIROp<"opaque_pack", [Pure]> {
  let summary = "Pack multiple values into a single opaque bundle";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs AnyType:$result);
  let assemblyFormat = [{
    `(` $operands `)` `:` `(` type($operands) `)` `->` type($result) attr-dict
  }];
}

def MIROpaqueUnpackOp : MIROp<"opaque_unpack", [Pure]> {
  let summary = "Unpack an opaque bundle into individual values";
  let arguments = (ins AnyType:$input);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $input `:` type($input) `->` type($results) attr-dict
  }];
}

#endif // SILICON_MIR_OPS_TD
