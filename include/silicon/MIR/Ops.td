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
include "silicon/Base/Types.td"
include "silicon/MIR/Dialect.td"

// Base class for the operations in this dialect.
class MIROp<string mnemonic, list<Trait> traits = []> :
  Op<MIRDialect, mnemonic, traits>;

/// A signed or unsigned integer type.
def SiNumericType : AnyTypeOf<[IntType, UIntType]>;

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

class MIRBinaryOp<string mnemonic>
    : MIROp<mnemonic, [Pure, SameOperandsAndResultType]> {
  let arguments = (ins SiNumericType:$lhs, SiNumericType:$rhs);
  let results = (outs SiNumericType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($result)";
  let hasFolder = 1;
}

class MIRCmpOp<string mnemonic>
    : MIROp<mnemonic, [Pure, SameTypeOperands]> {
  let arguments = (ins SiNumericType:$lhs, SiNumericType:$rhs);
  let results = (outs BoolType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs)";
  let hasFolder = 1;
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
  RecursiveMemoryEffects,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "A function with materialized MIR types";
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    TypeAttrOf<FunctionType>:$function_type,
    ArrayAttr:$argNames,
    ArrayAttr:$resultNames,
    OptionalAttr<UnitAttr>:$isModule
  );
  let regions = (region AnyRegion:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let extraClassDeclaration = [{
    ReturnOp getReturnOp();
  }];
}

def BoolToI1Op : MIROp<"bool_to_i1", [Pure]> {
  let summary = "cast bool to i1 for branch conditions";
  let description = [{
    Converts an `!si.bool` value to an `i1` value for use with `cf.cond_br`.
    During interpretation, this simply forwards the `BoolAttr` unchanged.
  }];
  let arguments = (ins BoolType:$input);
  let results = (outs I1:$result);
  let assemblyFormat = "$input attr-dict";
  let hasFolder = 1;
}

def ReturnOp : MIROp<"return", [Pure, ReturnLike, Terminator]> {
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
  let hasVerifier = 1;
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
    custom<SymbolVisibility>($sym_visibility) $sym_name $results attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// Opaque Bundles
//===----------------------------------------------------------------------===//

def MIROpaquePackOp : MIROp<"opaque_pack", [Pure]> {
  let summary = "Pack multiple values into a single opaque bundle";
  let arguments = (ins Variadic<AnyType>:$operands);
  let results = (outs OpaqueType:$result);
  let assemblyFormat = [{
    `(` $operands `)` `:` `(` type($operands) `)` attr-dict
  }];
}

def MIROpaqueUnpackOp : MIROp<"opaque_unpack", [Pure]> {
  let summary = "Unpack an opaque bundle into individual values";
  let arguments = (ins OpaqueType:$input);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = [{
    $input `:` type($results) attr-dict
  }];
  let hasFolder = 1;
}

#endif // SILICON_MIR_OPS_TD
