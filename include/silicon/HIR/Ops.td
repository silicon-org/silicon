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
include "silicon/Base/Attributes.td"
include "silicon/HIR/Dialect.td"
include "silicon/HIR/Types.td"

// Base class for the operations in this dialect.
class HIROp<string mnemonic, list<Trait> traits = []> :
  Op<HIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ConstantIntOp : HIROp<"constant_int", [Pure]> {
  let summary = "constant integer value";
  let arguments = (ins IntAttr:$value, HIRAnyType:$typeOperand);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    custom<IntAttr>($value) `:` $typeOperand attr-dict
  }];
}

def ConstantBoolOp : HIROp<"constant_bool", [ConstantLike, Pure]> {
  let summary = "constant boolean value";
  let arguments = (ins SiBoolAttr:$value);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$value attr-dict";
  let hasFolder = 1;
}

def ConstantUnitOp : HIROp<"constant_unit", [ConstantLike, Pure]> {
  let summary = "constant unit value";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Type Constructors
//===----------------------------------------------------------------------===//

def UnitTypeOp : HIROp<"unit_type", [Pure]> {
  let summary = "construct the unit type";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def BoolTypeOp : HIROp<"bool_type", [Pure]> {
  let summary = "construct the bool type";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def IntTypeOp : HIROp<"int_type", [Pure]> {
  let summary = "construct the int type";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def TypeTypeOp : HIROp<"type_type", [Pure]> {
  let summary = "Construct the metatype 'Type'";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def UIntTypeOp : HIROp<"uint_type", [Pure]> {
  let summary = "construct a uint type with given width";
  let arguments = (ins HIRAnyType:$width);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$width attr-dict";
}

def AnyfuncTypeOp : HIROp<"anyfunc_type", [Pure]> {
  let summary = "construct the anyfunc type";
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def RefTypeOp : HIROp<"ref_type", [Pure]> {
  let summary = "construct a ref type";
  let arguments = (ins HIRAnyType:$innerType);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$innerType attr-dict";
}

def FuncTypeOp : HIROp<"func_type", [
  AttrSizedOperandSegments,
  Pure,
]> {
  let summary = "construct a function type";
  let arguments = (ins
    Variadic<HIRAnyType>:$typeOfArgs,
    Variadic<HIRAnyType>:$typeOfResults
  );
  let results = (outs HIRAnyType:$result);
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
  let description = [{
    Groups a sequence of ops into a single expression that yields results.
    The `phaseShift` attribute indicates a phase shift relative to the parent
    scope: negative values shift towards earlier (more constant) phases,
    positive values shift towards later (more dynamic) phases, and zero
    indicates no shift.
  }];
  let arguments = (ins SI32Attr:$phaseShift);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let assemblyFormat = [{
    $phaseShift (`:` type($results)^)? attr-dict-with-keyword $body
  }];
}

def YieldOp : HIROp<"yield", [
  HasParent<"ExprOp">,
  Pure,
  Terminator,
]> {
  let summary = "Yield result values from an expression";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Value Typing
//===----------------------------------------------------------------------===//

def TypeOfOp : HIROp<"type_of", [Pure]> {
  let summary = "Determine the type of a value";
  let arguments = (ins HIRAnyType:$input);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    $input attr-dict
  }];
  let hasFolder = 1;
  let hasCanonicalizeMethod = 1;
}

def CoerceToI1Op : HIROp<"coerce_to_i1", [Pure]> {
  let summary = "cast bool to i1 for branch conditions";
  let description = [{
    Converts an `!hir.any` value (expected to be a bool after type inference)
    to `i1` for use with `cf.cond_br`. Lowered to `mir.bool_to_i1` by
    HIR-to-MIR conversion.
  }];
  let arguments = (ins HIRAnyType:$input);
  let results = (outs I1:$result);
  let assemblyFormat = "$input attr-dict";
}

def CoerceTypeOp : HIROp<"coerce_type", [Pure]> {
  let summary = "Coerce a value to a given type";
  let arguments = (ins HIRAnyType:$input, HIRAnyType:$typeOperand);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    $input `,` $typeOperand attr-dict
  }];
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Operations
//===----------------------------------------------------------------------===//

/// Must NOT be `Pure`: two distinct `hir.inferrable` ops may infer to different
/// concrete types, but since they have no operands, no attributes, and the same
/// result type, CSE would merge them if they were marked `Pure`.
def InferrableOp : HIROp<"inferrable", []> {
  let summary = "a free type variable for inference";
  let description = [{
    Represents a type that has not yet been determined and will be resolved
    by the type inference pass. Must not be marked `Pure` because two
    distinct inferrables may resolve to different concrete types, and CSE
    would incorrectly merge them.
  }];
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

class HIRBinaryOp<string mnemonic>
    : HIROp<mnemonic, [Pure]> {
  let arguments = (ins HIRAnyType:$lhs, HIRAnyType:$rhs, HIRAnyType:$resultType);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` $resultType attr-dict";
}

// Arithmetic
def AddOp : HIRBinaryOp<"add"> { let summary = "add two values"; }
def SubOp : HIRBinaryOp<"sub"> { let summary = "subtract two values"; }
def MulOp : HIRBinaryOp<"mul"> { let summary = "multiply two values"; }
def DivOp : HIRBinaryOp<"div"> { let summary = "divide two values"; }
def ModOp : HIRBinaryOp<"mod"> { let summary = "modulo of two values"; }

// Bitwise
def AndOp : HIRBinaryOp<"and"> { let summary = "bitwise AND"; }
def OrOp  : HIRBinaryOp<"or">  { let summary = "bitwise OR"; }
def XorOp : HIRBinaryOp<"xor"> { let summary = "bitwise XOR"; }
def ShlOp : HIRBinaryOp<"shl"> { let summary = "shift left"; }
def ShrOp : HIRBinaryOp<"shr"> { let summary = "shift right"; }

// Comparison
def EqOp  : HIRBinaryOp<"eq">  { let summary = "compare for equality"; }
def NeqOp : HIRBinaryOp<"neq"> { let summary = "compare for inequality"; }
def LtOp  : HIRBinaryOp<"lt">  { let summary = "compare less than"; }
def GtOp  : HIRBinaryOp<"gt">  { let summary = "compare greater than"; }
def GeqOp : HIRBinaryOp<"geq"> { let summary = "compare greater or equal"; }
def LeqOp : HIRBinaryOp<"leq"> { let summary = "compare less or equal"; }

def LetOp : HIROp<"let", []> {
  let summary = "declare a named variable";
  let arguments = (ins StrAttr:$name, HIRAnyType:$type);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$name `:` $type attr-dict";
}

def UnifyOp : HIROp<"unify", [
  SameOperandsAndResultType,
  Commutative,
  Pure,
]> {
  let summary = "unify two types during inference";
  let description = [{
    Constrains two type values to be equal, used by the type inference
    pass. The result is the unified type. Folds away once both operands
    are identical.
  }];
  let arguments = (ins HIRAnyType:$lhs, HIRAnyType:$rhs);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasFolder = 1;
}

def StoreOp : HIROp<"store", []> {
  let summary = "store a value into a variable";
  let arguments = (ins HIRAnyType:$target, HIRAnyType:$value, HIRAnyType:$valueType);
  let assemblyFormat = "$target `,` $value `:` $valueType attr-dict";
}


def FuncOp : HIROp<"func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "a single-phase HIR function";
  let description = [{
    Represents a function that operates within a single phase. Produced
    by `SplitPhases` from a `hir.unified_func`, or directly by codegen
    for functions that don't span multiple phases. Lowered to `mir.func`
    by the HIR-to-MIR conversion.

    The `argNames` and `resultNames` attributes define exactly how many
    arguments and results the function has. The signature region's
    `hir.signature` op must have exactly that many `typeOfArgs` and
    `typeOfResults` entries. The body's `hir.return` terminator must
    have exactly `resultNames.size()` values.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$argNames,
    ArrayAttr:$resultNames,
    OptionalAttr<UnitAttr>:$isModule
  );
  let regions = (region MinSizedRegion<1>:$signature, MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def ReturnOp : HIROp<"return", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must be empty or match values size",
    CPred<"$typeOfValues.size() == 0 || $values.size() == $typeOfValues.size()">>,
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"FuncOp, UnifiedFuncOp">,
]> {
  let summary = "return from an HIR function";
  let description = [{
    Terminates an `hir.func` or `hir.unified_func` body.  Carries the
    return values alongside optional type operands (`typeOfValues`).

    After `SplitPhases`, per-phase functions use the signature region as
    the source of truth for types; `typeOfValues` is empty.
    In `unified_func` bodies, it is populated by codegen.
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
  let hasVerifier = 1;
}

def NextPhaseOp : HIROp<"next_phase", [
  HasParent<"FuncOp">,
  Pure,
  Terminator,
]> {
  let summary = "transition to the next phase block";
  let description = [{
    Terminates a block within an `hir.func` and branches to the next
    phase block. Used by `SplitPhases` to separate code that runs in
    consecutive phases within the same function.
  }];
  let successors = (successor AnySuccessor:$dest);
  let assemblyFormat = "$dest attr-dict";
}

def CallOp : HIROp<"call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<InferTypeOpInterface>,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
  RangedTypesMatchWith<"arguments match argument types", "arguments", "typeOfArgs", "$_self">,
  RangedTypesMatchWith<"results match result types", "results", "typeOfResults", "$_self">,
]> {
  let summary = "A direct call to a named function";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<HIRAnyType>:$arguments,
    Variadic<HIRAnyType>:$typeOfArgs,
    Variadic<HIRAnyType>:$typeOfResults
  );
  let results = (outs Variadic<HIRAnyType>:$results);
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
    ArrayAttr:$resultNames,
    OptionalAttr<UnitAttr>:$isModule
  );
  let regions = (region
    MinSizedRegion<1>:$signature,
    MinSizedRegion<1>:$body
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

def UnifiedCallOp : HIROp<"unified_call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "A call not yet split into phases";
  let arguments = (ins
    FlatSymbolRefAttr:$callee,
    Variadic<AnyType>:$arguments,
    Variadic<HIRAnyType>:$typeOfArgs,
    Variadic<HIRAnyType>:$typeOfResults,
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
  ParentOneOf<["FuncOp", "UnifiedFuncOp", "SplitFuncOp"]>,
  Pure,
  Terminator,
]> {
  let summary = "Terminator for the signature region of a function";
  let arguments = (ins
    Variadic<HIRAnyType>:$typeOfArgs,
    Variadic<HIRAnyType>:$typeOfResults
  );
  let assemblyFormat = [{
    ` ` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)` attr-dict
  }];
  let hasVerifier = 1;
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

    Once phased evaluation is complete, unused private split funcs can be
    removed by SymbolDCE.
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
}

def MultiphaseFuncOp : HIROp<"multiphase_func", [
  Symbol,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "A chain of sub-functions evaluated iteratively, one phase at a time";
  let description = [{
    A `hir.multiphase_func` represents a single phase of a split function that
    itself spans multiple internal sub-phases. The sub-functions are evaluated
    in order. Arguments are marked `first` or `last` to indicate whether they
    are available from the first or last sub-phase.

    Multiphase funcs are always private. They are kept alive by SymbolDCE
    through the `FlatSymbolRefAttr` references in their parent split func.
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
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def OpaquePackOp : HIROp<"opaque_pack", [Pure]> {
  let summary = "Pack multiple values into a single opaque bundle";
  let arguments = (ins Variadic<HIRAnyType>:$operands);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{ `(` $operands `)` attr-dict }];
  let hasCanonicalizeMethod = 1;
}

def OpaqueUnpackOp : HIROp<"opaque_unpack", [Pure]> {
  let summary = "Unpack an opaque bundle into individual values";
  let arguments = (ins HIRAnyType:$input);
  let results = (outs Variadic<HIRAnyType>:$results);
  let assemblyFormat = [{ $input attr-dict (`:` type($results)^)? }];
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
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    $value attr-dict
  }];
  let hasFolder = 1;
}

#endif // SILICON_HIR_OPS_TD
