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
include "silicon/HIR/OpInterfaces.td"
include "silicon/HIR/Types.td"

// Base class for the operations in this dialect.
class HIROp<string mnemonic, list<Trait> traits = []> :
  Op<HIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Constants
//===----------------------------------------------------------------------===//

def ConstantIntOp : HIROp<"constant_int", [
  Pure,
  DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
]> {
  let summary = "constant integer value";
  let arguments = (ins IntAttr:$value, HIRAnyType:$typeOperand);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    custom<IntAttr>($value) `:` $typeOperand attr-dict
  }];
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getTypeOperandMutable();
    }
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
  HasParent<"ExprOp, ReplicateOp, PickOp">,
  Pure,
  Terminator,
]> {
  let summary = "Yield result values from an expression, replicate, or pick body";
  let arguments = (ins Variadic<AnyType>:$operands);
  let assemblyFormat = [{
    ($operands^ `:` type($operands))? attr-dict
  }];
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Replicate
//===----------------------------------------------------------------------===//

def ReplicateOp : HIROp<"replicate", [
  RecursiveMemoryEffects,
  SingleBlock,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "stamp out body once per record in a hit list";
  let description = [{
    Consumes a hit list from an earlier phase and stamps out its body
    region once per record in the list, substituting the record's
    snapshot values for the body's block arguments.

    Each record in the hit list is an opaque-packed bundle whose fields
    correspond one-to-one to the body's block arguments. Bodies have no
    results and no threaded values; state flowing from one instantiation
    to the next goes through variables of the receiving phase. The body
    is terminated by `hir.yield` with no operands, or by an exit
    terminator such as `uir.return` when an early exit of the producing
    phase rides the record: if the exit fires during expansion, it
    terminates the receiving phase's function.

    All values have the implicit `!hir.any` type. The replicate is always
    expanded by `SpecializeFuncs` and never lowered to MIR.
  }];
  let arguments = (ins HIRAnyType:$hits);
  let regions = (region MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// Hits Vector
//===----------------------------------------------------------------------===//

def OpaqueListCreateOp : HIROp<"opaque_list_create", [
  MemoryEffects<[MemAlloc<DefaultResource>]>,
]> {
  let summary = "create an empty opaque list";
  let description = [{
    Allocates an empty opaque list for collecting hits during
    earlier-phase execution. Entries are appended with
    `hir.opaque_list_push` and the list is consumed by
    `hir.replicate`.
  }];
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def OpaqueListPushOp : HIROp<"opaque_list_push", [
  MemoryEffects<[MemWrite<DefaultResource>]>,
]> {
  let summary = "append an entry to an opaque list";
  let description = [{
    Appends an opaque-packed entry to the given opaque list. The list
    is mutated in place; there is no result. Used to build hits vectors
    during earlier-phase execution.
  }];
  let arguments = (ins HIRAnyType:$list, HIRAnyType:$entry);
  let assemblyFormat = "$list `,` $entry attr-dict";
}

//===----------------------------------------------------------------------===//
// Choices
//===----------------------------------------------------------------------===//

def ChoiceTypeOp : HIROp<"choice_type", [Pure]> {
  let summary = "type constructor for choice values";
  let description = [{
    Constructs the type of choice values as an SSA value, for use as the
    type operand of control flow results and yields that carry a choice
    across a phase boundary. Mirrors `hir.opaque_type`: choice values are
    `!hir.any` like all HIR values, and the concrete `!si.choice` type
    only materializes during HIR-to-MIR conversion.
  }];
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "attr-dict";
}

def ChoiceOp : HIROp<"choice", [Pure]> {
  let summary = "construct a choice value";
  let description = [{
    Constructs a choice value: the tag of the taken alternative of a
    branch-shaped control flow region, plus the payload of snapshot
    values crossing the phase boundary out of that alternative. Choices
    are constructed per arm in the producing phase and consumed by
    `hir.pick` in the receiving phase.

    Tags are dense per control flow op: the i-th alternative of the
    consuming `hir.pick` handles tag i.
  }];
  let arguments = (ins I32Attr:$tag, Variadic<HIRAnyType>:$payload);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    $tag `(` $payload `)` attr-dict
  }];
}

def PickOp : HIROp<"pick", [
  RecursiveMemoryEffects,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "instantiate the taken alternative of a choice";
  let description = [{
    Consumes a choice value from an earlier phase and instantiates
    exactly the alternative named by the choice's tag, substituting the
    choice's payload values for the alternative's block arguments. The
    i-th alternative region handles tag i.

    Picks are valued: an alternative on a rejoining path is terminated
    by `hir.yield` whose operands become the pick's results. An
    alternative may instead be terminated by a loop exit (`uir.break`,
    `uir.continue`) targeting an enclosing loop, or by `uir.unreachable`
    on a dead path; the pick's results never materialize on such a path
    and their consumers are unreachable. Function exits never appear in
    alternatives: exit normalization rewrites them into choice/pick
    value form before any pick is built. All values have the implicit
    `!hir.any` type. The pick is always expanded by `SpecializeFuncs`
    and never lowered to MIR; a pick whose choice value was never
    produced during phase evaluation is an internal compiler error at
    expansion time.
  }];
  let arguments = (ins HIRAnyType:$choice);
  let results = (outs Variadic<HIRAnyType>:$results);
  let regions = (region VariadicRegion<SizedRegion<1>>:$alternatives);
  let hasCustomAssemblyFormat = 1;
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

def CoerceTypeOp : HIROp<"coerce_type", [
  Pure,
  DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
]> {
  let summary = "Coerce a value to a given type";
  let arguments = (ins HIRAnyType:$input, HIRAnyType:$typeOperand);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = [{
    $input `,` $typeOperand attr-dict
  }];
  let hasCanonicalizeMethod = 1;
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getTypeOperandMutable();
    }
  }];
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
    : HIROp<mnemonic, [
        Pure,
        DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
      ]> {
  let arguments = (ins HIRAnyType:$lhs, HIRAnyType:$rhs, HIRAnyType:$resultType);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs `:` $resultType attr-dict";
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getResultTypeMutable();
    }
  }];
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

def UnifyOp : HIROp<"unify", [
  SameOperandsAndResultType,
  Commutative,
  MemoryEffects<[MemWrite]>,
]> {
  let summary = "unify two types during inference";
  let description = [{
    Constrains two type values to be equal, used by the type inference
    pass. The result is the unified type. Canonicalizes away once both
    operands are identical.

    Uses a `MemWrite` effect instead of `Pure` so that dead-code
    elimination does not discard unused unify ops before InferTypes has
    a chance to process them. For example, CheckCalls creates
    `hir.unify %inferrable, %callee_type` to record that a literal's
    type should match the callee's parameter type; with `Pure` (or
    `MemRead`, which MLIR also considers trivially dead), DCE would
    discard this constraint.
  }];
  let arguments = (ins HIRAnyType:$lhs, HIRAnyType:$rhs);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$lhs `,` $rhs attr-dict";
  let hasCanonicalizeMethod = 1;
}

//===----------------------------------------------------------------------===//
// Variables
//===----------------------------------------------------------------------===//

def VarOp : HIROp<"var", [
  MemoryEffects<[MemAlloc<DefaultResource>]>,
  DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
]> {
  let summary = "declare a mutable variable";
  let description = [{
    Declares a mutable variable holding values of the given type and
    produces a reference to it. Storage is per call frame. The variable's
    home phase is the phase of this op; `hir.write_var` is legal only at
    the home phase, and `hir.read_var` at the home phase or later. The
    reference is an opaque SSA value and the home phase is evaluation-time
    data carried alongside the storage cell: home-phase legality is checked
    when reads and writes execute, not by PhaseAnalysis.

    Must not be `Pure`: two distinct variables are distinct mutable
    cells, and the `MemAlloc` effect keeps CSE from merging them.
  }];
  let arguments = (ins HIRAnyType:$varType);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "`:` $varType attr-dict";
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getVarTypeMutable();
    }
  }];
}

def ReadVarOp : HIROp<"read_var", [
  MemoryEffects<[MemRead<DefaultResource>]>,
  DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
]> {
  let summary = "read the current value of a variable";
  let description = [{
    Reads the current value of a variable. Like `hir.write_var`, the read
    inherits the phase of the surrounding control flow region; it does not
    float to consumer demand. A read at the variable's home phase observes
    the value at this op's program-order position; a read at a later phase
    observes the frozen final value. To sample a variable at a specific
    program-order position from a later phase, the read is explicitly
    wrapped in a const region that places it at the home phase, and only
    the snapshot value crosses the phase boundary. Reading at a phase
    earlier than the home phase is illegal; this is checked when the read
    executes, since the reference is opaque and the home phase is
    evaluation-time data.

    The `varType` operand carries the variable's value type, matching the
    type operand of the defining `hir.var`.
  }];
  let arguments = (ins HIRAnyType:$var, HIRAnyType:$varType);
  let results = (outs HIRAnyType:$result);
  let assemblyFormat = "$var `:` $varType attr-dict";
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getVarTypeMutable();
    }
  }];
}

def WriteVarOp : HIROp<"write_var", [
  MemoryEffects<[MemWrite<DefaultResource>]>,
  DeclareOpInterfaceMethods<TypeOperandsOpInterface>,
]> {
  let summary = "write a new value to a variable";
  let description = [{
    Writes a new value to a variable. Writes are legal only at the
    variable's home phase. Since the reference is an opaque SSA value and
    the home phase is evaluation-time data, this is checked when the write
    executes; PhaseAnalysis treats the write as an ordinary side-effecting
    op anchored at its block phase.

    The `varType` operand carries the variable's value type, matching the
    type operand of the defining `hir.var`.
  }];
  let arguments = (ins HIRAnyType:$var, HIRAnyType:$value, HIRAnyType:$varType);
  let assemblyFormat = "$var `,` $value `:` $varType attr-dict";
  let extraClassDefinition = [{
    bool $cppClass::isTypeOperand(::mlir::OpOperand &operand) {
      return &operand == &getVarTypeMutable();
    }
  }];
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
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  ReturnLike,
  Terminator,
  HasParent<"FuncOp, UnifiedFuncOp">,
]> {
  let summary = "return from an HIR function";
  let description = [{
    Terminates an `hir.func` or `hir.unified_func` body.  Carries the
    return values alongside their type operands (`typeOfValues`).

    The `typeOfValues` list must match the `values` list in size: each
    return value has a corresponding type-level value describing its type.
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

def EagerFuncOp : HIROp<"eager_func", [
  Symbol,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "A chain of eagerly-evaluated compile-time links ending in a runtime tail";
  let description = [{
    An `hir.eager_func` names a chain of functions whose leading links are
    compile-time phases the evaluation driver evaluates eagerly, ending in a
    tail link that carries the function's declared runtime arguments. The shape
    is:

    ```mlir
    hir.eager_func @f(a, b) -> (r, ctx) [@f.pre0, @f.pre1, @f.tail]
    ```

    The links chain an opaque context value from one to the next:

    - The head link takes no arguments and produces exactly one result, the
      outgoing context.
    - Each middle link takes exactly one argument, the incoming context, and
      produces exactly one result, the outgoing context.
    - The tail link takes the op's declared arguments plus one trailing context
      argument, and produces exactly the op's declared results.

    There are always at least two links, and every link resolves to an
    `hir.func`. The op's external signature is exactly its declared arguments
    and results, so `hir.call` targets an `hir.eager_func` like it targets an
    `hir.func`.

    Eagerness is policy carried by this op's existence, not inferred from a
    function being zero-argument: a plain zero-argument function is runtime code
    and is never eagerly interpreted, whereas the links of an `hir.eager_func`
    are. The evaluation driver evaluates the head link once it becomes evaluable
    and specializes the next link forward. When reduction leaves a single link,
    the whole op is replaced by a same-name `hir.func` shim that tail-calls that
    link, so no symbol use is ever rewritten.
  }];
  let arguments = (ins
    SymbolNameAttr:$sym_name,
    OptionalAttr<StrAttr>:$sym_visibility,
    ArrayAttr:$argNames,
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
