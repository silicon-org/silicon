//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_OPS_TD
#define SILICON_UIR_OPS_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "silicon/HIR/Types.td"
include "silicon/UIR/Dialect.td"

// Base class for the operations in this dialect.
class UIROp<string mnemonic, list<Trait> traits = []> :
  Op<UIRDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// Op Doc Groups
//===----------------------------------------------------------------------===//

def RegionTerminators : OpDocGroup {
  let summary = "Region terminators";
  let description = [{
    Terminators for regions inside structured CF and expression ops.
    Each structured region must end with exactly one of these.
  }];
}

def StructuredControlFlow : OpDocGroup {
  let summary = "Structured control flow";
  let description = [{
    Ops representing control flow as nested regions, preserving the
    expression tree structure for phase inference. Consumed by FlattenCF
    after phase splitting.
  }];
}

def PhaseGrouping : OpDocGroup {
  let summary = "Phase grouping";
  let description = [{
    Ops for grouping expressions and pinning phases during phase
    inference. Consumed by SplitPhases.
  }];
}

def UnifiedFunctions : OpDocGroup {
  let summary = "Unified functions";
  let description = [{
    Ops for representing functions that span multiple phases and their
    call sites. Consumed by SplitPhases, which decomposes them into
    per-phase `hir.func` and `hir.call` ops.
  }];
}

//===----------------------------------------------------------------------===//
// Region Terminators
//===----------------------------------------------------------------------===//

let opDocGroup = RegionTerminators in {

def YieldOp : UIROp<"yield", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "yield values from a region";
  let description = [{
    Terminates and yields values from a `uir.expr` or `uir.if` region.
    Each value has a corresponding type operand. For void regions the op
    carries no operands.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$values,
    Variadic<HIRAnyType>:$typeOfValues
  );
  let assemblyFormat = [{
    ($values^ `:` $typeOfValues)?
    attr-dict
  }];
}

def BreakOp : UIROp<"break", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "exit enclosing loop with values";
  let description = [{
    Exits the nearest enclosing `uir.loop`, providing the loop's result
    values. Must be transitively nested inside a `uir.loop` region
    (possibly inside `uir.if` etc.). Each value has a corresponding type
    operand, unified with the enclosing loop's result type operands.
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

def ContinueOp : UIROp<"continue", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "advance enclosing loop to next iteration";
  let description = [{
    Advances the nearest enclosing `uir.loop` to its next iteration. Must
    be transitively nested inside a `uir.loop` region. Carries the
    loop-carried values for the next iteration, one per loop iteration
    argument, with a corresponding type operand for each value.
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

def ReturnOp : UIROp<"return", [
  AttrSizedOperandSegments,
  PredOpTrait<"typeOfValues must match values size",
    CPred<"$values.size() == $typeOfValues.size()">>,
  Pure,
  Terminator,
]> {
  let summary = "return from enclosing function";
  let description = [{
    Returns from the enclosing function. This is the structured early-exit
    counterpart to `hir.return`. Unlike `hir.return` (which terminates the
    function body's block in flat CFG), `uir.return` can appear inside
    structured CF regions at any nesting depth.

    Carries the same operand structure as `hir.return`: return values and
    their type operands.
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
}

def UnreachableOp : UIROp<"unreachable", [
  Pure,
  Terminator,
]> {
  let summary = "marks unreachable control flow";
  let description = [{
    Marks a point in the block that control flow never reaches. Emitted
    by codegen after structured CF ops where all regions exit early (e.g.,
    a `uir.if` where both branches use `uir.return`).

    This avoids the need for dummy return values or placeholder ops after
    exhaustive early exits.
  }];
  let assemblyFormat = [{ attr-dict }];
}

} // let opDocGroup = RegionTerminators

//===----------------------------------------------------------------------===//
// Structured Control Flow
//===----------------------------------------------------------------------===//

let opDocGroup = StructuredControlFlow in {

def IfOp : UIROp<"if", [
  RecursiveMemoryEffects,
]> {
  let summary = "structured if/else expression";
  let description = [{
    A structured if/else expression with two regions: `then` and `else`.
    Each region has a single block terminated by `uir.yield` or an
    early-exit op (`uir.return`, `uir.break`, `uir.continue`,
    `uir.unreachable`).

    The condition is a `!si.bool` operand. The op produces variadic
    results and carries result type operands. Inside each region,
    `hir.unify` ops connect the yielded value's type to the
    corresponding result type operand.

    If some regions exit early, only the yielding regions contribute
    results. If all regions exit early, the op produces no results and
    the subsequent op must be `uir.unreachable`.

    The `else` region may be omitted for if-without-else, in which case
    the op produces no results.

    For phase inference rules, see `docs/design/phase-inference.md`.
    For FlattenCF lowering, see `docs/design/control-flow.md`.
  }];
  let arguments = (ins
    HIRAnyType:$condition,
    Variadic<HIRAnyType>:$resultTypes
  );
  let results = (outs Variadic<HIRAnyType>:$results);
  let regions = (region
    MinSizedRegion<1>:$thenRegion,
    AnyRegion:$elseRegion
  );
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def LoopOp : UIROp<"loop", [
  AttrSizedOperandSegments,
  RecursiveMemoryEffects,
  SingleBlock,
]> {
  let summary = "structured loop expression";
  let description = [{
    A structured loop expression with one region: the loop body.

    The op carries a list of initial values for its loop-carried
    iteration arguments, with a corresponding type operand for each
    initial value. The body's single block has one block argument per
    iteration argument, receiving the current iteration's value. The body
    is terminated by `uir.continue`, which carries the loop-carried values
    for the next iteration, or by `uir.break`, which exits the loop with
    its result values. A `uir.break` nested in a `uir.if` exits the loop
    as well.

    The op produces variadic results and carries result type operands.
    The result count is independent of the iteration argument count.
    `uir.break` carries type operands unified with the loop's result type
    operands.

    `while` and `for` loops desugar to `uir.loop`.

    For phase inference rules, see `docs/design/phase-inference.md`.
    For FlattenCF lowering, see `docs/design/control-flow.md`.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$inits,
    Variadic<HIRAnyType>:$initTypes,
    Variadic<HIRAnyType>:$resultTypes
  );
  let results = (outs Variadic<HIRAnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
  let hasRegionVerifier = 1;
}

} // let opDocGroup = StructuredControlFlow

//===----------------------------------------------------------------------===//
// Phase Grouping
//===----------------------------------------------------------------------===//

let opDocGroup = PhaseGrouping in {

def ExprOp : UIROp<"expr", [
  RecursiveMemoryEffects,
  SingleBlock,
]> {
  let summary = "block expression with optional phase pin";
  let description = [{
    Groups a sequence of ops into a single expression. The body region
    must be terminated by `uir.yield`.

    A `uir.expr` can be **floating** or **pinned**:
    - **Floating** (`uir.expr { ... }`): the expression's phase comes
      from its consumer (demand-driven).
    - **Pinned** (`uir.expr pin { ... }` or `uir.expr pin -1 { ... }`):
      the expression is fixed at `p(enclosing_block) + phaseShift`.
      Used for `const { ... }` / `dyn { ... }` blocks.

    Like `uir.if`, carries result type operands and produces type
    results alongside value results.

    For codegen and phase inference details, see
    `docs/design/phase-inference.md`.
  }];
  let arguments = (ins
    OptionalAttr<UnitAttr>:$pin,
    DefaultValuedAttr<SI32Attr, "0">:$phaseShift,
    Variadic<HIRAnyType>:$resultTypes
  );
  let results = (outs Variadic<HIRAnyType>:$results);
  let regions = (region MinSizedRegion<1>:$body);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

def PinOp : UIROp<"pin", []> {
  let summary = "pin values at a phase offset";
  let description = [{
    Pins one or more values at a phase offset relative to the enclosing
    block. Lightweight alternative to `uir.expr` when no region is
    needed (clean use-def chain, no disconnected side effects).

    Used for:
    - `let` bindings: `uir.pin %val, 0 : !hir.any`
    - `const { expr }` after inlining: `uir.pin %val, -1 : !hir.any`
    - `dyn { expr }` after inlining: `uir.pin %val, 1 : !hir.any`

    Not marked `Pure` to prevent CSE from merging distinct pins (same
    rationale as `hir.inferrable`).
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$inputs,
    SI32Attr:$phaseOffset
  );
  let results = (outs Variadic<HIRAnyType>:$outputs);
  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;
}

} // let opDocGroup = PhaseGrouping

//===----------------------------------------------------------------------===//
// Unified Functions
//===----------------------------------------------------------------------===//

include "mlir/IR/OpAsmInterface.td"

let opDocGroup = UnifiedFunctions in {

def SignatureOp : UIROp<"signature", [
  AttrSizedOperandSegments,
  ParentOneOf<["FuncOp", "SplitFuncOp"]>,
  Pure,
  Terminator,
]> {
  let summary = "terminator for a UIR function's signature region";
  let description = [{
    Terminates the signature region of a `uir.func` or `uir.split_func`.
    Carries the type-level values for each argument and result.
    Mirrors `hir.signature` but lives in the UIR dialect to avoid
    cross-dialect parent dependencies.
  }];
  let arguments = (ins
    Variadic<HIRAnyType>:$typeOfArgs,
    Variadic<HIRAnyType>:$typeOfResults
  );
  let assemblyFormat = [{
    ` ` `(` $typeOfArgs `)` `->` `(` $typeOfResults `)` attr-dict
  }];
  let hasVerifier = 1;
}

def FuncOp : UIROp<"func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "unified function spanning multiple phases";
  let description = [{
    A function whose body spans multiple phases, before phase splitting.
    Mirrors `hir.unified_func` in the UIR dialect.

    The function has a signature region (terminated by `uir.signature`)
    and a body region. Both share the same entry block arguments,
    corresponding to the function's declared arguments.

    Argument and result phase offsets are determined by `const`/`dyn`
    modifiers on the function's parameters and return type.
  }];
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

def CallOp : UIROp<"call", [
  AttrSizedOperandSegments,
  DeclareOpInterfaceMethods<SymbolUserOpInterface>,
]> {
  let summary = "call to a unified function";
  let description = [{
    A call not yet split into phases. Mirrors `hir.unified_call` in the
    UIR dialect. Carries the full set of arguments for all phases.

    SplitPhases decomposes this into per-phase `hir.call` ops using the
    callee's `uir.split_func` witness.
  }];
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

def SplitFuncOp : UIROp<"split_func", [
  IsolatedFromAbove,
  Symbol,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmBlockArgumentNames"]>,
]> {
  let summary = "witness recording how a unified function was split";
  let description = [{
    Records the result of splitting a `uir.func` into separate per-phase
    functions. Preserves the original function's signature (for library
    callers) and maps phase numbers to the generated per-phase `hir.func`
    ops.

    Its sole purpose is to allow `uir.call` ops to be decomposed into
    per-phase `hir.call` ops without re-running phase inference on the
    callee.
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

} // let opDocGroup = UnifiedFunctions

#endif // SILICON_UIR_OPS_TD
