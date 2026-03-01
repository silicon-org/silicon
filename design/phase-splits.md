# Phase Splits

This document describes how the Silicon language intends to implement metaprogramming, function specialization, and phased compile-time execution.
It is aspirational and does not reflect the current state of the code base.

## Unified IR

After parsing and an initial set of optimizations, value inference, and call checking, the IR is in its unified form.
It uses `hir.unified_func` and `hir.unified_call` ops to represent functions and calls, where arguments and regions of all execution phases are merged in a single structure, closely reflecting the AST after parsing.
The call checking pass has copied the function signature into call sites and the function body as a constraint.
Note that all HIR values have MLIR type `!hir.any`, but it is omitted from all assembly formats.

For example, consider the following unified function:

```mlir
hir.unified_func @foo(%a: -3, %b: -1) -> (c: 0) {
  // signature omitted for brevity
  // (CheckCalls pass has copied most of the type computation into the body)
} {
  %int_type = hir.int_type                                                // any phase
  %type_type = hir.type_type                                              // any phase
  %a.type = hir.unified_call @doU() : () -> (%type_type: 0)               // phase -4
  %a0 = hir.coerce_type %a : %a.type                                      // phase -3
  %a1 = hir.unified_call @doV(%a0) : (%a.type: 0) -> (%a.type: 0)         // phase -3
  %a2 = hir.const { hir.yield %a1 }                                       // phase -2
  %b.type = hir.unified_call @doW(%a2) : (%a.type: 0) -> (%type_type: 0)  // phase -2
  %b0 = hir.coerce_type %b : %b.type                                      // phase -1
  %b1 = hir.unified_call @doX(%b0) : (%b.type: 0) -> (%b.type: 0)         // phase -1
  %b2 = hir.const { hir.yield %b1 }                                       // phase 0
  %c = hir.unified_call @doY(%b2) : (%b.type: 0) -> (%int_type: 0)        // phase 0
  hir.return %c : %int_type
}
```

Arguments indicate in which phase they are available.
They are unknown in earlier phases and cannot be used for control flow or typing.
Results indicate in which phase they are computed.
The return op assigns these phases to its operands, and back-propagates phase requirements across the function body.
Uses of a value as a type require that the type is available in the phase before the op that uses it as a type.
This enforces that we evaluate all types to constant values in the phase before, such that they can then be absorbed into the MIR lowering for the next phase.
Ops like `hir.const` and `hir.dyn` introduce an additional phase shift.
Constant-like ops don't have any phase requirements, since they are trivially copied to whichever phase they are needed in.

The above example is roughly equivalent to the following input:

```silicon
fn foo(const const const a: doU(), const b: doW(const { doV(a) })) -> (c: int) {
  doY(const { doX(b) })
}
```

### `hir.unified_func`

A unified function consists of:

- a symbol visibility
- a symbol name, such as `@hello`
- a list of arguments and their phases, such as `(%a: 0, %b: -3, %c: 2)`
- a list of result names and their phases, such as `(x: 4, y: -2, z: 0)`
- a signature region computing the types of arguments and results, terminated with `hir.signature`
- a body region describing the actual computation of the function, terminated with `hir.return`

The block arguments of the first block in the signature and body region correspond to the function arguments and are not printed.
This corresponds to how `func.func` prints the names of its body region block arguments inline in the list of function arguments.
The syntax of a unified function looks roughly like this:

```mlir
hir.unified_func @hello(%a: 0, %b: -3, %c: 2) -> (x: 4, y: -2, z: 0) {
  // signature region
  hir.signature (%a.type, %b.type, %c.type) -> (%x.type, %y.type, %z.type)
} {
  // body region
  hir.return %x, %y, %z : %x.type, %y.type, %z.type
}
```

### `hir.unified_call`

A unified function call consists of:

- a symbol name of the function to be called, such as `@hello`
- a list of operands to pass as arguments, such as `(%i, %j, %k)`
- a list of argument types and phases, such as `(%a.type: 0, %b.type: -3, %c.type: 2)`
- a list of result types and phases, such as `(%x.type: 4, %y.type: -2, %z.type: 0)`

The phases indicate in which phase a call operand must be available, and in which phase a call result becomes available.
These phases are relative to the call op; the entire call op may be phase-shifted relative to other ops in the caller, e.g., if the result of a call is required in a specific phase, the call op's phase is shifted to accommodate all phase requirements.
The type operands must be available one phase before the corresponding argument or result.
The number, names, and phases of arguments and results of the call must match the corresponding function definition.

## Split IR

Consider the previous example split into separate phases:

```mlir
hir.split_func @foo(%a: -3, %b: -1) -> (c: 0) {
  // signature same as unified func
} [
  -3: @foo.0,  // implied (a) -> (ctx01)
  -1: @foo.1,  // implied (b, ctx01) -> (ctx12)
  0: @foo.2    // implied (ctx12) -> (c)
]
hir.multiphase_func @foo.0(last a) -> (ctx01) [
  @foo.0a,  // implied () -> (ctx0ab)
  @foo.0b   // implied (a, ctx0ab) -> (ctx01)
]
hir.func @foo.0a() -> (ctx0ab) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  %a.type = hir.call @doU() : () -> (%type_type)  // phase -4
  %ctx0ab = hir.opaque_pack (%a.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @foo.0b(%a, %ctx0ab) -> (ctx01) {
  %opaque_type = hir.opaque_type
  %a.type = hir.opaque_unpack %ctx0ab
  %a0 = hir.coerce_type %a : %a.type                 // phase -3
  %a1 = hir.call @doV(%a0) : (%a.type) -> (%a.type)  // phase -3
  %ctx01 = hir.opaque_pack (%a1, %a.type)
  hir.return %ctx01 : %opaque_type
}
hir.multiphase_func @foo.1(last b, first ctx01) -> (ctx12) [
  @foo.1a,  // implied (ctx01) -> (ctx1ab)
  @foo.1b   // implied (b, ctx1ab) -> (ctx12)
]
hir.func @foo.1a(%ctx01) -> (ctx1ab) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  %a2, %a.type = hir.opaque_unpack %ctx01
  %b.type = hir.call @doW(%a2) : (%a.type) -> (%type_type)  // phase -2
  %ctx1ab = hir.opaque_pack (%b.type)
  hir.return %ctx1ab : %opaque_type
}
hir.func @foo.1b(%b, %ctx1ab) -> (ctx12) {
  %opaque_type = hir.opaque_type
  %b.type = hir.opaque_unpack %ctx1ab
  %b0 = hir.coerce_type %b : %b.type                 // phase -1
  %b1 = hir.call @doX(%b0) : (%b.type) -> (%b.type)  // phase -1
  %ctx12 = hir.opaque_pack (%b1, %b.type)
  hir.return %ctx12 : %opaque_type
}
hir.func @foo.2(%ctx12) -> (c) {
  %int_type = hir.int_type
  %b2, %b.type = hir.opaque_unpack %ctx12
  %c = hir.call @doY(%b2) : (%b.type) -> (%int_type)  // phase 0
  hir.return %c : %int_type
}
```

The original `@foo` unified function had 5 internal phases in total.
However, only 3 of these are visible to the outside when looking at the function arguments and results.
The split function only lists separate functions for the externally visible phases.
All internal phases are absorbed into multiphase functions.

The HIR operations `hir.opaque_type`, `hir.opaque_pack`, and `hir.opaque_unpack` allow SSA values to be packed and unpacked from a single SSA value.
This allows functions to bundle op internal state that needs to flow between execution phases.

In this example, `@foo.0` is a multiphase function without any arguments for the first phase.
This means that `@foo.0a` can be fully evaluated at compile time and `@foo.0b` can be specialized with the result, even if the input is compiled to a library.

### `hir.split_func`

This operation indicates how a unified function has been split up into separate phases.
Unified functions are lowered to split functions of the same name.
A split function consists of:

- a symbol visibility
- a symbol name, such as `@hello`
- a list of arguments and their phases, such as `(%a: 0, %b: -3, %c: 2)`
- a list of result names and their phases, such as `(x: 4, y: -2, z: 0)`
- a signature region computing the types of arguments and results, terminated with `hir.signature`
- a list of phases and the name of the corresponding function

The signature region is identical to the original unified function.
This allows a piece of input IR to be compiled into a library by splitting it into distinct phases, but then still allow unified user code to link against the library and have the CheckCalls impose the function signature as constraints onto call sites in the user code.

Private unified functions may not need a corresponding split function, since no later user code can do a unified call to the function.
Dead symbol elimination should automatically collect private split functions.

Each entry in the list of phases consists of:

- a phase number, such as `42:`
- a name of the function that contains the code for this phase

Each function in the list must accept the arguments of the split function with the corresponding phase, and must produce the results with the corresponding phase.
All except for the last function are expected to return one additional, opaque result containing internal values that flow to the next split function.
All except for the first function are expected to accept one additional, opaque argument containing internal values that flow from the previous split function.
The symbol user verifier of the op verifies that each function in the list of phases accepts the expected number of arguments and returns the expected number of results.

The syntax of a split function looks roughly like this:

```mlir
hir.split_func @hello(%a: 0, %b: -3, %c: 2) -> (x: 4, y: -2, z: 0) {
  // signature region
  hir.signature (%a.type, %b.type, %c.type) -> (%x.type, %y.type, %z.type)
} [
  -3: @hello.0,  // implied (b) -> (ctx01)
  -2: @hello.1,  // implied (ctx01) -> (y, ctx12)
  0: @hello.2,   // implied (a, ctx12) -> (z, ctx23)
  2: @hello.3,   // implied (c, ctx23) -> (ctx34)
  4: @hello.4    // implied (ctx34) -> (x)
]
```

### `hir.multiphase_func`

A multiphase function explicitly encodes that a series of functions should be evaluated iteratively in separate phases.
It consists of:

- a symbol visibility
- a symbol name, such as `@world`
- a list of argument names and a keyword indicating whether each argument goes into the first or last phase, such as `(last a, first b, last c)`
- a list of result names, such as `(d, e, f)`
- a list of names of functions to call in subsequent phases

All except for the last function are expected to return one additional, opaque result containing internal values that flow to the next function.
All except for the first function are expected to accept one additional, opaque argument containing internal values that flow from the previous function.
Once a phase has been evaluated, the function's opaque result is passed to the subsequent phase function as a constant, specializing it.
The first function accepts the multiphase function arguments specified as "first".
The last function accepts the multiphase function arguments specified as "last", and the opaque result from the previous phase.
The last function returns exactly the multiphase function results.

This op can be iteratively reduced to a regular function by removing and evaluating the first function in the list, and using the returned values to specialize the subsequent function.
As functions are evaluated and specialized, the names in the list are likely to change since specializing a function with multiple users means creating a specialized copy of it instead of modifying the original declaration.
When only a single function remains in the list, the op can be replaced with a direct call to that function.

The op intentionally only supports arguments for the first and last phases.
If it has arguments for the first phase, specializing the function for those arguments yields a multiphase function that has no more arguments for the first phase.
A multiphase function that only has arguments for the last phase can be fully evaluated and reduced to a regular function at compile time.

The syntax of a multiphase function looks roughly like this:

```mlir
hir.multiphase_func @world(last a, first b, last c) -> (d, e, f) [
  @world.0,  // implied (b) -> (ctx01)
  @world.1,  // implied (ctx01) -> (ctx12)
  @world.2   // implied (a, c, ctx12) -> (d, e, f)
]
```

### `hir.func`

A regular function consists of:

- a symbol visibility
- a symbol name, such as `@hello`
- a list of block arguments, such as `(%a, %b, %c)`
- a list of result names, such as `(d, e, f)`
- a body region describing the computation, terminated with `hir.return`

All block arguments have MLIR type `!hir.any`, which is omitted in the assembly format.
The SSA name of each block argument matches the name of the corresponding function argument.
Types of block arguments are not passed into the function as distinct inputs, but can be obtained through a `hir.type_of` op.
When a call site wants to specialize a function for concrete argument and result types, it inserts `hir.coerce_type` ops immediately after the block arguments and immediately before return ops, and inserts unification ops into the return ops' type operands.

The syntax of a function looks roughly like this:

```mlir
hir.func @hello(%a, %b, %c) -> (d, e, f) {
  %a.type = hir.type_of %a
  %b.type = hir.type_of %b
  %c.type = hir.type_of %c
  // ...
  hir.return %d, %e, %f : %d.type, %e.type, %f.type
}
```

### `hir.call`

A regular function call consists of:

- a symbol name of the function to call, such as `@hello`
- a list of arguments to pass to the function, such as `(%a, %b, %c)`
- a list of argument types, such as `(%a.type, %b.type, %c.type)`
- a list of result types, such as `(%d.type, %e.type, %f.type)`

The callee may be an `hir.func` or `hir.multiphase_func` operation.
The number of arguments and results must match the callee's block arguments and return values.
When all argument and result types are constants, the call can be replaced with a `mir.call` operation with concrete types baked into the op.
Unlike `hir.unified_call`, which references a unified function across all phases and carries per-argument phase annotations, `hir.call` invokes a single concrete function with no phase distinctions.

The syntax of a call looks roughly like this:

```mlir
%d, %e, %f = hir.call @hello(%a, %b, %c) : (%a.type, %b.type, %c.type) -> (%d.type, %e.type, %f.type)
```

## Calls

Call operations must be split according to the phases indicated on the arguments and results of the callee.

### Unified

Consider the following unified IR which calls the `@foo` function described above:

```mlir
hir.unified_func @bar() {
  hir.signature () -> ()
} {
  %int_type = hir.int_type                                                           // any phase
  %type_type = hir.type_type                                                         // any phase
  %a.type = hir.unified_call @doU() : () -> (%type_type: 0)                          // phase -4
  %a = hir.unified_call @makeA() : () -> (%a.type: 0)                                // phase -3
  %a2 = hir.unified_call @doV(%a) : (%a.type: 0) -> (%a.type: 0)                     // phase -2
  %b.type = hir.unified_call @doW(%a2) : (%a.type: 0) -> (%type_type: 0)             // phase -2
  %b = hir.unified_call @makeB() : () -> (%b.type: 0)                                // phase -1
  %c = hir.unified_call @foo(%a, %b) : (%a.type: -3, %b.type: -1) -> (%int_type: 0)  // phases -3, -1, 0
  hir.unified_call @consumeC(%c) : (%int_type: 0) -> ()                              // phase 0
  hir.return
}
```

A single unified call invokes `@foo`.
The argument and result phases of the call dictate most of the other phases in the surrounding operations.
The calls to `@doU`, `@doV`, and `@doW` compute types for `%a` and `%b`, and were likely inlined from foo's signature region into the call site by the CheckCalls pass earlier.
We omit a few `hir.const` ops to bunch calls to these functions more closely together into phases.
This is just for illustrative purposes to highlight that the phase structure around a call to `@foo` can differ significantly from the phase structure of `@foo` itself.
The interesting additions are calls to `@makeA`, `@makeB`, and `@consumeC`.
Since the type of `%b` is dependent on the type of `%a`, an earlier type inference pass has likely propagated the type of `%b` onto the result of the call to `@makeB`.

The above example is roughly equivalent to the following input:

```silicon
fn bar() {
  consumeC(foo(makeA(), makeB()))
}
```

### Split

Consider the previous example split into separate phases:

```mlir
hir.split_func @bar() {
  hir.signature () -> ()
} [
  0: @bar.0  // since no phase constraints from arguments/results, default to phase 0
]
hir.multiphase_func @bar.0() [
  @bar.0a,  // implied () -> (ctx0ab)
  @bar.0b,  // implied (ctx0ab) -> (ctx0bc)
  @bar.0c,  // implied (ctx0bc) -> (ctx0cd)
  @bar.0d,  // implied (ctx0cd) -> (ctx0de)
  @bar.0e   // implied (ctx0de) -> ()
]
hir.func @bar.0a() -> (ctx0ab) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  %a.type = hir.call @doU() : () -> (%type_type)  // phase -4
  %ctx0ab = hir.opaque_pack (%a.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @bar.0b(%ctx0ab) -> (ctx0bc) {
  %opaque_type = hir.opaque_type
  %a.type = hir.opaque_unpack %ctx0ab
  %a = hir.call @makeA() : () -> (%a.type)                        // phase -3
  %foo.ctx01 = hir.call @foo.0(%a) : (%a.type) -> (%opaque_type)  // phase -3
  %ctx0bc = hir.opaque_pack (%a, %foo.ctx01)
  hir.return %ctx0bc : %opaque_type
}
hir.func @bar.0c(%ctx0bc) -> (ctx0cd) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  %a, %foo.ctx01 = hir.opaque_unpack %ctx0bc
  %a.type = hir.type_of %a
  %a2 = hir.call @doV(%a) : (%a.type) -> (%a.type)          // phase -2
  %b.type = hir.call @doW(%a2) : (%a.type) -> (%type_type)  // phase -2
  %ctx0cd = hir.opaque_pack (%b.type, %foo.ctx01)
  hir.return %ctx0cd : %opaque_type
}
hir.func @bar.0d(%ctx0cd) -> (ctx0de) {
  %opaque_type = hir.opaque_type
  %b.type, %foo.ctx01 = hir.opaque_unpack %ctx0cd
  %b = hir.call @makeB() : () -> (%b.type)                                                  // phase -1
  %foo.ctx12 = hir.call @foo.1(%b, %foo.ctx01) : (%b.type, %opaque_type) -> (%opaque_type)  // phase -1
  %ctx0de = hir.opaque_pack (%foo.ctx12)
  hir.return %ctx0de : %opaque_type
}
hir.func @bar.0e(%ctx0de) -> () {
  %int_type = hir.int_type
  %opaque_type = hir.opaque_type
  %foo.ctx12 = hir.opaque_unpack %ctx0de
  %c = hir.call @foo.2(%foo.ctx12) : (%opaque_type) -> (%int_type)  // phase 0
  hir.call @consumeC(%c) : (%int_type) -> ()                        // phase 0
  hir.return
}
```

The original `@bar` unified function had 5 internal phases in total, but none of them were externally visible since the function has no arguments or results.
Therefore, only a single default phase 0 is listed in the corresponding split function.
All 5 internal phases are absorbed into a multiphase function, which allows all but the last phase to be executed at compile time.

Note how the unified call to `@foo` has been split up into three separate calls to the splits listed in `hir.split_func @foo`.
Even if foo has been precompiled as a library and partially evaluated and specialized, the split function still allows the unified bar function to know which phases the call to foo has to be split up into, and which functions correspond to each phase.

## Phased Evaluation

The phases ascribed to split and multiphase functions describe an execution schedule in which the compile-time evaluation of code should occur.
A fixed pipeline of compiler passes takes the earliest phase, executes it, and uses its results to specialize functions for the next phase.
The compiler executes this pipeline in a loop until no more multiphase functions remain.
If we are compiling a design (as opposed to a library for later reuse), the compiler marks all but the main function as private.
Each iteration through the pipeline performs roughly the following steps:

1.  **Canonicalize.**
    Propagate and fold constant values.
    This includes pushing unification ops through trivially equivalent operations and substituting inferrable ops.
    Also run the symbol dead code elimination pass to clean up any functions that may have become unused.

2.  **Unify.**
    Ensure that all unification ops have been resolved and emit errors for values and types that cannot be unified.
    This is the user-facing mechanism that produces error messages about incompatible types.

3.  **Lower HIR to MIR.**
    Lower functions to MIR if all ops in the body have type operands defined entirely by `ConstantLike` operations.

4.  **Pick functions to evaluate.**
    Collect the first sub-function from all executable multiphase functions.
    A multiphase function is executable if it does not have any `first` arguments, and the first phase function and its entire transitive call graph has been lowered to MIR.
    Note that this means that none of the called functions are multiphase functions, which are HIR dialect ops; multiphase functions can only be executed once they have been fully reduced to a single function and lowered to MIR.

5.  **Lower MIR to LLVM IR.**
    Lower all functions picked for evaluation to the LLVM dialect, including transitively any functions they call.
    The lowered LLVM functions reside in a separate MLIR top-level module, since the JIT execution engine requires the entire module to be in LLVM.
    If all call sites of a function reside within the transitive call graphs of sub-functions picked for evaluation, the function can simply be moved to the other module and lowered.
    If there are call sites that remain in HIR or MIR, the function has to be cloned for LLVM lowering.
    This ensures that a function can both be used in a compile-time evaluated context, but also in a context where it ends up being lowered to CIRCT IR later.

6.  **Evaluate.**
    Call the picked functions using MLIR's execution engine, which uses LLVM's JIT compiler under the hood.
    Since we are incrementally adding new LLVM functions, make use of any object caching offered by the infrastructure.

7.  **Materialize results.**
    Map the results of the picked functions to MLIR attributes.
    Then remove the picked functions from their parent multiphase function's phase list, and specialize the subsequent phase with the results.
    If only one sub-function remains in a multiphase function's list, replace all uses of the multiphase function's symbol with that sub-function's symbol.
    This step reduces individual multiphase functions; the next step propagates the resulting constants through call sites.

8.  **Specialize functions.**
    Transitively specialize any function calls by copying constant arguments into a distinct copy of the function.
    Don't create a copy if specializing the last remaining call of a private function, but modify that function directly instead.
    If a function has been specialized for the exact same constant argument before, reuse that function instead of creating a redundant copy.
    This ensures that the results obtained during evaluation get propagated through the call graph and are used to specialize functions.

9.  **Repeat.**
    If there are any remaining multiphase functions and at least one function was picked and evaluated in this iteration of the pipeline, go back to step 1.

After exiting the loop, the IR has been compile-time evaluated as far as possible.
Discard the LLVM top-level module at this point.

If we are compiling a library, we can simply store the IR in its current state into an MLIRBC file.

If we are compiling a design, run an additional compiler pass to ensure that all multiphase functions have been fully evaluated.
Any unused functions must have been deleted and all others fully compile-time evaluated down to a single phase at this point.
The loop must also have lowered the entire input to MIR.
If this is not the case, report a compiler bug to the user and abort.

## Worked Example

This section describes step-by-step how the previous examples of `@foo` and `@bar` are processed by the phased evaluation pipeline.
We assume compilation as a design, such that any unused functions can be eliminated.
We also assume that any of the `@do*`, `@make*`, and the `@consumeC` function are already lowered to MIR and LLVM; we skip their implementation details for brevity.

### First Iteration

The IR is already fully canonicalized and unified.
Lower `@foo.0a` and `@bar.0a` to MIR since all type operands in the body are constants.

```mlir
mir.func @foo.0a() -> (ctx0ab: !mir.opaque) {
  %a.type = mir.call @doU() : () -> (!mir.type)
  %ctx0ab = mir.opaque_pack (%a.type) : (!mir.type)
  mir.return %ctx0ab : !mir.opaque
}

mir.func @bar.0a() -> (ctx0ab: !mir.opaque) {
  %a.type = mir.call @doU() : () -> (!mir.type)
  %ctx0ab = mir.opaque_pack (%a.type) : (!mir.type)
  mir.return %ctx0ab : !mir.opaque
}
```

These are the first sub-functions of multiphase functions `@foo.0` and `@bar.0`.
They have no arguments and the entire transitive call graph has been lowered to MIR.
Pick these for execution.
Then lower them to LLVM in a separate module.

```mlir
llvm.func @foo.0a() -> !llvm.ptr {
  %a.type = llvm.call @doU() : () -> (!llvm.ptr)
  %ctx0ab = llvm.call @opaque_pack.type(%a.type) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %ctx0ab : !llvm.ptr
}

llvm.func @bar.0a() -> !llvm.ptr {
  %a.type = llvm.call @doU() : () -> (!llvm.ptr)
  %ctx0ab = llvm.call @opaque_pack.type(%a.type) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %ctx0ab : !llvm.ptr
}
```

We omit the implementation of opaque value packing.
This would be implemented as a heap allocation of a struct with fields corresponding to the packed operands.
Similarly, we simply represent `!mir.type`s as heap-allocated `!llvm.ptr`, even though we may choose to implement these as a form of struct or union.

Evaluate the two functions.
We execute them through MLIR's execution engine.
Then convert the resulting packed opaque values from the runtime representation to an MLIR attribute that we can materialize.
This will likely have to be done by generating additional LLVM glue code that unpacks the opaque value again, and calls helper functions registered with the execution engine that construct MLIR attributes.
We omit this as an implementation detail.
The resulting MLIR attributes are expected to look like this, assuming that `@doU` returns a `uint<42>` type:

```mlir
#foo.0a = #mir.opaque<#mir.type<!mir.uint<42>>>
#bar.0a = #mir.opaque<#mir.type<!mir.uint<42>>>
```

Remove the evaluated sub-functions from the surrounding multiphase function and specialize the next sub-functions with the materialized results:

```mlir
// removed hir.multiphase_func @foo.0
// removed hir.func @foo.0a
hir.func @foo.0b(%a) -> (ctx01) {
  %opaque_type = hir.opaque_type
  // ctx0ab has been inlined and converted to HIR constant
  %a.type = hir.mir_type !mir.uint<42>
  %a0 = hir.coerce_type %a : %a.type
  %a1 = hir.call @doV(%a0) : (%a.type) -> (%a.type)
  %ctx01 = hir.opaque_pack (%a1, %a.type)
  hir.return %ctx01 : %opaque_type
}

// removed hir.func @bar.0a
hir.multiphase_func @bar.0() [
  @bar.0b,  // implied () -> (ctx0bc)
  @bar.0c,  // implied (ctx0bc) -> (ctx0cd)
  @bar.0d,  // implied (ctx0cd) -> (ctx0de)
  @bar.0e   // implied (ctx0de) -> ()
]
hir.func @bar.0b() -> (ctx0bc) {
  %opaque_type = hir.opaque_type
  // ctx0ab has been inlined and converted to HIR constant
  %a.type = hir.mir_type !mir.uint<42>
  %a = hir.call @makeA() : () -> (%a.type)
  // call to @foo.0 replaced with @foo.0b
  %foo.ctx01 = hir.call @foo.0b(%a) : (%a.type) -> (%opaque_type)
  %ctx0bc = hir.opaque_pack (%a, %foo.ctx01)
  hir.return %ctx0bc : %opaque_type
}
```

The multiphase function `@foo.0` gets reduced down to a single phase.
We therefore replace all occurrences of it with direct calls to the specialized `@foo.0b`.

The multiphase functions are the only users of the `@foo.0b` and `@bar.0b` functions, which allows us to directly modify these functions instead of creating a specialized copy.

The packed opaque values obtained from execution are MIR attributes, which we materialize as constants in the HIR dialect.

### Second Iteration

Lower `@foo.0b` and `@bar.0b` to MIR since all type operands in the body are constants, and the type coercion on the block arguments provides for a concrete type to assign to those arguments.

```mlir
mir.func @foo.0b(%a: !mir.uint<42>) -> (ctx01: !mir.opaque) {
  %a.type = mir.constant_type !mir.uint<42>
  %a1 = mir.call @doV(%a) : (!mir.uint<42>) -> (!mir.uint<42>)
  %ctx01 = mir.opaque_pack (%a1, %a.type) : (!mir.uint<42>, !mir.type)
  mir.return %ctx01 : !mir.opaque
}

mir.func @bar.0b() -> (ctx0bc: !mir.opaque) {
  %a.type = mir.constant_type !mir.uint<42>
  %a = mir.call @makeA() : () -> (!mir.uint<42>)
  %foo.ctx01 = mir.call @foo.0b(%a) : (!mir.uint<42>) -> (!mir.opaque)
  %ctx0bc = mir.opaque_pack (%a, %foo.ctx01) : (!mir.uint<42>, !mir.type)
  mir.return %ctx0bc : !mir.opaque
}
```

The `@bar.0b` function is again the first sub-function of the multiphase function `@bar.0`.
It has no arguments, and the transitive call graph has been lowered to MIR.
Therefore, pick `@bar.0b` for execution.
Note that `@foo.0b` is no longer part of a multiphase function, but it gets transitively called by `@bar.0b`.
Then lower the picked function and any transitively called functions to LLVM:

```mlir
llvm.func @foo.0b(%a: i42) -> (ctx01: !llvm.ptr) {
  %c42_i64 = llvm.mlir.constant 42 : i64
  %a.type = llvm.call @make_int_type(%c42_64) : (i64) -> !llvm.ptr
  %a1 = llvm.call @doV(%a) : (i42) -> (i42)
  %ctx01 = llvm.call @opaque_pack.i42.type(%a1, %a.type) : (i42, !llvm.ptr) -> !llvm.ptr
  llvm.return %ctx01 : !llvm.ptr
}

llvm.func @bar.0b() -> (ctx0bc: !llvm.ptr) {
  %c42_i64 = llvm.mlir.constant 42 : i64
  %a.type = llvm.call @make_int_type(%c42_64) : (i64) -> !llvm.ptr
  %a = llvm.call @makeA() : () -> (i42)
  %foo.ctx01 = llvm.call @foo.0b(%a) : (i42) -> (!llvm.ptr)
  %ctx0bc = llvm.call @opaque_pack.i42.type(%a, %foo.ctx01) : (i42, !llvm.ptr) -> !llvm.ptr
  llvm.return %ctx0bc : !llvm.ptr
}
```

Execute the `@bar.0b` function using MLIR's execution engine.
The resulting MLIR attributes are expected to look like this, assuming that `@makeA` returned 1337 and `@doV` returned 1338:

```mlir
#bar.0b = #mir.opaque<
  #mir.uint<1337, 42>,
  #mir.opaque<#mir.uint<1338, 42>, #mir.type<!mir.uint<42>>>
>
```

Remove the evaluated function and specialize the next function with the materialized results:

```mlir
// removed hir.func @foo.0b
// removed hir.func @bar.0b
hir.multiphase_func @bar.0() [
  @bar.0c,  // implied () -> (ctx0cd)
  @bar.0d,  // implied (ctx0cd) -> (ctx0de)
  @bar.0e   // implied (ctx0de) -> ()
]
hir.func @bar.0c() -> (ctx0cd) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  // ctx0bc has been inlined and converted to HIR constants
  %a = hir.mir_constant #mir.uint<1337, 42>
  %foo.ctx01 = hir.mir_constant #mir.opaque<#mir.uint<1338, 42>, #mir.type<!mir.uint<42>>>
  %a.type = hir.type_of %a
  %a2 = hir.call @doV(%a) : (%a.type) -> (%a.type)
  %b.type = hir.call @doW(%a2) : (%a.type) -> (%type_type)
  %ctx0cd = hir.opaque_pack (%b.type, %foo.ctx01)
  hir.return %ctx0cd : %opaque_type
}
```

The `@foo.0b` and `@bar.0b` functions have no more users and can be removed.

### Third Iteration

Canonicalization simplifies a few ops after specialization:

```mlir
hir.func @bar.0c ... {
  // ...
  %a = hir.mir_constant #mir.uint<1337, 42>
  %a.type = hir.mir_type !mir.uint<42>  // from hir.type_of %a
  // ...
}
```

Lower `@bar.0c` to MIR since all type operands in the body are constants:

```mlir
mir.func @bar.0c() -> (ctx0cd: !mir.opaque) {
  %a = mir.constant 1337 : !mir.uint<42>
  %a2 = mir.call @doV(%a) : (!mir.uint<42>) -> (!mir.uint<42>)
  %b.type = mir.call @doW(%a2) : (!mir.uint<42>) -> (!mir.type)
  %foo.ctx01 = mir.constant #mir.opaque<#mir.uint<1338, 42>, #mir.type<!mir.uint<42>>> : !mir.opaque
  %ctx0cd = mir.opaque_pack (%b.type, %foo.ctx01) : (!mir.type, !mir.opaque)
  mir.return %ctx0cd : !mir.opaque
}
```

`@bar.0c` is the first sub-function of `@bar.0`.
It takes no arguments and the entire transitive call graph has been lowered to MIR.
Pick it for execution.
Then lower to LLVM in a separate module:

```mlir
llvm.func @bar.0c() -> !llvm.ptr {
  %c1337 = llvm.mlir.constant(1337 : i42) : i42
  %a2 = llvm.call @doV(%c1337) : (i42) -> i42
  %b.type = llvm.call @doW(%a2) : (i42) -> !llvm.ptr
  %c1338 = llvm.mlir.constant(1338 : i42) : i42
  %c42 = llvm.mlir.constant(42 : i64) : i64
  %a.type = llvm.call @make_int_type(%c42) : (i64) -> !llvm.ptr
  %foo.ctx01 = llvm.call @opaque_pack.i42.type(%c1338, %a.type) : (i42, !llvm.ptr) -> !llvm.ptr
  %ctx0cd = llvm.call @opaque_pack.type.opaque(%b.type, %foo.ctx01) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
  llvm.return %ctx0cd : !llvm.ptr
}
```

Execute `@bar.0c`.
The resulting MLIR attributes are expected to look like this, assuming that `@doV` returns 1338 and `@doW` returns a `uint<9001>` type:

```mlir
#bar.0c = #mir.opaque<
  #mir.type<!mir.uint<9001>>,
  #mir.opaque<#mir.uint<1338, 42>, #mir.type<!mir.uint<42>>>
>
```

Remove the evaluated sub-function from the surrounding multiphase function and specialize the next sub-function with the materialized results:

```mlir
// removed hir.func @bar.0c
hir.multiphase_func @bar.0() [
  @bar.0d,  // implied () -> (ctx0de)
  @bar.0e   // implied (ctx0de) -> ()
]
hir.func @bar.0d() -> (ctx0de) {
  %opaque_type = hir.opaque_type
  // ctx0cd has been inlined and converted to HIR constants
  %b.type = hir.mir_type !mir.uint<9001>
  %foo.ctx01 = hir.mir_constant #mir.opaque<#mir.uint<1338, 42>, #mir.type<!mir.uint<42>>>
  %b = hir.call @makeB() : () -> (%b.type)
  %foo.ctx12 = hir.call @foo.1(%b, %foo.ctx01) : (%b.type, %opaque_type) -> (%opaque_type)
  %ctx0de = hir.opaque_pack (%foo.ctx12)
  hir.return %ctx0de : %opaque_type
}
```

`@bar.0d` calls `@foo.1`, which is still a multiphase function.
The `%foo.ctx01` argument is now a known constant, so the specialization step inlines it into `@foo.1a` as the `first` argument of `@foo.1`:

```mlir
hir.multiphase_func @foo.1(last b) -> (ctx12) [
  @foo.1a,  // implied () -> (ctx1ab); ctx01 has been inlined
  @foo.1b   // implied (b, ctx1ab) -> (ctx12)
]
hir.func @foo.1a() -> (ctx1ab) {
  %type_type = hir.type_type
  %opaque_type = hir.opaque_type
  // ctx01 has been inlined and converted to HIR constants
  %a2 = hir.mir_constant #mir.uint<1338, 42>
  %a.type = hir.mir_type !mir.uint<42>
  %b.type = hir.call @doW(%a2) : (%a.type) -> (%type_type)
  %ctx1ab = hir.opaque_pack (%b.type)
  hir.return %ctx1ab : %opaque_type
}

hir.func @bar.0d() -> (ctx0de) {
  %opaque_type = hir.opaque_type
  %b.type = hir.mir_type !mir.uint<9001>
  %b = hir.call @makeB() : () -> (%b.type)
  // foo.ctx01 has been inlined into foo.1
  %foo.ctx12 = hir.call @foo.1(%b) : (%b.type) -> (%opaque_type)
  %ctx0de = hir.opaque_pack (%foo.ctx12)
  hir.return %ctx0de : %opaque_type
}
```

`@foo.1a` now takes no arguments and can be evaluated in the next iteration.
`@bar.0d` cannot yet be evaluated since its callee `@foo.1` remains a multiphase function.

### Fourth Iteration

Lower `@foo.1a` to MIR since all type operands in the body are constants:

```mlir
mir.func @foo.1a() -> (ctx1ab: !mir.opaque) {
  %a2 = mir.constant 1338 : !mir.uint<42>
  %b.type = mir.call @doW(%a2) : (!mir.uint<42>) -> (!mir.type)
  %ctx1ab = mir.opaque_pack (%b.type) : (!mir.type)
  mir.return %ctx1ab : !mir.opaque
}
```

`@foo.1a` is the first sub-function of `@foo.1`.
It takes no arguments and the entire transitive call graph has been lowered to MIR.
Pick it for execution.
Then lower to LLVM in a separate module:

```mlir
llvm.func @foo.1a() -> !llvm.ptr {
  %c1338 = llvm.mlir.constant(1338 : i42) : i42
  %b.type = llvm.call @doW(%c1338) : (i42) -> !llvm.ptr
  %ctx1ab = llvm.call @opaque_pack.type(%b.type) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %ctx1ab : !llvm.ptr
}
```

Execute `@foo.1a`.
`@doW` receives the same input as in the third iteration (1338 as `uint<42>`) and returns `uint<9001>`.
The resulting MLIR attribute is:

```mlir
#foo.1a = #mir.opaque<#mir.type<!mir.uint<9001>>>
```

Remove the evaluated sub-function from `@foo.1` and specialize `@foo.1b` with the materialized result:

```mlir
// removed hir.multiphase_func @foo.1
// removed hir.func @foo.1a
hir.func @foo.1b(%b) -> (ctx12) {
  %opaque_type = hir.opaque_type
  // ctx1ab has been inlined and converted to HIR constants
  %b.type = hir.mir_type !mir.uint<9001>
  %b0 = hir.coerce_type %b : %b.type
  %b1 = hir.call @doX(%b0) : (%b.type) -> (%b.type)
  %ctx12 = hir.opaque_pack (%b1, %b.type)
  hir.return %ctx12 : %opaque_type
}
```

The multiphase function `@foo.1` gets reduced down to a single phase.
We therefore replace all occurrences of it with direct calls to `@foo.1b`.
`@bar.0d`'s call to `@foo.1` is updated accordingly:

```mlir
hir.func @bar.0d() -> (ctx0de) {
  %opaque_type = hir.opaque_type
  %b.type = hir.mir_type !mir.uint<9001>
  %b = hir.call @makeB() : () -> (%b.type)
  // call to @foo.1 replaced with @foo.1b
  %foo.ctx12 = hir.call @foo.1b(%b) : (%b.type) -> (%opaque_type)
  %ctx0de = hir.opaque_pack (%foo.ctx12)
  hir.return %ctx0de : %opaque_type
}
```

`@bar.0d` now calls only regular functions and takes no arguments.
It can be evaluated in the next iteration.

### Fifth Iteration

Lower `@bar.0d` and `@foo.1b` to MIR since all type operands in their bodies are constants:

```mlir
mir.func @foo.1b(%b: !mir.uint<9001>) -> (ctx12: !mir.opaque) {
  %b1 = mir.call @doX(%b) : (!mir.uint<9001>) -> (!mir.uint<9001>)
  %b.type = mir.constant_type !mir.uint<9001>
  %ctx12 = mir.opaque_pack (%b1, %b.type) : (!mir.uint<9001>, !mir.type)
  mir.return %ctx12 : !mir.opaque
}

mir.func @bar.0d() -> (ctx0de: !mir.opaque) {
  %b = mir.call @makeB() : () -> (!mir.uint<9001>)
  %foo.ctx12 = mir.call @foo.1b(%b) : (!mir.uint<9001>) -> (!mir.opaque)
  %ctx0de = mir.opaque_pack (%foo.ctx12) : (!mir.opaque)
  mir.return %ctx0de : !mir.opaque
}
```

`@bar.0d` is the first sub-function of `@bar.0`.
It takes no arguments and the entire transitive call graph has been lowered to MIR.
Pick it for execution.
Then lower to LLVM in a separate module:

```mlir
llvm.func @foo.1b(%b: i9001) -> !llvm.ptr {
  %b1 = llvm.call @doX(%b) : (i9001) -> i9001
  %c9001 = llvm.mlir.constant(9001 : i64) : i64
  %b.type = llvm.call @make_int_type(%c9001) : (i64) -> !llvm.ptr
  %ctx12 = llvm.call @opaque_pack.i9001.type(%b1, %b.type) : (i9001, !llvm.ptr) -> !llvm.ptr
  llvm.return %ctx12 : !llvm.ptr
}

llvm.func @bar.0d() -> !llvm.ptr {
  %b = llvm.call @makeB() : () -> i9001
  %foo.ctx12 = llvm.call @foo.1b(%b) : (i9001) -> !llvm.ptr
  %ctx0de = llvm.call @opaque_pack.opaque(%foo.ctx12) : (!llvm.ptr) -> !llvm.ptr
  llvm.return %ctx0de : !llvm.ptr
}
```

Execute `@bar.0d`.
The resulting MLIR attributes are expected to look like this, assuming that `@makeB` returns 17 and `@doX` returns 18:

```mlir
#bar.0d = #mir.opaque<
  #mir.opaque<#mir.uint<18, 9001>, #mir.type<!mir.uint<9001>>>
>
```

Remove the evaluated sub-function from the surrounding multiphase function and specialize the next sub-function with the materialized results:

```mlir
// removed hir.multiphase_func @bar.0
// removed hir.func @bar.0d
hir.func @bar.0e() -> () {
  %int_type = hir.int_type
  %opaque_type = hir.opaque_type
  // ctx0de has been inlined and converted to HIR constants
  %foo.ctx12 = hir.mir_constant #mir.opaque<#mir.uint<18, 9001>, #mir.type<!mir.uint<9001>>>
  %c = hir.call @foo.2(%foo.ctx12) : (%opaque_type) -> (%int_type)
  hir.call @consumeC(%c) : (%int_type) -> ()
  hir.return
}
```

The multiphase function `@bar.0` gets reduced down to a single phase.
We therefore replace all occurrences of it with direct calls to `@bar.0e`.

The specialization step propagates the now-constant `%foo.ctx12` into `@foo.2`:

```mlir
hir.func @foo.2() -> (c) {
  %int_type = hir.int_type
  // ctx12 has been inlined and converted to HIR constants
  %b2 = hir.mir_constant #mir.uint<18, 9001>
  %b.type = hir.mir_type !mir.uint<9001>
  %c = hir.call @doY(%b2) : (%b.type) -> (%int_type)
  hir.return %c : %int_type
}

hir.func @bar.0e() -> () {
  %int_type = hir.int_type
  %opaque_type = hir.opaque_type
  // foo.ctx12 has been inlined into foo.2
  %c = hir.call @foo.2() : () -> (%int_type)
  hir.call @consumeC(%c) : (%int_type) -> ()
  hir.return
}
```

No multiphase functions remain and the pipeline loop exits.

### Aftermath

The LLVM top-level module is discarded.
The remaining phase 0 functions `@bar.0e` and `@foo.2` represent the final hardware behavior.
Since `@bar.0e` is the only phase in the split function `@bar`, it is renamed to `@bar`.
Lower the remaining functions to MIR.

```mlir
mir.func @foo.2() -> (c: !mir.int) {
  %b2 = mir.constant 18 : !mir.uint<9001>
  %c = mir.call @doY(%b2) : (!mir.uint<9001>) -> (!mir.int)
  mir.return %c : !mir.int
}

mir.func @bar() {
  %c = mir.call @foo.2() : () -> (!mir.int)
  mir.call @consumeC(%c) : (!mir.int) -> ()
  mir.return
}
```

Done — `@bar` is fully monomorphized.
All compile-time phases have been evaluated and their results baked into the final IR as constants.
The remainder of the compiler pipeline lowers this MIR representation to CIRCT to arrive at an actual hardware module.

## Deduplication

When const arguments propagate transitively across multiple call levels, the phased evaluation creates a tree of specializations.
Consider the following example:

```silicon
fn top(a: int) {
  middle(0, a);
  middle(1, a);
}
fn middle(const b: int, c: int) {
  leaf(b + 42, c);
  leaf(b + 1337, c);
}
fn leaf(const d: int, e: int) {
  print(d + e);
}
```

`top` calls `middle` twice with distinct const args (0 and 1).
`middle` computes derived const values (`b+42`, `b+1337`) and passes them to `leaf`.
This creates a tree of four leaf specializations:

- `top` → `middle(b=0)` → `leaf(d=42)`, `leaf(d=1337)`
- `top` → `middle(b=1)` → `leaf(d=43)`, `leaf(d=1338)`

After phase splitting, all three functions follow the same structural pattern as the worked example: a `hir.split_func` listing the caller-visible phases, with internal phases absorbed into `hir.multiphase_func` ops.
`top` has a single caller-visible phase since its only argument `a` is phase 0.

During phased evaluation, the compile-time sub-function `@top.0a` calls `@middle.0b` twice (with `b=0` and `b=1`), and each `@middle.0b` call invokes `@leaf.0b` twice with different `d` values.
The entire call tree executes in a single JIT batch, producing an opaque context that carries the computed constants for all four leaf instances.

When specializing the runtime functions, the pipeline creates distinct copies for each unique set of constant arguments:

- `@middle.1` is specialized twice: `@middle.b0` (for `b=0`) and `@middle.b1` (for `b=1`).
- `@leaf.1` is specialized four times: `@leaf.d42`, `@leaf.d1337`, `@leaf.d43`, and `@leaf.d1338`.

For trivial types and constants, the specialized function name includes the constant value directly.
For more complex structures, a short hash is used instead.

Each specialization has the const argument baked in as a constant:

```mlir
mir.func @leaf.d42(%e: !mir.int) {
  %d = mir.constant 42 : !mir.int
  %0 = mir.binary add %d, %e : !mir.int
  mir.call @print(%0) : (!mir.int) -> ()
  mir.return
}
// @leaf.d1337, @leaf.d43, @leaf.d1338 are identical with their respective d values
```

The final monomorphized IR is a flat set of `mir.func` ops with no remaining `split_func`, `multiphase_func`, or opaque operations:

```mlir
mir.func @top(%a: !mir.int) {
  mir.call @middle.b0(%a) : (!mir.int) -> ()
  mir.call @middle.b1(%a) : (!mir.int) -> ()
  mir.return
}

mir.func @middle.b0(%c: !mir.int) {
  mir.call @leaf.d42(%c) : (!mir.int) -> ()
  mir.call @leaf.d1337(%c) : (!mir.int) -> ()
  mir.return
}

mir.func @leaf.d42(%e: !mir.int) {
  %d = mir.constant 42 : !mir.int
  %0 = mir.binary add %d, %e : !mir.int
  mir.call @print(%0) : (!mir.int) -> ()
  mir.return
}
// @leaf.d1337, @leaf.d43, @leaf.d1338 follow the same pattern
// @middle.b1 calls @leaf.d43 and @leaf.d1338
```

The specialization step uses the key `(original function symbol, constant argument values)` to identify specializations.
Before creating a specialized copy, the compiler checks if a specialization with the same key already exists and reuses it.
In this example all four leaf specializations have distinct `d` values, so no deduplication occurs.
If two different call sites had both produced `d=42`, they would share the same specialized function instead of creating redundant copies.
