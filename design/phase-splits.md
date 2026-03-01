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

## JIT Batching

Multiple independent sub-functions can be compiled into a single LLVM module and JIT-executed together, minimizing the number of JIT round-trips.
When a sub-function contains `hir.call` ops to other functions, the compiler consults the callee's `hir.split_func` phase list to identify which function handles each caller-visible phase.
The resolved functions are then lowered to MIR together and compiled into a single LLVM module, allowing the JIT to transitively evaluate an entire subgraph of phase functions in one execution.

## Specialization and Monomorphization

Substituting known constants for a function's opaque context values (via `hir.opaque_unpack`) is _specialization_.
When all opaque context values flowing into a sub-function are known constants, the function is fully specialized and can be lowered to MIR.
An `hir.multiphase_func` is ready for reduction when its first sub-function has no arguments or all arguments are constants.

_Monomorphization_ creates a dedicated clone of a function for a specific set of constant arguments.
The specialization key is (original function symbol, constant arg values).
Before cloning, the compiler checks if a clone with the same key already exists and reuses it — this is how deduplication works.

## Library Compilation

When we compile an input into a library, we want to execute as many phases as possible.
An `hir.multiphase_func` with no `first` arguments can have its first sub-function evaluated immediately, progressively shrinking the function list.
For example, `@foo.0a` in the Split IR section above can be fully evaluated at compile time since it takes no arguments, and the result specializes `@foo.0b`.
This process continues until no more sub-functions can be evaluated without caller-provided arguments.
Once the library is compiled, we store it as an MLIR bytecode blob on disk.
A user may then use this library in their code compiled in a later invocation of the compiler.
At that point the new user code is still using `hir.unified_{func,call}` ops.
The library must retain enough information about how the individual phase splits combine to form a unified function to allow the later compiler invocation to resolve the user's unified calls to the precompiled phase splits in the library.

## Example 1

Consider the following input:

```silicon
fn foo(const a: int, b: int) -> (const x: int, y: int) {
  (a + a, b + b)
}
fn bar() -> int {
  let (x, y) = foo(21, 4501);
  x + y
}
```

### Unified IR

Running this through `silc --parse-only` should yield the following IR.
We want to support multiple and named results in functions, and allow for each result to have a distinct phase.
It's okay if the IR contains other type-of and inferrable ops, since parse-only does not run any canonicalizers.

```mlir
hir.unified_func @foo(%a: -1, %b: 0) -> (x: -1, y: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0, %0)
} {
  %0 = hir.int_type
  %1 = hir.binary %a, %a : %0
  %2 = hir.binary %b, %b : %0
  hir.unified_return %1, %2 : %0, %0
}

hir.unified_func @bar() -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 21
  %2 = hir.constant_int 4501
  %x, %y = hir.unified_call @foo(%1, %2) : (%0: -1, %0: 0) -> (%0: -1, %0: 0)
  %3 = hir.binary %x, %y : %0
  hir.unified_return %3 : %0
}
```

### Split IR

Running the above through the SplitPhases pass produces `hir.split_func`, `hir.multiphase_func`, and `hir.func` ops as described in the Split IR section above.

**Foo** has caller-visible phases {-1, 0}.
The split function lists a function for each phase.
Each function absorbs any internal phases as a multiphase function.

```mlir
hir.split_func @foo(%a: -1, %b: 0) -> (x: -1, y: 0) {
  %int_type = hir.int_type
  hir.signature (%int_type, %int_type) -> (%int_type, %int_type)
} [
  -1: @foo.0,  // implied (a) -> (x, ctx01)
  0: @foo.1    // implied (b, ctx01) -> (y)
]
hir.multiphase_func @foo.0(last a) -> (x, ctx01) [
  @foo.0a,  // implied () -> (ctx0ab)
  @foo.0b   // implied (a, ctx0ab) -> (x, ctx01)
]
hir.func @foo.0a() -> (ctx0ab) {
  %opaque_type = hir.opaque_type
  %a.type = hir.int_type
  %x.type = hir.int_type
  %ctx0ab = hir.opaque_pack (%a.type, %x.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @foo.0b(%a, %ctx0ab) -> (x, ctx01) {
  %opaque_type = hir.opaque_type
  %a.type, %x.type = hir.opaque_unpack %ctx0ab
  %a0 = hir.coerce_type %a : %a.type
  %x = hir.binary %a0, %a0 : %x.type
  %b.type = hir.int_type
  %y.type = hir.int_type
  %ctx01 = hir.opaque_pack (%b.type, %y.type)
  hir.return %x, %ctx01 : %x.type, %opaque_type
}
hir.multiphase_func @foo.1(last b, first ctx01) -> (y) [
  @foo.1a,  // implied (ctx01) -> (ctx1ab)
  @foo.1b   // implied (b, ctx1ab) -> (y)
]
hir.func @foo.1a(%ctx01) -> (ctx1ab) {
  %opaque_type = hir.opaque_type
  %b.type, %y.type = hir.opaque_unpack %ctx01
  %ctx1ab = hir.opaque_pack (%b.type, %y.type)
  hir.return %ctx1ab : %opaque_type
}
hir.func @foo.1b(%b, %ctx1ab) -> (y) {
  %b.type, %y.type = hir.opaque_unpack %ctx1ab
  %b0 = hir.coerce_type %b : %b.type
  %y = hir.binary %b0, %b0 : %y.type
  hir.return %y : %y.type
}
```

`@foo.0` is a multiphase function with two sub-functions.
`@foo.0a` computes the types of `a` and `x` at phase -2.
Since `@foo.0a` takes no arguments, it can be fully evaluated at compile time even when compiling to a library.
`@foo.0b` receives `a` and the opaque type context, coerces `a` to the computed type, computes `x = a + a`, and packs the types needed by the next phase (`b.type`, `y.type`) into `ctx01`.

`@foo.1` absorbs the internal phase between receiving `ctx01` and receiving `b`.
`@foo.1a` unpacks the context from the previous phase and repacks it for `@foo.1b`.
In this simple example, `@foo.1a` trivially passes through the type values; in a more complex example, it could compute derived types (as in the authoritative `@foo.1a` which calls `@doW` to derive `b.type`).
`@foo.1b` receives `b` and the type context, coerces `b`, and computes `y = b + b`.

**Bar** has no arguments and one result at phase 0, so it has a single caller-visible phase {0}.
All of bar's internal phases are absorbed into a multiphase function: one sub-function for the compile-time phase (calling foo's phase -1 function), and one for the runtime phase (calling foo's phase 0 function).

```mlir
hir.split_func @bar() -> (result: 0) {
  %int_type = hir.int_type
  hir.signature () -> (%int_type)
} [
  0: @bar.0
]
hir.multiphase_func @bar.0() -> (result) [
  @bar.0a,  // implied () -> (ctx0ab)
  @bar.0b   // implied (ctx0ab) -> (result)
]
hir.func @bar.0a() -> (ctx0ab) {
  %opaque_type = hir.opaque_type
  %int_type = hir.int_type
  %21 = hir.constant_int 21
  %x, %foo.ctx01 = hir.call @foo.0(%21) : (%int_type) -> (%int_type, %opaque_type)
  %result.type = hir.int_type
  %ctx0ab = hir.opaque_pack (%x, %foo.ctx01, %result.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @bar.0b(%ctx0ab) -> (result) {
  %opaque_type = hir.opaque_type
  %int_type = hir.int_type
  %x, %foo.ctx01, %result.type = hir.opaque_unpack %ctx0ab
  %4501 = hir.constant_int 4501
  %y = hir.call @foo.1(%4501, %foo.ctx01) : (%int_type, %opaque_type) -> (%int_type)
  %result = hir.binary %x, %y : %result.type
  hir.return %result : %result.type
}
```

Note how the unified call to `@foo` has been split into two separate calls: `@bar.0a` calls `@foo.0` (phase -1) to compute `x`, and `@bar.0b` calls `@foo.1` (phase 0) to compute `y`.
The opaque context `%foo.ctx01` flows from `@foo.0` to `@foo.1` via bar's own opaque pack.
Even if foo has been precompiled as a library and partially evaluated, the split function still allows bar to know which functions correspond to each phase.

### Library Compilation

When compiling foo as a library, we evaluate multiphase functions whose first sub-function has no arguments.

**Step 1: Evaluate `@foo.0a`.**
`@foo.0a` takes no arguments — it can be fully evaluated.
Lower to MIR, JIT-execute.
Result: `ctx0ab = opaque_pack(int_type, int_type)`.
Specialize `@foo.0b` by inlining the opaque context constants → produces `@foo.0b'` that only takes `(%a)`:

```mlir
hir.func @foo.0b'(%a) -> (x, ctx01) {
  %opaque_type = hir.opaque_type
  %a.type = hir.int_type    // inlined from ctx0ab
  %x.type = hir.int_type    // inlined from ctx0ab
  %a0 = hir.coerce_type %a : %a.type
  %x = hir.binary %a0, %a0 : %x.type
  %b.type = hir.int_type
  %y.type = hir.int_type
  %ctx01 = hir.opaque_pack (%b.type, %y.type)
  hir.return %x, %ctx01 : %x.type, %opaque_type
}
```

Update `@foo.0`: list shrinks from `[@foo.0a, @foo.0b]` to just `[@foo.0b']`.
Since only one function remains, `@foo.0` is replaced with `@foo.0b'`.

**Step 2: Check `@foo.1a`.**
`@foo.1a` takes `ctx01` as a `first` argument.
`ctx01` depends on a caller providing `%a` — we can't evaluate `@foo.1a` without knowing `ctx01`.
`@foo.1` remains a multiphase function, waiting for a caller to provide the context.

Pre-specialization means that when a caller later provides `%a`, `@foo.0b'` computes `x` and `ctx01` directly — the type computation from `@foo.0a` has already been baked in.

### Progressive Evaluation

The following traces the full evaluation of both foo and bar, step by step.
We assume foo has been library-compiled as shown above.

#### Initial state

Foo after library compilation — `@foo.0` has been reduced to `@foo.0b'`:

```mlir
hir.split_func @foo(%a: -1, %b: 0) -> (x: -1, y: 0) { ... } [
  -1: @foo.0b',  // (a) -> (x, ctx01)
  0: @foo.1      // still multiphase: (b, ctx01) -> (y)
]
```

Bar after SplitPhases:

```mlir
hir.split_func @bar() -> (result: 0) { ... } [
  0: @bar.0  // multiphase: () -> (result)
]
```

#### Step 1: Evaluate `@bar.0a`

`@bar.0a` takes no arguments → can evaluate.
Inside `@bar.0a`, `hir.call @foo.0(%21)` now resolves to `@foo.0b'(21)`, since `@foo.0` has been reduced.
Lower `@bar.0a` and its callees to MIR, JIT-execute.

Execution trace:

- `@foo.0b'(21)`: a=21, x = 21+21 = 42, ctx01 = opaque_pack(int_type, int_type).
- `@bar.0a`: x = 42, foo.ctx01 = opaque_pack(int_type, int_type), result.type = int_type.
  Returns ctx0ab = opaque_pack(42, opaque_pack(int_type, int_type), int_type).

Materialize `ctx0ab` as constants.
Specialize `@bar.0b` by inlining → `@bar.0b'` takes no arguments:

```mlir
hir.func @bar.0b'() -> (result) {
  %opaque_type = hir.opaque_type
  %int_type = hir.int_type
  %x = hir.constant_int 42                // inlined
  %foo.ctx01 = hir.opaque_pack (          // inlined
    hir.int_type, hir.int_type)
  %result.type = hir.int_type             // inlined
  %4501 = hir.constant_int 4501
  %y = hir.call @foo.1(%4501, %foo.ctx01) : (%int_type, %opaque_type) -> (%int_type)
  %result = hir.binary %x, %y : %result.type
  hir.return %result : %result.type
}
```

Update `@bar.0`: list shrinks from `[@bar.0a, @bar.0b]` to just `[@bar.0b']`.

#### Step 2: Evaluate `@bar.0b'`

`@bar.0b'` takes no arguments → can evaluate.
Inside `@bar.0b'`, `hir.call @foo.1(%4501, %foo.ctx01)` calls the multiphase function `@foo.1` with both arguments known.
The evaluation pipeline reduces `@foo.1`:

1. `@foo.1a(foo.ctx01)` → ctx1ab = opaque_pack(int_type, int_type).
2. Specialize `@foo.1b` with ctx1ab → `@foo.1b'(%b)`.
3. `@foo.1b'(4501)` → y = 4501 + 4501 = 9002.

`@bar.0b'` computes result = 42 + 9002 = 9044.

Replace `@bar.0` with the constant result:

```mlir
hir.split_func @bar() -> (result: 0) { ... } [
  0: @bar.result  // constant: 9044
]
```

Done — bar is fully monomorphized.

### `hir.unified_call` Decomposition

When a caller has `hir.unified_call @foo(...)`, the SplitPhases pass decomposes it by looking up the callee's `hir.split_func` phase list:

1. Determine the callee's caller-visible phases from its split function's phase list (e.g., {-1, 0} for foo).
2. For each caller-visible phase, generate an `hir.call` to the corresponding function in the phase list, placed in the appropriate sub-function of the caller.
   Arguments are the callee's original args at that phase, plus the context from the previous phase.
3. Thread the opaque context between successive calls via `hir.opaque_pack`/`hir.opaque_unpack`.
4. Map the visible results back to the SSA values that the original `hir.unified_call` produced.

The caller references the functions listed in the callee's `hir.split_func` — for example, `@foo.0` and `@foo.1` — not any internal sub-functions like `@foo.0a` or `@foo.0b`.

### Interpretation Pipeline

Running this all the way through compilation in silc should produce a few iterations through the HIR-MIR-interpret-specialize pipeline:

1. Walk `hir.multiphase_func` ops.
2. For each, find the first sub-function in the list whose arguments are all constants (= all opaque context values unpacked within it are known) or that takes no arguments.
   If a sub-function contains `hir.call` ops to other multiphase functions, those calls are resolved by looking up the callee's `hir.split_func` phase list to find the corresponding function.
3. Lower the sub-function and its transitive callees to MIR.
   All called functions are compiled into one LLVM module for batch execution.
4. JIT-execute the LLVM module.
5. Materialize the returned opaque values as constants.
6. Specialize the next sub-function in the list by inlining the materialized constants.
   Remove the evaluated sub-function from the list.
   When only one function remains, replace the `hir.multiphase_func` with a direct call.
   6b. Pre-specialize: for any sub-function that now has some arguments resolved to constants (but still depends on other arguments), inline those constants into the function to reduce its argument count.
7. Check if any `hir.multiphase_func` now has its first sub-function fully evaluable; repeat from step 2.
8. Stop when no more sub-functions can be evaluated (remaining ones depend on caller-provided arguments).

This naturally supports iterative reduction: walk the multiphase function list front-to-back, evaluate sub-functions that have all-constant arguments, and stop when hitting one that depends on external arguments.
silc should check if there are any more compile-time-executable or specializable functions in the IR where we know the constant values of some parameters, usually because the previous iteration's interpretation has computed some constants.
If any such functions are left, it should run that specific HIR-MIR-interpret-specialize pass pipeline, and then rinse and repeat.

## Example 2

Consider a chain of functions where const arguments propagate transitively across multiple call levels:

```silicon
fn bark(a: int) {
  cark(0, a);
  cark(1, a);
}
fn cark(const b: int, c: int) {
  dark(b + 42, c);
  dark(b + 1337, c);
}
fn dark(const d: int, e: int) {
  print(d + e);
}
```

bark calls cark twice with distinct const args (0, 1).
cark computes derived const values (b+42, b+1337) and passes them to dark.
This creates a tree of four leaf specializations:

- bark → cark(b=0) → dark(d=42), dark(d=1337)
- bark → cark(b=1) → dark(d=43), dark(d=1338)

### Unified IR

```mlir
hir.unified_func @dark(%d: -1, %e: 0) -> () {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> ()
} {
  %0 = hir.int_type
  %1 = hir.binary %d, %e : %0
  hir.call @print(%1) : (%0) -> ()
  hir.unified_return
}

hir.unified_func @cark(%b: -1, %c: 0) -> () {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> ()
} {
  %0 = hir.int_type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b, %42 : %0
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b, %1337 : %0
  hir.unified_call @dark(%d1, %c) : (%0: -1, %0: 0) -> ()
  hir.unified_call @dark(%d2, %c) : (%0: -1, %0: 0) -> ()
  hir.unified_return
}

hir.unified_func @bark(%a: 0) -> () {
  %0 = hir.int_type
  hir.unified_signature (%0) -> ()
} {
  %0 = hir.int_type
  %1 = hir.constant_int 0
  %2 = hir.constant_int 1
  hir.unified_call @cark(%1, %a) : (%0: -1, %0: 0) -> ()
  hir.unified_call @cark(%2, %a) : (%0: -1, %0: 0) -> ()
  hir.unified_return
}
```

### Split IR

After the SplitPhases pass, each function gets an `hir.split_func` with a phase list, plus `hir.multiphase_func` and `hir.func` ops for each phase.

**dark** has caller-visible phases {-1, 0}.
Its body (`print(d + e)`) is entirely phase 0 since both the addition and the print call need `e`.
Phase -1 receives `d` and computes the types needed by phase 0.

```mlir
hir.split_func @dark(%d: -1, %e: 0) -> () {
  %int_type = hir.int_type
  hir.signature (%int_type, %int_type) -> ()
} [
  -1: @dark.0,  // implied (d) -> (ctx01)
  0: @dark.1    // implied (e, ctx01) -> ()
]
hir.multiphase_func @dark.0(last d) -> (ctx01) [
  @dark.0a,  // implied () -> (ctx0ab)
  @dark.0b   // implied (d, ctx0ab) -> (ctx01)
]
hir.func @dark.0a() -> (ctx0ab) {
  %opaque_type = hir.opaque_type
  %d.type = hir.int_type
  %ctx0ab = hir.opaque_pack (%d.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @dark.0b(%d, %ctx0ab) -> (ctx01) {
  %opaque_type = hir.opaque_type
  %d.type = hir.opaque_unpack %ctx0ab
  %d0 = hir.coerce_type %d : %d.type
  %e.type = hir.int_type
  %ctx01 = hir.opaque_pack (%d0, %d.type, %e.type)
  hir.return %ctx01 : %opaque_type
}
hir.multiphase_func @dark.1(last e, first ctx01) -> () [
  @dark.1a,  // implied (ctx01) -> (ctx1ab)
  @dark.1b   // implied (e, ctx1ab) -> ()
]
hir.func @dark.1a(%ctx01) -> (ctx1ab) {
  %opaque_type = hir.opaque_type
  %d, %d.type, %e.type = hir.opaque_unpack %ctx01
  %ctx1ab = hir.opaque_pack (%d, %d.type, %e.type)
  hir.return %ctx1ab : %opaque_type
}
hir.func @dark.1b(%e, %ctx1ab) -> () {
  %d, %d.type, %e.type = hir.opaque_unpack %ctx1ab
  %e0 = hir.coerce_type %e : %e.type
  %0 = hir.binary %d, %e0 : %d.type
  hir.call @print(%0) : (%e.type) -> ()
  hir.return
}
```

`%d` (the phase -1 argument) flows into `@dark.0b` via the multiphase function.
`@dark.0b` coerces `d` to its computed type, computes `e.type`, and packs everything that `@dark.1` will need into `ctx01`.
`@dark.1` is also a multiphase function: `@dark.1a` processes the incoming context (trivially repacking it in this example), and `@dark.1b` performs the runtime computation.

When a caller calls `@dark.0` with a specific `d` value, the evaluation pipeline reduces `@dark.0` (evaluating `@dark.0a`, then `@dark.0b` with the known `d`), producing a `ctx01` that can then be used to specialize `@dark.1`.

**cark** has caller-visible phases {-1, 0}.
Unlike dark, cark has real phase -1 computation: it evaluates b+42 and b+1337, then calls dark's phase -1 function for each result.

```mlir
hir.split_func @cark(%b: -1, %c: 0) -> () {
  %int_type = hir.int_type
  hir.signature (%int_type, %int_type) -> ()
} [
  -1: @cark.0,  // implied (b) -> (ctx01)
  0: @cark.1    // implied (c, ctx01) -> ()
]
hir.multiphase_func @cark.0(last b) -> (ctx01) [
  @cark.0a,  // implied () -> (ctx0ab)
  @cark.0b   // implied (b, ctx0ab) -> (ctx01)
]
hir.func @cark.0a() -> (ctx0ab) {
  %opaque_type = hir.opaque_type
  %b.type = hir.int_type
  %ctx0ab = hir.opaque_pack (%b.type)
  hir.return %ctx0ab : %opaque_type
}
hir.func @cark.0b(%b, %ctx0ab) -> (ctx01) {
  %opaque_type = hir.opaque_type
  %b.type = hir.opaque_unpack %ctx0ab
  %b0 = hir.coerce_type %b : %b.type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b0, %42 : %b.type
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b0, %1337 : %b.type
  %int_type = hir.int_type
  %dark.ctx01.1 = hir.call @dark.0(%d1) : (%int_type) -> (%opaque_type)
  %dark.ctx01.2 = hir.call @dark.0(%d2) : (%int_type) -> (%opaque_type)
  %c.type = hir.int_type
  %ctx01 = hir.opaque_pack (%c.type, %dark.ctx01.1, %dark.ctx01.2)
  hir.return %ctx01 : %opaque_type
}
hir.multiphase_func @cark.1(last c, first ctx01) -> () [
  @cark.1a,  // implied (ctx01) -> (ctx1ab)
  @cark.1b   // implied (c, ctx1ab) -> ()
]
hir.func @cark.1a(%ctx01) -> (ctx1ab) {
  %opaque_type = hir.opaque_type
  %c.type, %dark.ctx01.1, %dark.ctx01.2 = hir.opaque_unpack %ctx01
  %ctx1ab = hir.opaque_pack (%c.type, %dark.ctx01.1, %dark.ctx01.2)
  hir.return %ctx1ab : %opaque_type
}
hir.func @cark.1b(%c, %ctx1ab) -> () {
  %opaque_type = hir.opaque_type
  %c.type, %dark.ctx01.1, %dark.ctx01.2 = hir.opaque_unpack %ctx1ab
  %c0 = hir.coerce_type %c : %c.type
  hir.call @dark.1(%c0, %dark.ctx01.1) : (%c.type, %opaque_type) -> ()
  hir.call @dark.1(%c0, %dark.ctx01.2) : (%c.type, %opaque_type) -> ()
  hir.return
}
```

`@cark.0b` receives `b` and its type context, computes `b+42` and `b+1337`, and calls `@dark.0` for each.
The dark contexts returned by `@dark.0` are packed into cark's `ctx01` alongside `c.type`.
`@cark.1b` unpacks the contexts and calls `@dark.1` for each at runtime.

**bark** has a single caller-visible phase {0} (its only arg `a` is phase 0).
Internally, bark provides const args to cark and needs to compute `a`'s type.

```mlir
hir.split_func @bark(%a: 0) -> () {
  %int_type = hir.int_type
  hir.signature (%int_type) -> ()
} [
  0: @bark.0
]
hir.multiphase_func @bark.0(last a) -> () [
  @bark.0a,  // implied () -> (ctx0ab)
  @bark.0b   // implied (a, ctx0ab) -> ()
]
hir.func @bark.0a() -> (ctx0ab) {
  %opaque_type = hir.opaque_type
  %int_type = hir.int_type
  %0 = hir.constant_int 0
  %cark.ctx01.1 = hir.call @cark.0(%0) : (%int_type) -> (%opaque_type)
  %1 = hir.constant_int 1
  %cark.ctx01.2 = hir.call @cark.0(%1) : (%int_type) -> (%opaque_type)
  %a.type = hir.int_type
  %ctx0ab = hir.opaque_pack (%a.type, %cark.ctx01.1, %cark.ctx01.2)
  hir.return %ctx0ab : %opaque_type
}
hir.func @bark.0b(%a, %ctx0ab) -> () {
  %opaque_type = hir.opaque_type
  %a.type, %cark.ctx01.1, %cark.ctx01.2 = hir.opaque_unpack %ctx0ab
  %a0 = hir.coerce_type %a : %a.type
  hir.call @cark.1(%a0, %cark.ctx01.1) : (%a.type, %opaque_type) -> ()
  hir.call @cark.1(%a0, %cark.ctx01.2) : (%a.type, %opaque_type) -> ()
  hir.return
}
```

`@bark.0a` takes no arguments — the const values 0 and 1 are literals inside the function, `a`'s type is computed inline, and the calls to `@cark.0` produce opaque contexts.
This means `@bark.0a` can be fully evaluated at compile time without any external inputs, driving the entire const-evaluation cascade.

### Progressive Evaluation

#### Step 1: Evaluate leaf first sub-functions

Scan for `hir.multiphase_func` ops whose first sub-function takes no arguments.
Three candidates qualify: `@dark.0a`, `@cark.0a`, and `@bark.0a` — all take no arguments.
However, `@bark.0a` calls `@cark.0` (a multiphase function) and `@cark.0b` calls `@dark.0` (also multiphase).
These calls cannot be resolved until the callees are reduced.
Start with the leaf functions that contain no calls to unreduced multiphase functions: `@dark.0a` and `@cark.0a`.

**Lower to MIR and JIT-execute.**

```mlir
// dark.0a lowered to MIR
mir.func @dark.0a() -> (!mir.opaque) {
  %0 = mir.type_constant !mir.int
  %1 = mir.opaque_pack (%0)
  mir.return %1
}

// cark.0a lowered to MIR
mir.func @cark.0a() -> (!mir.opaque) {
  %0 = mir.type_constant !mir.int
  %1 = mir.opaque_pack (%0)
  mir.return %1
}
```

Both trivially return `opaque_pack(int_type)`.

**Materialize and reduce.**

- `@dark.0`: evaluate `@dark.0a` → specialize `@dark.0b` with `d.type = int_type` → `@dark.0b'`.
  List shrinks to `[@dark.0b']`. Since one function remains, replace `@dark.0` with `@dark.0b'`.
- `@cark.0`: evaluate `@cark.0a` → specialize `@cark.0b` with `b.type = int_type` → `@cark.0b'`.
  List shrinks to `[@cark.0b']`. Replace `@cark.0` with `@cark.0b'`.

Pre-specialized functions:

```mlir
hir.func @dark.0b'(%d) -> (ctx01) {
  %opaque_type = hir.opaque_type
  %d.type = hir.int_type                   // inlined
  %d0 = hir.coerce_type %d : %d.type
  %e.type = hir.int_type
  %ctx01 = hir.opaque_pack (%d0, %d.type, %e.type)
  hir.return %ctx01 : %opaque_type
}

hir.func @cark.0b'(%b) -> (ctx01) {
  %opaque_type = hir.opaque_type
  %b.type = hir.int_type                   // inlined
  %b0 = hir.coerce_type %b : %b.type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b0, %42 : %b.type
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b0, %1337 : %b.type
  %int_type = hir.int_type
  %dark.ctx01.1 = hir.call @dark.0b'(%d1) : (%int_type) -> (%opaque_type)
  %dark.ctx01.2 = hir.call @dark.0b'(%d2) : (%int_type) -> (%opaque_type)
  %c.type = hir.int_type
  %ctx01 = hir.opaque_pack (%c.type, %dark.ctx01.1, %dark.ctx01.2)
  hir.return %ctx01 : %opaque_type
}
```

`@dark.0b'` and `@cark.0b'` each take only a single argument — they cannot be evaluated further without a caller providing the value.

#### Step 2: Evaluate `@bark.0a`

Now `@bark.0a`'s calls to `@cark.0` resolve to `@cark.0b'` (a regular function), and `@cark.0b'` calls `@dark.0b'` (also a regular function).
All calls are transitively resolvable.

Lower `@bark.0a`, `@cark.0b'`, and `@dark.0b'` together to MIR and compile into a single LLVM module:

```mlir
mir.func @dark.0b'(%d: !mir.int) -> (!mir.opaque) {
  %d.type = mir.type_constant !mir.int
  %e.type = mir.type_constant !mir.int
  %0 = mir.opaque_pack (%d, %d.type, %e.type)
  mir.return %0
}

mir.func @cark.0b'(%b: !mir.int) -> (!mir.opaque) {
  %42 = mir.constant 42 : !mir.int
  %d1 = mir.binary add %b, %42 : !mir.int
  %1337 = mir.constant 1337 : !mir.int
  %d2 = mir.binary add %b, %1337 : !mir.int
  %dark.ctx01.1 = mir.call @dark.0b'(%d1) : (!mir.int) -> (!mir.opaque)
  %dark.ctx01.2 = mir.call @dark.0b'(%d2) : (!mir.int) -> (!mir.opaque)
  %c.type = mir.type_constant !mir.int
  %0 = mir.opaque_pack (%c.type, %dark.ctx01.1, %dark.ctx01.2)
  mir.return %0
}

mir.func @bark.0a() -> (!mir.opaque) {
  %0 = mir.constant 0 : !mir.int
  %cark.ctx01.1 = mir.call @cark.0b'(%0) : (!mir.int) -> (!mir.opaque)
  %1 = mir.constant 1 : !mir.int
  %cark.ctx01.2 = mir.call @cark.0b'(%1) : (!mir.int) -> (!mir.opaque)
  %a.type = mir.type_constant !mir.int
  %2 = mir.opaque_pack (%a.type, %cark.ctx01.1, %cark.ctx01.2)
  mir.return %2
}
```

All three functions go into a single LLVM module with `@bark.0a` as the entry point.
The call graph resolves transitively at runtime.

#### Step 3: JIT-execute and materialize results

Execute the LLVM module with `@bark.0a()` as the entry point.
The call graph is traversed transitively:

- `@cark.0b'(0)`: b=0, d1 = 0+42 = 42, d2 = 0+1337 = 1337.
  Calls `@dark.0b'(42)` → opaque_pack(42, int_type, int_type).
  Calls `@dark.0b'(1337)` → opaque_pack(1337, int_type, int_type).
  Returns opaque_pack(int_type, dark_ctx1, dark_ctx2).
- `@cark.0b'(1)`: b=1, d1 = 43, d2 = 1338.
  Calls `@dark.0b'(43)` → opaque_pack(43, int_type, int_type).
  Calls `@dark.0b'(1338)` → opaque_pack(1338, int_type, int_type).
  Returns opaque_pack(int_type, dark_ctx1, dark_ctx2).
- `@bark.0a()`: a.type = int_type, collects both cark context tuples.

Materialize the result.
Specialize `@bark.0b` by inlining the opaque context → `@bark.0b'(%a)`.
Update `@bark.0`: list shrinks from `[@bark.0a, @bark.0b]` to just `[@bark.0b']`.
Since one function remains, replace `@bark.0` with `@bark.0b'`.

All const-phase work is done.
The contexts nest: each cark context carries `c.type` plus two dark contexts; each dark context carries `d`, `d.type`, and `e.type`.

#### Step 4: Specialize runtime functions

`@bark.0b'` has all context inlined and takes only `%a`.
It contains `hir.call @cark.1(%a, %cark.ctx01.1)` and `hir.call @cark.1(%a, %cark.ctx01.2)`.
`@cark.1` is still a multiphase function, but now the `first` argument (`ctx01`) is a known constant for each call.

Reduce `@cark.1` for each constant context:

For `cark.ctx01.1` (cark called with b=0):

1. `@cark.1a(cark.ctx01.1)`: unpacks c.type=int_type, dark_ctx1=opaque_pack(42,...), dark_ctx2=opaque_pack(1337,...).
   Repacks into ctx1ab.
2. Specialize `@cark.1b` with ctx1ab → `@cark.1b.0(%c)`.
   `@cark.1b.0` has c.type, dark_ctx1, dark_ctx2 baked in.
3. `@cark.1` for this context reduces to `@cark.1b.0`.

Similarly for `cark.ctx01.2` (b=1) → `@cark.1b.1(%c)`.

`@bark.0b'` becomes:

```mlir
hir.func @bark.0b'(%a) -> () {
  %a.type = hir.int_type                   // inlined
  %a0 = hir.coerce_type %a : %a.type
  hir.call @cark.1b.0(%a0) : (%a.type) -> ()
  hir.call @cark.1b.1(%a0) : (%a.type) -> ()
  hir.return
}
```

Each `@cark.1b.N` contains `hir.call @dark.1` with materialized dark contexts.
Similarly reduce `@dark.1` for each context:

For dark_ctx carrying d=42:

1. `@dark.1a(dark_ctx)`: unpacks d=42, d.type=int_type, e.type=int_type. Repacks.
2. Specialize `@dark.1b` → `@dark.1b.0(%e)` with d=42 baked in.

This produces four dark specializations:

- `@dark.1b.0(%e)`: d=42
- `@dark.1b.1(%e)`: d=1337
- `@dark.1b.2(%e)`: d=43
- `@dark.1b.3(%e)`: d=1338

`@cark.1b.0(%c)` calls `@dark.1b.0` and `@dark.1b.1`.
`@cark.1b.1(%c)` calls `@dark.1b.2` and `@dark.1b.3`.

**Deduplication.**
The specialization key is (original function symbol, constant arg values).
Before cloning, the compiler checks if a specialization with the same key already exists and reuses it.
In this example all four dark specializations have distinct d values, so no deduplication occurs.
If two different call sites had both produced d=42, they would share the same specialization.

#### Step 5: Lower runtime functions to MIR

The specialized HIR functions from Step 4 are mechanically lowered to MIR.
Type-as-value operands become concrete MLIR types, and `hir.call` ops become `mir.call` ops:

```mlir
mir.func @dark.1b.0(%e: !mir.int) {
  %d = mir.constant 42 : !mir.int
  %0 = mir.binary add %d, %e : !mir.int
  mir.call @print(%0) : (!mir.int) -> ()
  mir.return
}
```

Specialization replaces opaque context values with constants, making type values known; MIR lowering then replaces `!hir.any` with concrete types (e.g., `%e.type = int_type` turns `%e: !hir.any` into `%e: !mir.int`).
The same lowering applies to all runtime functions — `@dark.1b.1` through `@dark.1b.3`, `@cark.1b.0`, `@cark.1b.1`, and `@bark.0b'`.

### Final Runtime IR

All phases have been resolved into a flat set of `mir.func` ops with concrete types and baked-in constants.
No `split_func`, `multiphase_func`, `opaque_pack`, or `opaque_unpack` ops remain:

```mlir
mir.func @bark(%a: !mir.int) {
  mir.call @cark.0(%a) : (!mir.int) -> ()
  mir.call @cark.1(%a) : (!mir.int) -> ()
  mir.return
}

mir.func @cark.0(%c: !mir.int) {
  mir.call @dark.0(%c) : (!mir.int) -> ()
  mir.call @dark.1(%c) : (!mir.int) -> ()
  mir.return
}

mir.func @dark.0(%e: !mir.int) {
  %0 = mir.constant 42 : !mir.int
  %1 = mir.binary add %0, %e : !mir.int
  mir.call @print(%1) : (!mir.int) -> ()
  mir.return
}
```

`@cark.1` follows the same pattern as `@cark.0`, calling `@dark.2` and `@dark.3`.
`@dark.1` through `@dark.3` are identical to `@dark.0` with their respective constant values (1337, 43, 1338).
bark takes a runtime argument `a` and calls through the monomorphized tree; this IR is lowered to CIRCT to arrive at the final hardware design.
