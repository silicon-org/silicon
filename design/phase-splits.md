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

Phase evaluation follows a loop through the HIR-MIR-LLVM-Execute-Specialize pipeline:

1.  **Lower HIR to MIR.**
    Type-as-value ops (`hir.int_type`, type block args) become concrete MLIR types on MIR ops.
    For example, `hir.binary %a, %b : %type` becomes `mir.binary %a, %b : !mir.int` once `%type` is known to be `int_type`.
    This is only possible when all type operands have been resolved to known constants, i.e. they are defined by ConstantLike ops such as `hir.int_type` or `hir.constant_int 42`.

2.  **Lower MIR to LLVM IR.**
    Standard MLIR-to-LLVM lowering.

3.  **JIT-execute.**
    Run the LLVM module and collect results.

4.  **Materialize constants.**
    Replace the evaluated `hir.phase_call` in the wiring region with constant ops holding the results.

5.  **Specialize functions.**
    Copy constant call arguments into function bodies, creating specialized/monomorphized versions of the function.
    This is the metaprogramming mechanism in Silicon.

6.  **Repeat.**
    Check if any more `hir.phase_call` ops now have all-constant operands and can be evaluated.
    If there are, go back to the first step and repeat.

## JIT Batching

Multiple independent phase functions can be compiled into a single LLVM module and JIT-executed together, minimizing the number of JIT round-trips.
When a phase function contains `hir.split_call` ops, they are first resolved to direct `hir.call` ops by consulting callee wiring regions — each `hir.split_call @callee[phase]` becomes an `hir.call` to the concrete split function for that phase.
The resolved functions are then lowered to MIR together and compiled into a single LLVM module, allowing the JIT to transitively evaluate an entire subgraph of phase functions in one execution.

## Specialization and Monomorphization

Substituting known constants for a phase function's block args is _specialization_.
When all type and constant block args of a phase function are known, the function is fully specialized and can be lowered to MIR.

_Monomorphization_ creates a dedicated clone of a split function for a specific set of constant arguments.
The specialization key is (original function symbol, constant arg values).
Before cloning, the compiler checks if a clone with the same key already exists and reuses it — this is how deduplication works.

## Library Compilation

When we compile an input into a library, we want to execute as many phases as possible.
This means that any internal phases, i.e. phases that don't depend on a function argument, should be evaluated until no such phases remain.
Once the library is compiled, we store it as an MLIR bytecode blob on disk.
A user may then use this library in their code compiled in a later invocation of the compiler.
At that point the new user code is still using `hir.unified_{func,call}` ops.
The library must retain enough information about how the individual phase splits combine to form a unified function to allow the later compiler invocation to resolve the user's unified calls to the precompiled phase splits in the library.

## Example 1

Consider the following input:

```
fn foo(const a: int, b: int) -> (const x: int, y: int) {
  (a + a, b + b)
}
fn bar() -> int {
  let (x, y) = foo(21, 4501);
  x + y
}
```

Running this through `silc --parse-only` should yield the following IR.
We want to support multiple and named results in functions, and allow for each result to have a distinct phase.
It's okay if the IR contains other type-of and inferrable ops, since parse-only does not run any canonicalizers.

```
hir.unified_func @foo [-1, 0] -> [-1, 0] attributes {argNames = ["a", "b"], resultNames = ["x", "y"]} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0, %0)
} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.int_type
  %1 = hir.binary %a, %a : %0
  %2 = hir.binary %b, %b : %0
  hir.unified_return %1, %2 : %0, %0
}

hir.unified_func @bar [] -> [0] attributes {argNames = [], resultNames = [""]} {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 21
  %2 = hir.constant_int 4501
  %x, %y = hir.unified_call @foo(%1, %2) : (%0, %0) -> (%0, %0) [-1, 0] -> [-1, 0]
  %3 = hir.binary %x, %y : %0
  hir.unified_return %3 : %0
}
```

### `hir.split_func` with a Wiring Region

Running the above through the SplitPhases pass should yield the IR below.
The current codebase names split functions `@foo.const0`, `@foo.const1`, `@foo.dyn1`, etc.
This proposal changes the naming to a flat `@foo.split0`, `@foo.split1`, `@foo.split2` scheme, numbered by phase order.

The key idea is that `hir.split_func` has a **wiring region** that uses standard MLIR SSA to express how a function's own phase functions chain together.
Each internal phase is an `hir.phase_call` op.
Original function arguments are block arguments of the region; original function results are operands of `hir.split_return`.

Cross-function calls use `hir.split_call`, which invokes a specific caller-visible phase of another `split_func`.
An opaque **phase context** threads between successive `split_call`s, hiding the callee's internal inter-phase values from the caller.
This ensures that the callee can pre-evaluate phases and pre-specialize its split functions without affecting any call sites.

The wiring region is purely plumbing: it only contains `hir.phase_call` ops to the enclosing function's own split functions, plus constants and `hir.split_return`.
All cross-function calls (`hir.split_call`) live inside the split functions themselves, keeping business logic out of the wiring region.

### New op: `hir.phase_call`

Calls one of the enclosing `split_func`'s own split functions.
A simple op with a callee symbol and positional operands/results.
No type annotations -- it's a wiring/descriptor op, not lowered to MIR.
Operands match the phase function's block args 1:1.
Results match the phase function's return values 1:1.

### New op: `hir.split_call`

Invokes a specific caller-visible phase of another `split_func`.
The caller-visible phases are the distinct phase indices from the callee's argument and result phases.
For `@foo [-1, 0] -> [-1, 0]`, the caller-visible phases are {-1, 0}.

```
%results..., %ctx = hir.split_call @foo[-1](%phase_minus1_args...)
%results...       = hir.split_call @foo[0](%phase_0_args..., %ctx)
```

- Arguments: the callee's original args for that phase, plus the context from the previous phase (if not the first phase).
- Results: the callee's original results for that phase, plus an opaque context for the next phase (if not the last phase).

The context bundles all internal inter-phase values that later phases need.
The caller never looks inside it -- only the callee's wiring region knows how to unpack it when the next phase is called.

`hir.split_call` ops appear inside split functions (the "business logic"), not in the wiring region.
This keeps the wiring region as pure plumbing between the enclosing function's own phases.

For three or more caller-visible phases, contexts chain linearly:

```
%r0, %ctx0 = hir.split_call @func[-2](%args0...)
%r1, %ctx1 = hir.split_call @func[-1](%args1..., %ctx0)
%r2        = hir.split_call @func[0](%args2..., %ctx1)
```

### New op: `hir.split_return`

Terminator for the wiring region.
Its operands are the original function's results.

### Change to phase functions

Phase functions gain **explicit block args for type context**, not just value arguments.
Previously `@foo.split1` had one block arg `(%a)` with types carried in the `hir.call` type annotation.
With this proposal, `@foo.split1` has block args for both the value AND the type context: `(%a, %a.type, %x.type)`.

This makes specialization uniform: inlining constants for some block args = specializing the function.

### How do we know what's what?

The wiring region's SSA structure tells us everything:

For **foo's** wiring region:

- `%a`, `%b` are block args of `hir.split_func` -> original function arguments
- `%a.type`, `%x.type` are results of `hir.phase_call @foo.split0()` -> inter-phase values (specialization constants once evaluated)
- `%x`, `%y` appear in `hir.split_return` -> original function results
- Everything else (`%b.type`, `%y.type`) -> inter-phase values for later phases

For **bar's** wiring region:

- No block args (bar takes no arguments)
- `%x`, `%foo.ctx`, `%bar.result.type` are results of `hir.phase_call @bar.split0()` -> inter-phase values
- `%foo.ctx` is an opaque phase context -- bar's wiring region threads it but never inspects it
- `%result` appears in `hir.split_return` -> original function result

No annotations needed -- SSA provenance is sufficient.

### Example: `@foo`

Foo's split functions and wiring region are purely internal.
The wiring region describes how foo's own phases chain together.

```
hir.split_func @foo [-1, 0] -> [-1, 0]
    attributes {argNames = ["a", "b"], resultNames = ["x", "y"]} {
^bb0(%a: !hir.any, %b: !hir.any):
  %a.type, %x.type = hir.phase_call @foo.split0()
  %x, %b.type, %y.type = hir.phase_call @foo.split1(%a, %a.type, %x.type)
  %y = hir.phase_call @foo.split2(%b, %b.type, %y.type)
  hir.split_return %x, %y
}

hir.func @foo.split0 {
  %0 = hir.int_type
  hir.return %0, %0
}
hir.func @foo.split1 {
^bb0(%a: !hir.any, %a.type: !hir.any, %x.type: !hir.any):
  %0 = hir.binary %a, %a : %a.type
  %1 = hir.int_type
  hir.return %0, %1, %1
}
hir.func @foo.split2 {
^bb0(%b: !hir.any, %b.type: !hir.any, %y.type: !hir.any):
  %0 = hir.binary %b, %b : %b.type
  hir.return %0
}
```

### Example: `@bar`

Bar's wiring region only threads values between bar's own two splits.
The `hir.split_call` ops to foo live inside bar's split functions -- the wiring region never mentions foo.

```
hir.split_func @bar [] -> [0]
    attributes {argNames = [], resultNames = [""]} {
  %x, %foo.ctx, %bar.result.type = hir.phase_call @bar.split0()
  %result = hir.phase_call @bar.split1(%bar.result.type, %x, %foo.ctx)
  hir.split_return %result
}

hir.func @bar.split0 {
  %21 = hir.constant_int 21
  %x, %foo.ctx = hir.split_call @foo[-1](%21)
  %bar.result.type = hir.int_type
  hir.return %x, %foo.ctx, %bar.result.type
}
hir.func @bar.split1 {
^bb0(%bar.result.type: !hir.any, %x: !hir.any, %foo.ctx: !hir.any):
  %4501 = hir.constant_int 4501
  %y = hir.split_call @foo[0](%4501, %foo.ctx)
  %result = hir.binary %x, %y : %bar.result.type
  hir.return %result
}
```

Note how bar's wiring region is compact: just two `hir.phase_call`s to bar's own splits, with `%foo.ctx` flowing opaquely between them.
If foo pre-specializes its internal phases, bar's IR is completely unaffected.

### Library Compilation

When compiling foo as a library, we evaluate phases that have no block-arg dependencies.

**Step 1: Evaluate foo.split0.**
`hir.phase_call @foo.split0()` has no operands and no block-arg dependencies.
Lower to MIR, interpret.
Result: (int_type, int_type).
Replace the `hir.phase_call` in foo's wiring region:

```
hir.split_func @foo [-1, 0] -> [-1, 0] ... {
^bb0(%a: !hir.any, %b: !hir.any):
  %a.type = hir.int_type         // phase_call @foo.split0 evaluated
  %x.type = hir.int_type         // phase_call @foo.split0 evaluated
  %x, %b.type, %y.type = hir.phase_call @foo.split1(%a, %a.type, %x.type)
  %y = hir.phase_call @foo.split2(%b, %b.type, %y.type)
  hir.split_return %x, %y
}
```

**Step 2: Pre-specialize foo.split1.**
Now foo.split1's operands are (%a, int_type, int_type) -- two of three are constants.
We pre-specialize by inlining the known constants, producing `@foo.split1'` that only takes `(%a)`:

```
hir.func @foo.split1' {
^bb0(%a: !hir.any):
  %a.type = hir.int_type   // inlined
  %0 = hir.binary %a, %a : %a.type
  %1 = hir.int_type
  hir.return %0, %1, %1
}
```

Update foo's wiring region to use the pre-specialized version:

```
hir.split_func @foo [-1, 0] -> [-1, 0] ... {
^bb0(%a: !hir.any, %b: !hir.any):
  %a.type = hir.int_type
  %x.type = hir.int_type
  %x, %b.type, %y.type = hir.phase_call @foo.split1'(%a)
  %y = hir.phase_call @foo.split2(%b, %b.type, %y.type)
  hir.split_return %x, %y
}
```

Phases -1 and 0 still depend on block args `%a` and `%b` -> can't evaluate further without a caller.
But the pre-specialization means that when a caller later provides `%a`, only one argument needs to be substituted instead of three.

### Progressive Evaluation

The following traces the full evaluation of both foo and bar, step by step.
We assume foo has been library-compiled as shown above.

#### Initial state

Foo after library compilation:

```
hir.split_func @foo [-1, 0] -> [-1, 0] ... {
^bb0(%a: !hir.any, %b: !hir.any):
  %a.type = hir.int_type
  %x.type = hir.int_type
  %x, %b.type, %y.type = hir.phase_call @foo.split1'(%a)
  %y = hir.phase_call @foo.split2(%b, %b.type, %y.type)
  hir.split_return %x, %y
}
```

Bar after SplitPhases:

```
hir.split_func @bar [] -> [0] ... {
  %x, %foo.ctx, %bar.result.type = hir.phase_call @bar.split0()
  %result = hir.phase_call @bar.split1(%bar.result.type, %x, %foo.ctx)
  hir.split_return %result
}
```

#### Step 1: Evaluate bar.split0

`hir.phase_call @bar.split0()` has no operands -> can evaluate.
Specialize bar.split0 (no block args), lower to MIR, interpret.

bar.split0 contains `hir.split_call @foo[-1](%21)`.
To resolve this, the pipeline reads foo's wiring region with `%a = 21`:

1. `%a.type = hir.int_type` -- already evaluated, use as-is.
2. `%x.type = hir.int_type` -- already evaluated, use as-is.
3. `hir.phase_call @foo.split1'(%a = 21)` -- all operands are now constants.
   Specialize `foo.split1'(21)`, lower to MIR, interpret.
   Result: `(42, int_type, int_type)`, i.e. `%x = 42`, `%b.type = int_type`, `%y.type = int_type`.

The split_call collects:

- Visible result: `%x = 42` (appears in foo's `hir.split_return` and has phase -1).
- Phase context: `{b.type = int_type, y.type = int_type}` (values from resolved phases that unresolved phases need).

bar.split0 also computes `%bar.result.type = hir.int_type`.
Overall bar.split0 returns: `(42, ctx{int_type, int_type}, int_type)`.

Replace `hir.phase_call @bar.split0()` in bar's wiring region:

```
hir.split_func @bar [] -> [0] ... {
  %x = hir.constant_int 42
  %foo.ctx = hir.phase_ctx @foo [hir.int_type, hir.int_type]
  %bar.result.type = hir.int_type
  %result = hir.phase_call @bar.split1(%bar.result.type, %x, %foo.ctx)
  hir.split_return %result
}
```

#### Step 2: Evaluate bar.split1

`hir.phase_call @bar.split1(int_type, 42, ctx{...})` -- all operands are constants -> can evaluate.
Specialize bar.split1 with the constants inlined, lower to MIR, interpret.

bar.split1 contains `hir.split_call @foo[0](%4501, %foo.ctx)`.
To resolve this, the pipeline reads foo's wiring region with `%b = 4501` and unpacks the context:

1. `%b.type = int_type` (from context).
2. `%y.type = int_type` (from context).
3. `hir.phase_call @foo.split2(%b = 4501, %b.type = int_type, %y.type = int_type)` -- all constants.
   Specialize `foo.split2(4501, int_type, int_type)`, lower to MIR, interpret.
   Result: `(9002)`, i.e. `%y = 9002`.

The split_call returns: `%y = 9002` (last phase, no context).

bar.split1 then computes `%result = hir.binary 42, 9002 : int_type` = `9044`.
Overall bar.split1 returns: `(9044)`.

Replace `hir.phase_call @bar.split1(...)` in bar's wiring region:

```
hir.split_func @bar [] -> [0] ... {
  %result = hir.constant_int 9044
  hir.split_return %result
}
```

Done -- bar is fully monomorphized.

### `hir.unified_call` Decomposition

When a caller has `hir.unified_call @foo(...)`, the SplitPhases pass decomposes it into `hir.split_call` ops inside the caller's split functions:

1. Determine the callee's caller-visible phases from its signature (e.g., {-1, 0} for foo).
2. For each caller-visible phase, generate an `hir.split_call @foo[phase](args...)` in the appropriate split function of the caller.
   Arguments are the callee's original args at that phase, plus the context from the previous phase.
3. Thread the opaque phase context between successive split_calls as a return value of one split function and a block arg of the next, wired through the caller's wiring region.
4. Map the visible results back to the SSA values that the original `hir.unified_call` produced.

The caller never references the callee's internal split functions (`foo.split0`, `foo.split1`, etc.) -- only the callee's `split_func` symbol and phase indices.

### Interpretation Pipeline

Running this all the way through compilation in silc should produce a few iterations through the HIR-MIR-interpret-specialize pipeline.
The interpret pass reads wiring regions instead of parsing naming conventions:

1. Walk `hir.split_func` ops.
2. For each, find the first `hir.phase_call` whose operands are all constants (= don't transitively depend on block args) and whose referenced split function can be lowered.
   A split function can be lowered when all its `hir.split_call` ops are transitively expandable: reading each callee's wiring region with the provided arguments and already-evaluated constants must make all relevant callee phase_calls all-constant.
   Pre-specialization (step 7b) simplifies this check by baking known constants into split functions ahead of time.
3. Specialize that phase function (inline constants for block args).
4. If the phase function contains `hir.split_call` ops, resolve them to direct `hir.call` ops by consulting each callee's wiring region.
   Each `hir.split_call @callee[phase]` becomes an `hir.call` to the concrete split function for that phase; the opaque context is replaced by the target function's explicit return values.
   This is a pure IR rewrite — no evaluation happens yet.
5. Lower the resolved function, along with any functions it calls via `hir.call`, to MIR.
   All called functions are compiled into one LLVM module for batch execution.
6. JIT-execute the LLVM module. The entry function's call graph resolves transitively at runtime.
7. Replace the `hir.phase_call` in the wiring region with the resulting constant ops.
   For `hir.split_call` results, the opaque context becomes an `hir.phase_ctx` constant in the wiring region.
   7b. Pre-specialize: for each `hir.phase_call` that now has some operands resolved to constants (but still depends on block args), inline those constants into the referenced split function and update the phase_call to remove the inlined operands.
   This is the same mechanism as in the Library Compilation section.
8. Check if any `hir.phase_call` now has all-constant operands or became newly evaluatable due to callee pre-specialization; repeat from step 2.
9. Stop when no more `hir.phase_call` ops can be evaluated (remaining ones depend on block args).

This naturally supports per-function monomorphization: walk the wiring region front-to-back, evaluate phases that have all-constant operands, and stop when hitting one that depends on block args.
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
hir.unified_func @dark [-1, 0] -> []
    attributes {argNames = ["d", "e"]} {
^bb0(%d: !hir.any, %e: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> ()
} {
^bb0(%d: !hir.any, %e: !hir.any):
  %0 = hir.int_type
  %1 = hir.binary %d, %e : %0
  hir.call @print(%1) : (%0) -> ()
  hir.unified_return
}

hir.unified_func @cark [-1, 0] -> []
    attributes {argNames = ["b", "c"]} {
^bb0(%b: !hir.any, %c: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> ()
} {
^bb0(%b: !hir.any, %c: !hir.any):
  %0 = hir.int_type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b, %42 : %0
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b, %1337 : %0
  hir.unified_call @dark(%d1, %c) : (%0, %0) -> () [-1, 0] -> []
  hir.unified_call @dark(%d2, %c) : (%0, %0) -> () [-1, 0] -> []
  hir.unified_return
}

hir.unified_func @bark [0] -> []
    attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0) -> ()
} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.constant_int 0
  %2 = hir.constant_int 1
  hir.unified_call @cark(%1, %a) : (%0, %0) -> () [-1, 0] -> []
  hir.unified_call @cark(%2, %a) : (%0, %0) -> () [-1, 0] -> []
  hir.unified_return
}
```

### Split IR

After the SplitPhases pass, each function gets a `split_func` with a wiring region and internal split functions.
The number of internal splits depends on the phase structure of the function and its callees.

**dark** has caller-visible phases {-1, 0}.
Its body (`print(d + e)`) is entirely phase 0 since both the addition and the print call need `e`.
There is no phase -1 computation — `d` is simply threaded through the wiring region as a pass-through block arg.
Following the mechanical pattern where each argument's type is computed one phase before the argument's own phase, dark gets three internal splits: d's type at phase -2, e's type at phase -1, and the runtime body at phase 0.

```mlir
hir.split_func @dark [-1, 0] -> [] {
^bb0(%d: !hir.any, %e: !hir.any):
  %d.type = hir.phase_call @dark.split0()
  %d.ctx, %d.type.ctx, %e.type = hir.phase_call @dark.split1(%d, %d.type)
  hir.phase_call @dark.split2(%e, %d.ctx, %d.type.ctx, %e.type)
  hir.split_return
}

// Computes the type of d (phase -2).
hir.func @dark.split0 {
  %d.type = hir.int_type
  hir.return %d.type
}

// Consumes d (phase -1 arg), computes the type of e, passes d through as context.
hir.func @dark.split1 {
^bb0(%d: !hir.any, %d.type: !hir.any):
  %e.type = hir.int_type
  hir.return %d, %d.type, %e.type
}

// Runtime: computes d+e and calls print (phase 0).
hir.func @dark.split2 {
^bb0(%e: !hir.any, %d: !hir.any, %d.type: !hir.any, %e.type: !hir.any):
  %0 = hir.binary %d, %e : %d.type
  hir.call @print(%0) : (%e.type) -> ()
  hir.return
}
```

`%d` (the phase -1 argument) flows into split1, which is the phase -1 split.
split1 consumes it, computes `%e.type`, and returns everything that split2 will need as opaque context.
This ensures that each split cleanly consumes its phase's arguments and produces context for subsequent phases — no values bypass a split in the wiring region.

When a caller does `hir.split_call @dark[-1](d_val)`, the pipeline reads dark's wiring region with `%d = d_val`, evaluates split0 and split1 (both now have all-constant operands), and bundles split1's return values into an opaque context.
The caller's later `hir.split_call @dark[0](e_val, ctx)` unpacks that context and feeds it into dark.split2.

**cark** has caller-visible phases {-1, 0}.
Unlike dark, cark has real phase -1 computation: it evaluates b+42 and b+1337, then calls dark's phase -1 for each result.
cark gets three internal splits: one to compute b's type, one for const evaluation (which also computes c's type), and one for the runtime body.

```mlir
hir.split_func @cark [-1, 0] -> [] {
^bb0(%b: !hir.any, %c: !hir.any):
  %b.type = hir.phase_call @cark.split0()
  %c.type, %dark.ctx1, %dark.ctx2 = hir.phase_call @cark.split1(%b, %b.type)
  hir.phase_call @cark.split2(%c, %c.type, %dark.ctx1, %dark.ctx2)
  hir.split_return
}

// Computes the type of b.
hir.func @cark.split0 {
  %0 = hir.int_type
  hir.return %0
}

// Evaluates b+42 and b+1337, calls dark's const phase, computes c's type.
hir.func @cark.split1 {
^bb0(%b: !hir.any, %b.type: !hir.any):
  %c.type = hir.int_type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b, %42 : %b.type
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b, %1337 : %b.type
  %dark.ctx1 = hir.split_call @dark[-1](%d1)
  %dark.ctx2 = hir.split_call @dark[-1](%d2)
  hir.return %c.type, %dark.ctx1, %dark.ctx2
}

// Runtime: calls dark's runtime phase with c and each dark context.
hir.func @cark.split2 {
^bb0(%c: !hir.any, %c.type: !hir.any, %dark.ctx1: !hir.any, %dark.ctx2: !hir.any):
  hir.split_call @dark[0](%c, %dark.ctx1)
  hir.split_call @dark[0](%c, %dark.ctx2)
  hir.return
}
```

Note that `%c.type` is computed in cark.split1, one phase before `%c` appears as an argument at phase 0.
This ensures that `%c`'s type can depend on values from the previous phase (such as `%b`), even though in this example the type is trivially `int_type`.
cark.split1 receives `%b` and `%b.type` for its const-phase computation; the `hir.split_call @dark[-1]` ops inside cark.split1 are cross-function calls that produce opaque dark contexts.
The `hir.split_call @dark[0]` ops in cark.split2 are specialized with these contexts to turn into simple functions accepting only arg `%c` at runtime.

**bark** has a single caller-visible phase {0} (its only arg `a` is phase 0).
Internally, bark provides const args to cark and needs to compute `a`'s type.
Since `a` is phase 0, its type must be computed in phase -1 — the same phase in which the cark const-evaluation happens.
bark gets two internal splits: one for computing `a`'s type and evaluating cark's const phase, and one for the runtime body.

```mlir
hir.split_func @bark [0] -> [] {
^bb0(%a: !hir.any):
  %a.type, %cark.ctx1, %cark.ctx2 = hir.phase_call @bark.split0()
  hir.phase_call @bark.split1(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}

// Computes a's type and evaluates cark's const phase for b=0 and b=1.
hir.func @bark.split0 {
  %a.type = hir.int_type
  %0 = hir.constant_int 0
  %cark.ctx1 = hir.split_call @cark[-1](%0)
  %1 = hir.constant_int 1
  %cark.ctx2 = hir.split_call @cark[-1](%1)
  hir.return %a.type, %cark.ctx1, %cark.ctx2
}

// Runtime: calls cark's runtime phase with a and each cark context.
hir.func @bark.split1 {
^bb0(%a: !hir.any, %a.type: !hir.any, %cark.ctx1: !hir.any, %cark.ctx2: !hir.any):
  hir.split_call @cark[0](%a, %cark.ctx1)
  hir.split_call @cark[0](%a, %cark.ctx2)
  hir.return
}
```

bark.split0 has no block args — the const values 0 and 1 are literals inside the function, `a`'s type is computed inline, and the split_call ops reference cark by symbol.
This means bark.split0 can be fully evaluated at compile time without any external inputs, driving the entire const-evaluation cascade.

### Progressive Evaluation

#### Step 1: Lower and JIT-execute dark.split0 and cark.split0

Scan for `hir.phase_call` ops whose operands are all constants.
Two candidates qualify: `@dark.split0()` and `@cark.split0()` — both take no operands and contain no `hir.split_call` ops, making them trivially lowerable.
(`@dark.split1(%d, %d.type)` depends on `%d` which is a block arg — it can only be evaluated when a caller provides `%d`.
`@bark.split0()` takes no operands but contains `hir.split_call @cark[-1]` ops — these are not yet expandable since cark.split0 hasn't been evaluated. We handle bark.split0 in step 2.)

**HIR → MIR lowering.**
The type-as-value ops become concrete type constants:

```mlir
// dark.split0 lowered to MIR
mir.func @dark.split0() -> (!mir.type) {
  %0 = mir.type_constant !mir.int
  mir.return %0
}

// cark.split0 lowered to MIR
mir.func @cark.split0() -> (!mir.type) {
  %0 = mir.type_constant !mir.int
  mir.return %0
}
```

**JIT execution.**
Both functions go into one LLVM module.
Lower MIR to LLVM IR, compile, JIT-execute.
Each trivially returns `int_type`.

**Materialize results.**
Replace the evaluated `hir.phase_call` ops with constant ops in the wiring regions:

```mlir
hir.split_func @dark [-1, 0] -> [] {
^bb0(%d: !hir.any, %e: !hir.any):
  %d.type = hir.int_type                    // split0 evaluated
  %d.ctx, %d.type.ctx, %e.type = hir.phase_call @dark.split1(%d, %d.type)
  hir.phase_call @dark.split2(%e, %d.ctx, %d.type.ctx, %e.type)
  hir.split_return
}

hir.split_func @cark [-1, 0] -> [] {
^bb0(%b: !hir.any, %c: !hir.any):
  %b.type = hir.int_type                    // split0 evaluated
  %c.type, %dark.ctx1, %dark.ctx2 = hir.phase_call @cark.split1(%b, %b.type)
  hir.phase_call @cark.split2(%c, %c.type, %dark.ctx1, %dark.ctx2)
  hir.split_return
}

hir.split_func @bark [0] -> [] {
^bb0(%a: !hir.any):
  %a.type, %cark.ctx1, %cark.ctx2 = hir.phase_call @bark.split0()
  hir.phase_call @bark.split1(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}
```

dark.split0 and cark.split0 are resolved.

**Pre-specialize.**
With `%d.type` and `%b.type` now known constants, pre-specialize split functions that have some constant operands.
`dark.split1(%d, %d.type)` has `%d.type = int_type` known — inline it to produce `dark.split1'(%d)`.
`cark.split1(%b, %b.type)` has `%b.type = int_type` known — inline it to produce `cark.split1'(%b)`.

Updated wiring regions:

```mlir
hir.split_func @dark [-1, 0] -> [] {
^bb0(%d: !hir.any, %e: !hir.any):
  %d.ctx, %d.type.ctx, %e.type = hir.phase_call @dark.split1'(%d)
  hir.phase_call @dark.split2(%e, %d.ctx, %d.type.ctx, %e.type)
  hir.split_return
}

hir.split_func @cark [-1, 0] -> [] {
^bb0(%b: !hir.any, %c: !hir.any):
  %c.type, %dark.ctx1, %dark.ctx2 = hir.phase_call @cark.split1'(%b)
  hir.phase_call @cark.split2(%c, %c.type, %dark.ctx1, %dark.ctx2)
  hir.split_return
}
```

dark.split1' and cark.split1' now each take only a single block-arg operand — they cannot be evaluated further without a caller providing the argument.
bark is unchanged — bark.split0 is still pending.

#### Step 2: Resolve phase -1 `hir.split_call` ops to direct `hir.call` ops

After Step 1, the pre-specialized functions `dark.split1'` and `cark.split1'` exist, and their earlier phases (dark.split0, cark.split0) have been evaluated.
The next step is a mechanical IR rewrite: replace every `hir.split_call @callee[phase]` in the phase -1 functions with a direct `hir.call` to the concrete split function that handles that phase.

To resolve a `hir.split_call`, consult the callee's wiring region.
Phases that have already been evaluated (replaced by constants in the wiring region) are skipped; the first remaining `hir.phase_call` for the requested phase identifies the target split function.
The opaque context result of the `hir.split_call` is replaced by the explicit return values of the target function.

**Resolving `cark.split1'`.**
cark.split1' contains two `hir.split_call @dark[-1]` ops.
Dark's wiring shows that phase -1 maps to `dark.split1'` (dark.split0 was evaluated in Step 1).
Replace each `hir.split_call` with a direct `hir.call`:

```mlir
hir.func @cark.split1' {
^bb0(%b: !hir.any):
  %b.type = hir.int_type                                       // inlined in step 1
  %c.type = hir.int_type
  %42 = hir.constant_int 42
  %d1 = hir.binary %b, %42 : %b.type
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b, %1337 : %b.type
  %d1.pass, %d1t, %e1t = hir.call @dark.split1'(%d1)           // was split_call @dark[-1]
  %d2.pass, %d2t, %e2t = hir.call @dark.split1'(%d2)           // was split_call @dark[-1]
  hir.return %c.type, %d1.pass, %d1t, %e1t, %d2.pass, %d2t, %e2t
}
```

The opaque `%dark.ctx` has been replaced by the three explicit return values of `dark.split1'` — the values that dark's later phase (dark.split2) will need.
The return list of cark.split1' grows from 3 to 7 values correspondingly.

**Resolving `bark.split0`.**
bark.split0 contains two `hir.split_call @cark[-1]` ops.
Cark's wiring shows that phase -1 maps to `cark.split1'`.
Replace:

```mlir
hir.func @bark.split0 {
  %a.type = hir.int_type
  %0 = hir.constant_int 0
  // cark.split1' returns 7 values: c.type, d1.pass, d1.type, e1.type, d2.pass, d2.type, e2.type
  %c0t, %c0.d1, %c0.d1t, %c0.e1t, %c0.d2, %c0.d2t, %c0.e2t
      = hir.call @cark.split1'(%0)                              // was split_call @cark[-1]
  %1 = hir.constant_int 1
  %c1t, %c1.d1, %c1.d1t, %c1.e1t, %c1.d2, %c1.d2t, %c1.e2t
      = hir.call @cark.split1'(%1)                              // was split_call @cark[-1]
  hir.return %a.type, %c0t, %c0.d1, %c0.d1t, %c0.e1t, %c0.d2, %c0.d2t, %c0.e2t,
                      %c1t, %c1.d1, %c1.d1t, %c1.e1t, %c1.d2, %c1.d2t, %c1.e2t
}
```

After this step, none of the phase -1 functions contain `hir.split_call` ops — only regular `hir.call` ops to concrete split functions.
This makes them directly lowerable to MIR without any wiring region inspection.

#### Step 3: Lower phase -1 functions to MIR

With all cross-function calls resolved to direct `hir.call` ops, the three phase -1 functions can be mechanically lowered to MIR.
The type-as-value rewriting follows the same pattern as in Step 1: `hir.binary %b, %42 : %b.type` where `%b.type = hir.int_type` becomes `mir.binary add %b, %42 : !mir.int`.

```mlir
// dark.split1': leaf function, no calls.
mir.func @dark.split1'(%d: !mir.int) -> (!mir.int, !mir.type, !mir.type) {
  %d.type = mir.type_constant !mir.int
  %e.type = mir.type_constant !mir.int
  mir.return %d, %d.type, %e.type
}

// cark.split1': calls dark.split1', all types known.
mir.func @cark.split1'(%b: !mir.int)
    -> (!mir.type, !mir.int, !mir.type, !mir.type, !mir.int, !mir.type, !mir.type) {
  %c.type = mir.type_constant !mir.int
  %42 = mir.constant 42 : !mir.int
  %d1 = mir.binary add %b, %42 : !mir.int
  %1337 = mir.constant 1337 : !mir.int
  %d2 = mir.binary add %b, %1337 : !mir.int
  %d1.pass, %d1t, %e1t = mir.call @dark.split1'(%d1)
      : (!mir.int) -> (!mir.int, !mir.type, !mir.type)
  %d2.pass, %d2t, %e2t = mir.call @dark.split1'(%d2)
      : (!mir.int) -> (!mir.int, !mir.type, !mir.type)
  mir.return %c.type, %d1.pass, %d1t, %e1t, %d2.pass, %d2t, %e2t
}

// bark.split0: calls cark.split1', all types known.
mir.func @bark.split0() -> (!mir.type, ...) {
  %a.type = mir.type_constant !mir.int
  %0 = mir.constant 0 : !mir.int
  %c0t, %c0.d1, %c0.d1t, %c0.e1t, %c0.d2, %c0.d2t, %c0.e2t
      = mir.call @cark.split1'(%0)
      : (!mir.int) -> (!mir.type, !mir.int, !mir.type, !mir.type, !mir.int, !mir.type, !mir.type)
  %1 = mir.constant 1 : !mir.int
  %c1t, %c1.d1, %c1.d1t, %c1.e1t, %c1.d2, %c1.d2t, %c1.e2t
      = mir.call @cark.split1'(%1)
      : (!mir.int) -> (!mir.type, !mir.int, !mir.type, !mir.type, !mir.int, !mir.type, !mir.type)
  mir.return %a.type, %c0t, %c0.d1, %c0.d1t, %c0.e1t, %c0.d2, %c0.d2t, %c0.e2t,
                      %c1t, %c1.d1, %c1.d1t, %c1.e1t, %c1.d2, %c1.d2t, %c1.e2t
}
```

All three functions go into a single LLVM module for batch JIT execution, with `bark.split0` as the entry point.
The call graph resolves transitively at runtime: bark.split0 calls cark.split1', which calls dark.split1' — all within one JIT execution.
Each function remains a separate compilation unit; the JIT linker resolves the inter-function calls.

#### Step 4: JIT-execute and materialize results

Execute the LLVM module with `bark.split0()` as the entry point.
The call graph is traversed transitively in a single JIT execution:

- `cark.split1'(0)`: b=0, computes d1 = 0+42 = 42, d2 = 0+1337 = 1337.
  Calls `dark.split1'(42)` → (42, int_type, int_type) and `dark.split1'(1337)` → (1337, int_type, int_type).
  Returns: (int_type, 42, int_type, int_type, 1337, int_type, int_type).
- `cark.split1'(1)`: b=1, computes d1 = 1+42 = 43, d2 = 1+1337 = 1338.
  Calls `dark.split1'(43)` → (43, int_type, int_type) and `dark.split1'(1338)` → (1338, int_type, int_type).
  Returns: (int_type, 43, int_type, int_type, 1338, int_type, int_type).
- `bark.split0()`: collects a.type = int_type plus both cark result tuples.

The materialization infrastructure maps the flat return values back into nested `hir.phase_ctx` ops.
The structure is derived from the call graph: bark calls cark calls dark, so the contexts nest correspondingly.
Replace `hir.phase_call @bark.split0()` in bark's wiring region:

```mlir
hir.split_func @bark [0] -> [] {
^bb0(%a: !hir.any):
  %a.type = hir.int_type
  %cark.ctx1 = hir.phase_ctx @cark [hir.int_type,
      hir.phase_ctx @dark [42, hir.int_type, hir.int_type],
      hir.phase_ctx @dark [1337, hir.int_type, hir.int_type]]
  %cark.ctx2 = hir.phase_ctx @cark [hir.int_type,
      hir.phase_ctx @dark [43, hir.int_type, hir.int_type],
      hir.phase_ctx @dark [1338, hir.int_type, hir.int_type]]
  hir.phase_call @bark.split1(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}
```

All const-phase work is done.
The contexts nest: each cark context carries c.type plus two dark contexts; each dark context carries d plus d.type and e.type.

**Pre-specialize `bark.split1`.**
`hir.phase_call @bark.split1(%a, int_type, ctx, ctx)` has three of four operands constant.
Inline the known constants to produce `bark.split1'` that only takes `%a`:

```mlir
hir.func @bark.split1' {
^bb0(%a: !hir.any):
  %a.type = hir.int_type                    // inlined
  %cark.ctx1 = hir.phase_ctx @cark [...]    // inlined
  %cark.ctx2 = hir.phase_ctx @cark [...]    // inlined
  hir.split_call @cark[0](%a, %cark.ctx1)
  hir.split_call @cark[0](%a, %cark.ctx2)
  hir.return
}
```

bark.split1' depends on `%a` (block arg from bark's caller) — this is the boundary between compile time and runtime.

#### Step 5: Resolve phase 0 `hir.split_call` ops and specialize runtime functions

With all contexts materialized, resolve the remaining `hir.split_call` ops in the runtime functions.
Unlike the phase -1 resolution in Step 2 where contexts had not yet been computed, the phase 0 split_calls carry materialized `hir.phase_ctx` ops that provide all the constant arguments.
For each split_call, unpack the context, read the callee's wiring region to identify the target runtime function, then clone and specialize it by substituting the constants.
The `hir.split_call` ops within the clone are also resolved to direct `hir.call` ops to the specialized functions.

**Specializing bark.split1' → `@bark`.**
bark.split1' contains `hir.split_call @cark[0](%a, %cark.ctx1)`.
Unpack cark.ctx1: c.type = int_type, dark_ctx1 = {42, int, int}, dark_ctx2 = {1337, int, int}.
Read cark's wiring region with `%c = %a` and the unpacked values.
The remaining phase_call is `@cark.split2(%c, c.type, dark_ctx1, dark_ctx2)` — all type/ctx args are constants.
Clone cark.split2 with c.type, dark_ctx1, dark_ctx2 baked in → `@cark.0`.
Similarly, `@cark[0](%a, %cark.ctx2)` produces `@cark.1` (with dark contexts for d=43 and d=1338).

`@bark` after specialization:

```mlir
hir.func @bark {
^bb0(%a: !hir.any):
  %a.type = hir.int_type
  hir.call @cark.0(%a)
  hir.call @cark.1(%a)
  hir.return
}
```

**Specializing cark.split2 → `@cark.0` and `@cark.1`.**
cark.split2 contains `hir.split_call @dark[0](%c, %dark.ctx)` ops.
For each, unpack the dark context, read dark's wiring to find dark.split2 as the target, and clone dark.split2 with the context values baked in.
The `hir.split_call` ops in the clone are resolved to direct `hir.call` ops to the specialized dark functions.

- @cark.0 (b=0): dark_ctx1 = {d=42, int, int} → `@dark.0`; dark_ctx2 = {d=1337, int, int} → `@dark.1`.
- @cark.1 (b=1): dark_ctx1 = {d=43, int, int} → `@dark.2`; dark_ctx2 = {d=1338, int, int} → `@dark.3`.

`@cark.0` after specialization:

```mlir
hir.func @cark.0 {
^bb0(%c: !hir.any):
  %c.type = hir.int_type                    // inlined from context
  hir.call @dark.0(%c)
  hir.call @dark.1(%c)
  hir.return
}
```

`@cark.1` follows the same pattern, calling `@dark.2` and `@dark.3`.

**Specializing dark.split2 → `@dark.0` through `@dark.3`.**
Each specialization clones dark.split2 and inlines the constant arguments (d, d.type, e.type) from the context:

```mlir
// @dark.0: d=42, d.type=int, e.type=int
hir.func @dark.0 {
^bb0(%e: !hir.any):
  %d = hir.constant_int 42                  // inlined from context
  %d.type = hir.int_type                    // inlined from context
  %e.type = hir.int_type                    // inlined from context
  %0 = hir.binary %d, %e : %d.type
  hir.call @print(%0) : (%e.type) -> ()
  hir.return
}
```

`@dark.1` (d=1337), `@dark.2` (d=43), and `@dark.3` (d=1338) are identical in structure with their respective constant values.

**Deduplication.**
The specialization key is (original function symbol, constant arg values).
Before cloning, the compiler checks if a specialization with the same key already exists and reuses it.
In this example all four dark specializations have distinct d values, so no deduplication occurs.
If two different call sites had both produced d=42, they would share `@dark.0`.

#### Step 6: Lower runtime functions to MIR

The specialized HIR functions from Step 5 are mechanically lowered to MIR.
Type-as-value operands become concrete MLIR types, and `hir.call` ops become `mir.call` ops:

```mlir
mir.func @dark.0(%e: !mir.int) {
  %d = mir.constant 42 : !mir.int
  %0 = mir.binary add %d, %e : !mir.int
  mir.call @print(%0) : (!mir.int) -> ()
  mir.return
}
```

Specialization replaces block args with constants, making type values known; MIR lowering then replaces `!hir.any` with concrete types (e.g., `%e.type = int_type` turns `%e: !hir.any` into `%e: !mir.int`).
The same lowering applies to all runtime functions — `@dark.1` through `@dark.3`, `@cark.0`, `@cark.1`, and `@bark`.

### Final Runtime IR

All phases have been resolved into a flat set of `mir.func` ops with concrete types and baked-in constants.
No `split_func`, `split_call`, `phase_call`, or `phase_ctx` ops remain:

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
