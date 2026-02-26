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

## `hir.split_func` with a Wiring Region

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

## Library Compilation

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

## Progressive Evaluation

The following traces the full evaluation of both foo and bar, step by step.
We assume foo has been library-compiled as shown above.

### Initial state

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

### Step 1: Evaluate bar.split0

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

### Step 2: Evaluate bar.split1

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

## `hir.unified_call` Decomposition

When a caller has `hir.unified_call @foo(...)`, the SplitPhases pass decomposes it into `hir.split_call` ops inside the caller's split functions:

1. Determine the callee's caller-visible phases from its signature (e.g., {-1, 0} for foo).
2. For each caller-visible phase, generate an `hir.split_call @foo[phase](args...)` in the appropriate split function of the caller.
   Arguments are the callee's original args at that phase, plus the context from the previous phase.
3. Thread the opaque phase context between successive split_calls as a return value of one split function and a block arg of the next, wired through the caller's wiring region.
4. Map the visible results back to the SSA values that the original `hir.unified_call` produced.

The caller never references the callee's internal split functions (`foo.split0`, `foo.split1`, etc.) -- only the callee's `split_func` symbol and phase indices.

## Interpretation Pipeline

Running this all the way through compilation in silc should produce a few iterations through the HIR-MIR-interpret-specialize pipeline.
The interpret pass reads wiring regions instead of parsing naming conventions:

1. Walk `hir.split_func` ops.
2. For each, find the first `hir.phase_call` whose operands are all constants (= don't transitively depend on block args).
3. Specialize that phase function (inline constants for block args).
4. If the phase function contains `hir.split_call` ops, resolve them by reading the callee's wiring region, evaluating the necessary internal phases, and collecting the visible results and phase context.
5. Lower the specialized+resolved function to MIR.
6. Interpret it.
7. Replace the `hir.phase_call` in the wiring region with the resulting constant ops.
   For `hir.split_call` results, the opaque context becomes an `hir.phase_ctx` constant in the wiring region.
8. Check if the next `hir.phase_call` now has all-constant operands; repeat.
9. Stop when a `hir.phase_call` depends on block args (needs caller-provided values).

This naturally supports per-function monomorphization: walk the wiring region front-to-back, evaluate phases that have all-constant operands, and stop when hitting one that depends on block args.
silc should check if there are any more compile-time-executable or specializable functions in the IR where we know the constant values of some parameters, usually because the previous iteration's interpretation has computed some constants.
If any such functions are left, it should run that specific HIR-MIR-interpret-specialize pass pipeline, and then rinse and repeat.

## Multi-Function Example: Transitive Monomorphization

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

Three compilation phases:

1. **Types**: compute types for all params (trivially `int_type`).
2. **Const evaluation**: evaluate cark for b=0 and b=1, transitively evaluating dark for d=42, d=1337, d=43, d=1338. Each produces a monomorphized function.
3. **Runtime**: the final program where bark(a) calls cark.0(a) and cark.1(a), which call dark.{0,1,2,3}(c), which call print({42,1337,43,1338}+e). This is not executed at compile time -- it is the shape of the final executable.

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
Split functions use `@name.split0`, `@name.split1`, etc. naming; the monomorphized functions produced later use `@name.0`, `@name.1`, etc.

**dark** has caller-visible phases {-1, 0}.
Its body (`print(d + e)`) is entirely phase 0 since both the addition and the print call need `e`.
There is no phase -1 computation -- `d` is simply threaded through the wiring region as a pass-through block arg.
dark gets two internal splits: type computation and the runtime body.

```mlir
hir.split_func @dark [-1, 0] -> [] {
^bb0(%d: !hir.any, %e: !hir.any):
  %d.type, %e.type = hir.phase_call @dark.split0()
  hir.phase_call @dark.split1(%e, %d, %d.type, %e.type)
  hir.split_return
}

hir.func @dark.split0 {
  %0 = hir.int_type
  hir.return %0, %0
}

hir.func @dark.split1 {
^bb0(%e: !hir.any, %d: !hir.any, %d.type: !hir.any, %e.type: !hir.any):
  %0 = hir.binary %d, %e : %d.type
  hir.call @print(%0) : (%e.type) -> ()
  hir.return
}
```

Note that `%d` flows from the wiring region's block args directly to split1.
When a caller does `hir.split_call @dark[-1](d_val)`, the pipeline reads dark's wiring region with `%d = d_val`, evaluates what it can (split0 and any phases whose operands become constant), and bundles the remaining inter-phase values -- including d -- into an opaque context.
The caller's later `hir.split_call @dark[0](e_val, ctx)` unpacks that context and specializes dark.split1 with the baked-in d.

**cark** has caller-visible phases {-1, 0}.
Unlike dark, cark has real phase -1 computation: it evaluates b+42 and b+1337, then calls dark's phase -1 for each result.
cark gets three internal splits: type computation, const evaluation, and the runtime body.

```mlir
hir.split_func @cark [-1, 0] -> [] {
^bb0(%b: !hir.any, %c: !hir.any):
  %b.type, %c.type = hir.phase_call @cark.split0()
  %dark.ctx1, %dark.ctx2 = hir.phase_call @cark.split1(%b, %b.type)
  hir.phase_call @cark.split2(%c, %c.type, %dark.ctx1, %dark.ctx2)
  hir.split_return
}

hir.func @cark.split0 {
  %0 = hir.int_type
  hir.return %0, %0
}

hir.func @cark.split1 {
^bb0(%b: !hir.any, %b.type: !hir.any):
  %42 = hir.constant_int 42
  %d1 = hir.binary %b, %42 : %b.type
  %1337 = hir.constant_int 1337
  %d2 = hir.binary %b, %1337 : %b.type
  %dark.ctx1 = hir.split_call @dark[-1](%d1)
  %dark.ctx2 = hir.split_call @dark[-1](%d2)
  hir.return %dark.ctx1, %dark.ctx2
}

hir.func @cark.split2 {
^bb0(%c: !hir.any, %c.type: !hir.any, %dark.ctx1: !hir.any, %dark.ctx2: !hir.any):
  hir.split_call @dark[0](%c, %dark.ctx1)
  hir.split_call @dark[0](%c, %dark.ctx2)
  hir.return
}
```

Note that `%c.type` flows from split0 directly to split2 in the wiring region, bypassing split1.
cark.split1 only receives what it needs for its computation: `%b` and `%b.type`.
The `hir.split_call @dark[-1]` ops inside cark.split1 are cross-function calls that produce opaque dark contexts; the `hir.split_call @dark[0]` ops in cark.split2 consume those contexts at runtime.

**bark** has a single caller-visible phase {0} (its only arg `a` is phase 0).
But internally, bark provides const args to cark, which induces an internal const-evaluation phase in addition to the type-computation phase.
bark gets three internal splits: types, const eval, and runtime.

```mlir
hir.split_func @bark [0] -> [] {
^bb0(%a: !hir.any):
  %a.type = hir.phase_call @bark.split0()
  %cark.ctx1, %cark.ctx2 = hir.phase_call @bark.split1()
  hir.phase_call @bark.split2(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}

hir.func @bark.split0 {
  %0 = hir.int_type
  hir.return %0
}

hir.func @bark.split1 {
  %0 = hir.constant_int 0
  %cark.ctx1 = hir.split_call @cark[-1](%0)
  %1 = hir.constant_int 1
  %cark.ctx2 = hir.split_call @cark[-1](%1)
  hir.return %cark.ctx1, %cark.ctx2
}

hir.func @bark.split2 {
^bb0(%a: !hir.any, %a.type: !hir.any, %cark.ctx1: !hir.any, %cark.ctx2: !hir.any):
  hir.split_call @cark[0](%a, %cark.ctx1)
  hir.split_call @cark[0](%a, %cark.ctx2)
  hir.return
}
```

bark.split1 has no block args -- the const values 0 and 1 are literals inside the function, and the split_call ops reference cark by symbol.
This means bark.split1 can be fully evaluated at compile time without any external inputs, driving the entire const-evaluation cascade.

### Progressive Evaluation

#### Step 1: Type Computation

Find all `hir.phase_call` ops with no operands across all split_funcs: all split0 functions qualify.
Lower dark.split0, cark.split0, bark.split0 to MIR, interpret.
All trivially return `int_type`.

Replace the phase_calls in wiring regions:

```mlir
hir.split_func @dark [-1, 0] -> [] {
^bb0(%d: !hir.any, %e: !hir.any):
  %d.type = hir.int_type                    // split0 evaluated
  %e.type = hir.int_type                    // split0 evaluated
  hir.phase_call @dark.split1(%e, %d, %d.type, %e.type)
  hir.split_return
}

hir.split_func @cark [-1, 0] -> [] {
^bb0(%b: !hir.any, %c: !hir.any):
  %b.type = hir.int_type                    // split0 evaluated
  %c.type = hir.int_type                    // split0 evaluated
  %dark.ctx1, %dark.ctx2 = hir.phase_call @cark.split1(%b, %b.type)
  hir.phase_call @cark.split2(%c, %c.type, %dark.ctx1, %dark.ctx2)
  hir.split_return
}

hir.split_func @bark [0] -> [] {
^bb0(%a: !hir.any):
  %a.type = hir.int_type                    // split0 evaluated
  %cark.ctx1, %cark.ctx2 = hir.phase_call @bark.split1()
  hir.phase_call @bark.split2(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}
```

#### Step 2: Const Evaluation

Scan for newly evaluatable phase_calls.
In bark's wiring region, `hir.phase_call @bark.split1()` has no operands -> evaluate.

bark.split1 contains two `hir.split_call @cark[-1]` ops.
To resolve these, the pipeline reads cark's wiring region and cascades through the callee's internal phases.

**Resolving `hir.split_call @cark[-1](0)`:**

1. Substitute `%b = 0` in cark's wiring region.
2. `%b.type = int_type` and `%c.type = int_type` are already constants (from step 1).
3. `hir.phase_call @cark.split1(0, int_type)` -- all constants, evaluate.
4. cark.split1(0, int_type) computes:
   - d1 = 0 + 42 = 42
   - d2 = 0 + 1337 = 1337
   - Resolve `hir.split_call @dark[-1](42)`: read dark's wiring with `%d = 42`.
     Types already evaluated. dark.split1 needs `%e` (block arg) -> can't evaluate further.
     Bundle into context: `{d=42, d.type=int_type, e.type=int_type}`.
   - Resolve `hir.split_call @dark[-1](1337)`: context `{d=1337, d.type=int_type, e.type=int_type}`.
   - Returns: `(dark_ctx{42,...}, dark_ctx{1337,...})`.
5. cark.split2 needs `%c` (block arg) -> can't evaluate further.
6. Bundle remaining values into cark context: `{c.type=int_type, dark_ctx{42,...}, dark_ctx{1337,...}}`.

**Resolving `hir.split_call @cark[-1](1)`:**

Same cascade with `%b = 1`:
- d1 = 1 + 42 = 43, d2 = 1 + 1337 = 1338
- dark contexts: `{d=43,...}`, `{d=1338,...}`
- cark context: `{int_type, dark_ctx{43,...}, dark_ctx{1338,...}}`

bark.split1 returns both cark contexts.

Replace in bark's wiring region:

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
  hir.phase_call @bark.split2(%a, %a.type, %cark.ctx1, %cark.ctx2)
  hir.split_return
}
```

All const-phase work is done.
The contexts nest: each cark context carries c.type plus two dark contexts; each dark context carries d plus the types.
bark.split2 depends on `%a` (block arg) -> cannot evaluate at compile time.
This is the boundary between compile time and runtime.

#### Step 3: Monomorphization

Pre-specialize bark.split2 by inlining the known constants (%a.type, %cark.ctx1, %cark.ctx2).
Then resolve each `hir.split_call` by unpacking contexts and creating dedicated monomorphized functions.

**Specialize bark** with constants inlined.
Resolve `hir.split_call @cark[0](%a, cark_ctx1)` by unpacking cark_ctx1 and specializing cark.split2:

- c.type = int_type (from cark context)
- dark_ctx1 = {42, int_type, int_type} (from cark context)
- dark_ctx2 = {1337, int_type, int_type} (from cark context)
- Substitute into cark.split2 -> produces `@cark.0`.

Similarly, `hir.split_call @cark[0](%a, cark_ctx2)` -> `@cark.1`.

**Inside @cark.0**, resolve `hir.split_call @dark[0]` by unpacking dark contexts and specializing dark.split1:

- dark_ctx{42, int_type, int_type}: d=42 baked in -> `@dark.0`
- dark_ctx{1337, int_type, int_type}: d=1337 baked in -> `@dark.1`

**Inside @cark.1**:

- dark_ctx{43,...}: d=43 -> `@dark.2`
- dark_ctx{1338,...}: d=1338 -> `@dark.3`

### Final Runtime IR

```mlir
hir.func @bark {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  hir.call @cark.0(%a) : (%0) -> ()
  hir.call @cark.1(%a) : (%0) -> ()
  hir.return
}

hir.func @cark.0 {
^bb0(%c: !hir.any):
  %0 = hir.int_type
  hir.call @dark.0(%c) : (%0) -> ()
  hir.call @dark.1(%c) : (%0) -> ()
  hir.return
}

hir.func @cark.1 {
^bb0(%c: !hir.any):
  %0 = hir.int_type
  hir.call @dark.2(%c) : (%0) -> ()
  hir.call @dark.3(%c) : (%0) -> ()
  hir.return
}

hir.func @dark.0 {
^bb0(%e: !hir.any):
  %0 = hir.constant_int 42
  %1 = hir.int_type
  %2 = hir.binary %0, %e : %1
  hir.call @print(%2) : (%1) -> ()
  hir.return
}

hir.func @dark.1 {
^bb0(%e: !hir.any):
  %0 = hir.constant_int 1337
  %1 = hir.int_type
  %2 = hir.binary %0, %e : %1
  hir.call @print(%2) : (%1) -> ()
  hir.return
}

hir.func @dark.2 {
^bb0(%e: !hir.any):
  %0 = hir.constant_int 43
  %1 = hir.int_type
  %2 = hir.binary %0, %e : %1
  hir.call @print(%2) : (%1) -> ()
  hir.return
}

hir.func @dark.3 {
^bb0(%e: !hir.any):
  %0 = hir.constant_int 1338
  %1 = hir.int_type
  %2 = hir.binary %0, %e : %1
  hir.call @print(%2) : (%1) -> ()
  hir.return
}
```

No `split_func`, `split_call`, or `phase_call` ops remain.
All phased execution has been resolved into a flat set of monomorphized functions with baked-in constants.
bark takes a runtime argument `a` and calls through the monomorphized tree.

If two different call sites had produced the same const argument (e.g., two callers both yielding d=42), the monomorphization pass could deduplicate them into a single function, similar to how linkers merge identical template instantiations.

### JIT Batching

Each compile-time step maps to one JIT execution, keeping the number of JIT round-trips minimal.

**JIT module 1 (types):** Contains dark.split0, cark.split0, bark.split0.
All are independent functions returning type constants.
One JIT call per function (or a single wrapper calling all three).

**JIT module 2 (const eval):** Contains bark.split1 as the entry point, plus cark.split1 as a callee.
During HIR->MIR lowering, the `hir.split_call @cark[-1]` ops inside bark.split1 are expanded into direct calls to cark.split1 (with types materialized as MIR types).
The `hir.split_call @dark[-1]` ops inside cark.split1 become context struct packing, since dark has no phase -1 computation.
The entire evaluation graph compiles to one LLVM module with bark.split1 as the entry point.
Executing it transitively evaluates cark.split1 for both b=0 and b=1, producing all four dark contexts.

**No JIT module 3:** Monomorphization is pure IR manipulation -- unpacking contexts, inlining constants into split functions, and renaming the results.
The final runtime IR is then lowered through MIR to LLVM (or CIRCT) for the target platform, not for JIT.
