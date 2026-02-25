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

Running this through the SplitPhases pass should yield the following IR.
Note: the current codebase names split functions `@foo.const0`, `@foo.const1`, `@foo.dyn1`, etc.
This proposal changes the naming to a flat `@foo.split0`, `@foo.split1`, `@foo.split2` scheme, numbered by phase order.

```
hir.split_func @foo(a: -1, b: 0) -> (x: -1, y: 0) {
  // Split 0 will return types for arg "a" and result "x".
  [-2] @foo.split0 {nextArgTypes: ["a"], nextResultTypes: ["x"]}
  // Split 1 will accept arg "a" and provide result "x" based on types of split 0, and it will return types for arg "b" and result "y" of split 2.
  [-1] @foo.split1 {args: ["a"], results: ["x"], nextArgTypes: ["b"], nextResultTypes: ["y"]}
  // Split 2 will accept arg "b" and provide result "y" based on types of split 1.
  [0] @foo.split2 {args: ["b"], results: ["y"]}
}
hir.func @foo.split0 attributes {argNames = [], resultNames = ["", ""]} {
  %0 = hir.int_type
  %1 = hir.type_type
  hir.return %0, %0 : %1, %1
}
hir.func @foo.split1 attributes {argNames = ["a"], resultNames = ["x", "", ""]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.binary %a, %a : %0
  %2 = hir.type_type
  hir.return %1, %0, %0 : %0, %2, %2
}
hir.func @foo.split2 attributes {argNames = ["b"], resultNames = ["y"]} {
^bb0(%b: !hir.any):
  %0 = hir.int_type
  %1 = hir.binary %b, %b : %0
  hir.return %1 : %0
}

hir.split_func @bar() -> ("": 0) {
  // Split 0 will return "foo.a" and "foo.x" type constants to be monomorphized into split 1.
  [-2] @bar.split0 {consts: 2}
  // Split 1 will accept 2 constants from split 0, and return type of unnamed result of bar, constant "x", and type constants "foo.b" and "foo.y" to be monomorphized into split 2.
  [-1] @bar.split1 {nextResultTypes: [""], consts: 3}
  // Split 2 will accept 3 constants from split 1 and provide unnamed result based on type returned by split 1.
  [0] @bar.split2 {results: [""]}
}
hir.func @bar.split0 attributes {argNames = [], resultNames = ["", ""]} {
  %0 = hir.type_type
  %1, %2 = hir.call @foo.split0() : () -> (%0, %0)
  hir.return %1, %2 : %0, %0
}
hir.func @bar.split1 attributes {argNames = ["", ""], resultNames = ["", "", "", ""]} {
^bb0(%a.type: !hir.any, %x.type: !hir.any):
  %0 = hir.int_type
  %1 = hir.constant_int 21
  %2 = hir.type_type
  %x, %3, %4 = hir.call @foo.split1(%1) : (%a.type) -> (%x.type, %2, %2)
  hir.return %0, %x, %3, %4 : %2, %0, %2, %2
}
hir.func @bar.split2 attributes {argNames = ["", "", ""], resultNames = [""]} {
^bb0(%x: !hir.any, %b.type: !hir.any, %y.type: !hir.any):
  %1 = hir.int_type
  %2 = hir.constant_int 4501
  %y = hir.call @foo.split2(%2) : (%b.type) -> (%y.type)
  %3 = hir.binary %x, %y : %1
  hir.return %3 : %1
}
```

Running this all the way through compilation in silc should produce a few iterations through the HIR-MIR-interpret-specialize pipeline.
We are currently only doing a crude single run through that pipeline in silc.
However, silc should check if there are any more compile-time-executable or specializable functions in the IR where we know the constant values of some parameters, usually because the previous iteration's interpretation has computed some constants.
If any such functions are left, it should run that specific HIR-MIR-interpret-specialize pass pipeline, and then rinse and repeat.
