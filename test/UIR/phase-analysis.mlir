// RUN: silicon-opt --test-phase-analysis2 %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//
// Trivial baselines: no phase modifiers, everything at phase 0.

// Constant return: constant_int is ConstantLike and floats.
// CHECK-LABEL: uir.func @Constant
uir.func @Constant() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  // CHECK: hir.constant_int 42 {{.*}} {{.*}}pa.phase = "float"
  %c42 = hir.constant_int 42 : %t
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %c42 -> (%t)
}

// Identity: single arg passed through.
// CHECK-LABEL: uir.func @Identity
uir.func @Identity(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %x -> (%t)
}

// Pure ops with body-phase args: no slack, diamond DAG stays at 0.
// CHECK-LABEL: uir.func @PureOps
uir.func @PureOps(%a: 0, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %sum = hir.add %a, %b : %t
  // CHECK: hir.sub {{.*}} {{.*}}pa.phase = "0"
  %diff = hir.sub %a, %b : %t
  // CHECK: hir.mul {{.*}} {{.*}}pa.phase = "0"
  %prod = hir.mul %sum, %diff : %t
  uir.return %prod -> (%t)
}

//===----------------------------------------------------------------------===//
// Single-phase function: all side-effecting ops inherit phase 0, constants
// float.

// CHECK-LABEL: uir.func @SinglePhase
uir.func @SinglePhase() -> () {
  uir.signature () -> () {phase = 0 : si16}
} {
  // CHECK: func.call @dummyA() {{.*}}pa.phase = "0"
  func.call @dummyA() : () -> ()
  // CHECK: uir.return -> () {{.*}}pa.phase = "0"
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Const argument: pure ops using a const arg float to the arg's phase.
// Type operands use hir.int_type (constant, floats to any phase).

// Simplest const arg: identity, arg at -1, return at 0.
// CHECK-LABEL: uir.func @ConstIdentity
uir.func @ConstIdentity(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %x -> (%t)
}

// Dyn arg basics: identity pass-through, dyn+literal, dyn+dyn, and a diamond
// DAG of pure ops — all at phase 1. Return op stays at phase 0.
// CHECK-LABEL: uir.func @DynArg
uir.func @DynArg(%a: 1, %b: 1) -> (r0: 1, r1: 1, r2: 1, r3: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  %t4 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0, %t2, %t3, %t4)
} {
  %t = hir.int_type
  // Dyn + literal: add floats to phase 1.
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.add %a, {{.*}}pa.phase = "1"
  // CHECK-SAME: pa.results = ["1"]
  %sum = hir.add %a, %c1 : %t
  // Dyn + dyn: add at phase 1.
  // CHECK: hir.add %a, %b {{.*}}pa.phase = "1"
  // CHECK-SAME: pa.results = ["1"]
  %ab = hir.add %a, %b : %t
  // Diamond DAG: (a+1)*(a-1), all pure ops at phase 1.
  // CHECK: hir.sub {{.*}}pa.phase = "1"
  %xm1 = hir.sub %a, %c1 : %t
  // CHECK: hir.mul {{.*}}pa.phase = "1"
  %prod = hir.mul %sum, %xm1 : %t
  // Return at phase 0, all value operands at phase 1.
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["1", "1", "1", "1", "float", "float", "float", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %a, %sum, %ab, %prod -> (%t, %t, %t, %t)
}

// CHECK-LABEL: uir.func @ConstArg
uir.func @ConstArg(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  uir.signature (%0, %1) -> (%2)
} {
  // hir.type_of shifts -1: type_of(%a at -1) = -2, type_of(%b at 0) = -1.
  // CHECK: hir.type_of %a {{.*}}pa.phase = "-2"
  %ta = hir.type_of %a
  // CHECK: hir.type_of %b {{.*}}pa.phase = "-1"
  %tb = hir.type_of %b
  // hir.unify is effectively pure: earliest = max(-2, -1) = -1.
  // CHECK: hir.unify {{.*}} {{.*}}pa.phase = "-1"
  %t = hir.unify %ta, %tb
  // The add's type operand needs phase -1 (add at 0, type at 0-1 = -1).
  // %t (unify) is at -1 — satisfies!
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %0 = hir.add %a, %b : %t
  // Return type operand needs phase 0 - 1 = -1. %t is at -1 — satisfies!
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Pure ops with const arg float to earliest (arg phase). Tests const+const
// (same arg), const+literal, chain, and wider DAG — all float to -1.

// CHECK-LABEL: uir.func @PureOpFloats
uir.func @PureOpFloats(%a: -1, %b: -1, %c: -1) -> (r0: 0, r1: 0, r2: 0, r3: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  %3 = hir.int_type
  %4 = hir.int_type
  uir.signature (%0, %1, %2) -> (%0, %3, %4, %4)
} {
  // type_of(%a at -1) = -2. The add's value operands are both at -1, so
  // earliest = max(-1, -1, -2) = -1. Type operand needs -1-1 = -2; %ta is
  // at -2 — satisfies!
  // CHECK: hir.type_of %a {{.*}}pa.phase = "-2"
  %ta = hir.type_of %a
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %0 = hir.add %a, %a : %ta
  // Const arg + literal: literal floats, so earliest = max(-1, float) = -1.
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %1 = hir.add %a, %c1 : %t
  // Chain of pure ops: n*2+1. Intermediate mul at -1 feeds add, which also
  // floats to -1 since all operands are const or floating.
  %c2 = hir.constant_int 2 : %t
  // CHECK: hir.mul {{.*}} {{.*}}pa.phase = "-1"
  %prod = hir.mul %a, %c2 : %t
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %2 = hir.add %prod, %c1 : %t
  // Wider DAG with multiple const args: (a+b)*c - (a-c). All float to -1.
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %ab = hir.add %a, %b : %t
  // CHECK: hir.mul {{.*}} {{.*}}pa.phase = "-1"
  %abc = hir.mul %ab, %c : %t
  // CHECK: hir.sub %a, %c {{.*}} {{.*}}pa.phase = "-1"
  %ac = hir.sub %a, %c : %t
  // CHECK: hir.sub {{.*}} {{.*}}pa.phase = "-1"
  %3 = hir.sub %abc, %ac : %t
  // Return type operand needs 0-1 = -1. %ta is at -2, satisfies.
  uir.return %0, %1, %2, %3 -> (%ta, %t, %t, %t)
}

//===----------------------------------------------------------------------===//
// Const return types: result offsets are -1, but the return op itself is at
// the body block phase (0). The returned values are at -1 (floating/const).

// CHECK-LABEL: uir.func @ConstReturn
uir.func @ConstReturn(%x: -1, %y: -1) -> (r0: -1, r1: -1, r2: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0, %t2, %t3)
} {
  %t = hir.int_type
  // Identity pass-through at -1.
  // Const+literal: add floats to -1.
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %sum = hir.add %x, %c1 : %t
  // Two const args combined.
  // CHECK: hir.add %x, %y {{.*}} {{.*}}pa.phase = "-1"
  %xy = hir.add %x, %y : %t
  // Return op at body block phase 0.
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %x, %sum, %xy -> (%t, %t, %t)
}

//===----------------------------------------------------------------------===//
// Literal-only const return: float satisfies -1 demand.

// CHECK-LABEL: uir.func @ConstLiteralReturn
uir.func @ConstLiteralReturn() -> (result: -1) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  // CHECK: hir.constant_int 42 {{.*}} {{.*}}pa.phase = "float"
  %c42 = hir.constant_int 42 : %t
  // Return op at body block phase 0, value is float.
  // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.return %c42 -> (%t)
}

//===----------------------------------------------------------------------===//
// Stacked const: phase -2 identity and pure op floating to -2.

// CHECK-LABEL: uir.func @DeepConst
uir.func @DeepConst(%x: -2) -> (r0: -2, r1: -2) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0) -> (%t1, %t2)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}}pa.phase = "-2"
  %sum = hir.add %x, %c1 : %t
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["-2", "-2", "float", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %x, %sum -> (%t, %t)
}

//===----------------------------------------------------------------------===//
// Stacked dyn: phase +2 identity and pure op floating to +2.

// CHECK-LABEL: uir.func @DeepDyn
uir.func @DeepDyn(%x: 2) -> (r0: 2, r1: 2) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0) -> (%t1, %t2)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}}pa.phase = "2"
  %sum = hir.add %x, %c1 : %t
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["2", "2", "float", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %x, %sum -> (%t, %t)
}

//===----------------------------------------------------------------------===//
// Mixed depth: const const (-2) + const (-1). Pure op at max(-2, -1) = -1.

// CHECK-LABEL: uir.func @MixedDepth
uir.func @MixedDepth(%a: -2, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  // CHECK: hir.add {{.*}}pa.operands = ["-2", "-1", "float"]
  // CHECK-SAME: pa.phase = "-1"
  %sum = hir.add %a, %b : %t
  uir.return %sum -> (%t)
}

//===----------------------------------------------------------------------===//
// Pure ops pinned by let bindings: diamond DAG, all at phase 0.

// CHECK-LABEL: uir.func @PinnedPureOps
uir.func @PinnedPureOps(%a: 0, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %sum = hir.add %a, %b : %t
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %psum = uir.pin %sum, 0 : !hir.any
  // CHECK: hir.sub {{.*}} {{.*}}pa.phase = "0"
  %diff = hir.sub %a, %b : %t
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %pdiff = uir.pin %diff, 0 : !hir.any
  // CHECK: hir.mul {{.*}} {{.*}}pa.phase = "0"
  %prod = hir.mul %psum, %pdiff : %t
  uir.return %prod -> (%t)
}

//===----------------------------------------------------------------------===//
// Pin absorbs slack: ops could float to -1 (const+literal), but pins force 0.
// Also tests two independent const streams pinned, then combined.

// CHECK-LABEL: uir.func @PinAbsorbsSlack
uir.func @PinAbsorbsSlack(%a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %c10 = hir.constant_int 10 : %t
  %sum = hir.add %a, %c10 : %t
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %x = uir.pin %sum, 0 : !hir.any
  // Second independent const stream, also pinned.
  %c3 = hir.constant_int 3 : %t
  %prod = hir.mul %b, %c3 : %t
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %y = uir.pin %prod, 0 : !hir.any
  // Combine pinned results.
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %r = hir.add %x, %y : %t
  uir.return %r -> (%t)
}

//===----------------------------------------------------------------------===//
// Mixed const and body args: inner add on two const args floats to -1,
// outer add mixing that result with a body arg stays at 0.

// CHECK-LABEL: uir.func @MixedConstBody
uir.func @MixedConstBody(%a: -1, %b: -1, %c: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // CHECK: hir.add %a, %b {{.*}} {{.*}}pa.phase = "-1"
  %ab = hir.add %a, %b : %t
  // CHECK: hir.add {{.*}}, %c {{.*}} {{.*}}pa.phase = "0"
  %abc = hir.add %ab, %c : %t
  uir.return %abc -> (%t)
}

//===----------------------------------------------------------------------===//
// Mixed body and dyn args: add pulls to phase 1 (max(0, 1) = 1).

// CHECK-LABEL: uir.func @MixedDynBody
uir.func @MixedDynBody(%x: 0, %y: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  // CHECK: hir.add {{.*}}pa.operands = ["0", "1", "float"]
  // CHECK-SAME: pa.phase = "1"
  %sum = hir.add %x, %y : %t
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["1", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %sum -> (%t)
}

//===----------------------------------------------------------------------===//
// Pinned uir.expr with phase shift (const block).

// CHECK-LABEL: uir.func @PinnedExpr
uir.func @PinnedExpr(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %a : %t
    uir.yield %sum : %t
  // CHECK: } {{.*}}pa.phase = "-1"
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Pinned uir.expr with dyn shift: block at phase 1, body-phase value flows in.

// CHECK-LABEL: uir.func @DynPinnedExpr
uir.func @DynPinnedExpr(%x: 0) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    // Yield at phase 1, carrying %x at phase 0 across the boundary.
    // CHECK: uir.yield {{.*}}pa.operands = ["0", "float"]
    // CHECK-SAME: pa.phase = "1"
    uir.yield %x : %t
  // CHECK: } {{.*}}pa.phase = "1"
  // CHECK-SAME: pa.results = ["0"]
  }
  // Return at phase 0, result value at phase 0 (propagated through yield).
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["0", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Nested const blocks: const { const { ... } } reaches phase -2.

// CHECK-LABEL: uir.func @NestedConstBlocks
uir.func @NestedConstBlocks(%a: -2) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // Outer const block at phase -1.
  %0 = uir.expr pin -1 : %t {
    // Inner const block at phase -2.
    // CHECK: hir.add {{.*}}pa.phase = "-2"
    // CHECK: } {{.*}}pa.phase = "-2"
    %inner = uir.expr pin -1 : %t {
      %c1 = hir.constant_int 1 : %t
      %sum = hir.add %a, %c1 : %t
      uir.yield %sum : %t
    }
    uir.yield %inner : %t
  // CHECK: } {{.*}}pa.phase = "-1"
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Const block as statement: side-effecting call at phase -1.

func.func private @sideEffectC()

// CHECK-LABEL: uir.func @ConstBlockStatement
uir.func @ConstBlockStatement(%a: -1) -> () {
  %t0 = hir.int_type
  uir.signature (%t0) -> ()
} {
  // CHECK: func.call @sideEffectC() {{.*}}pa.phase = "-1"
  // CHECK: } {{.*}}pa.phase = "-1"
  uir.expr pin -1 {
    func.call @sideEffectC() : () -> ()
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Nested dyn blocks: dyn { dyn { ... } } — literal floats through both yields.

// CHECK-LABEL: uir.func @NestedDynBlocks
uir.func @NestedDynBlocks() -> (result: 2) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    // CHECK: } {{.*}}pa.phase = "2"
    // CHECK-SAME: pa.results = ["float"]
    %inner = uir.expr pin 1 : %t {
      %c99 = hir.constant_int 99 : %t
      uir.yield %c99 : %t
    }
    uir.yield %inner : %t
  // CHECK: } {{.*}}pa.phase = "1"
  // CHECK-SAME: pa.results = ["float"]
  }
  // Literal floats through both yields — result and return operand are float.
  // CHECK: uir.return {{.*}} -> ({{.*}})
  // CHECK-SAME: pa.operands = ["float", "float"]
  // CHECK-SAME: pa.phase = "0"
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// dyn { const { ... } }: outer at +1, inner const at 0 (relative -1 from +1).
// Literal floats through both yields.

// CHECK-LABEL: uir.func @DynConstNesting
uir.func @DynConstNesting() -> (result: 1) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    // CHECK: } {{.*}}pa.phase = "0"
    // CHECK-SAME: pa.results = ["float"]
    %inner = uir.expr pin -1 : %t {
      %c42 = hir.constant_int 42 : %t
      uir.yield %c42 : %t
    }
    // Yield carries float value at phase 1.
    // CHECK: uir.yield {{.*}}pa.operands = ["float", "float"]
    // CHECK-SAME: pa.phase = "1"
    uir.yield %inner : %t
  // CHECK: } {{.*}}pa.phase = "1"
  // CHECK-SAME: pa.results = ["float"]
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Floating uir.expr: phase comes from consumer.

// CHECK-LABEL: uir.func @FloatingExpr
uir.func @FloatingExpr(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // The floating expr's phase is determined by its consumer (uir.return, which
  // needs the value at phase 0).
  %t = hir.int_type
  %0 = uir.expr : %t {
    // The add is pure, operands at -1, so earliest = -1.
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %a : %t
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %sum : %t
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Void uir.expr pin as discarded statement (dead pure op inside).

// CHECK-LABEL: uir.func @DeadExprStatement
uir.func @DeadExprStatement(%a: 0, %b: 0) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> ()
} {
  %t = hir.int_type
  // CHECK: uir.expr pin {
  uir.expr pin {
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
    %sum = hir.add %a, %b : %t
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Signature region: ConstantLike type ops float, satisfy any constraint.

// CHECK-LABEL: uir.func @SignaturePhases
uir.func @SignaturePhases(%a: -1, %b: 0) -> (result: 0) {
  // CHECK: hir.int_type {{.*}}pa.phase = "float"
  %0 = hir.int_type
  // CHECK: hir.int_type {{.*}}pa.phase = "float"
  %1 = hir.int_type
  // CHECK: hir.int_type {{.*}}pa.phase = "float"
  %2 = hir.int_type
  // CHECK: uir.signature ({{.*}}) -> ({{.*}}) {{.*}}pa.phase = "0"
  uir.signature (%0, %1) -> (%2)
} {
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Pinned op reached via DFS: pinning is authoritative, does not tighten.

// CHECK-LABEL: uir.func @PinnedOpViaDFS
uir.func @PinnedOpViaDFS(%a: 0) -> (result: 0) {
  %ty = hir.int_type
  uir.signature (%ty) -> (%ty)
} {
  %t = hir.int_type
  %sum = hir.add %a, %a : %t
  // Pin at phase 0 (blockPhase + 0). This is the authoritative phase.
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %pinned = uir.pin %sum, 0 : !hir.any
  // The return uses %pinned — resolveValue sees it's already resolved at 0.
  uir.return %pinned -> (%t)
}

//===----------------------------------------------------------------------===//
// uir.if: structured control flow.

// CHECK-LABEL: uir.func @IfOp
uir.func @IfOp(%cond: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.if %cond : %t {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond : %t
  } else {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond : %t
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Short-circuit &&: if (x >= lo) { x <= hi } else { false }.

// CHECK-LABEL: uir.func @ShortCircuitAnd
uir.func @ShortCircuitAnd(%x: 0, %lo: 0, %hi: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %bt = hir.bool_type
  // CHECK: hir.geq {{.*}} {{.*}}pa.phase = "0"
  %cond1 = hir.geq %x, %lo : %bt
  // CHECK: uir.if {{.*}} {
  %r = uir.if %cond1 : %bt {
    // CHECK: hir.leq {{.*}} {{.*}}pa.phase = "0"
    %cond2 = hir.leq %x, %hi : %bt
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond2 : %bt
  } else {
    %cfalse = hir.constant_bool <false>
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cfalse : %bt
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %r -> (%bt)
}

//===----------------------------------------------------------------------===//
// Const arg as if condition: condition available at -1, if stays at 0.

// CHECK-LABEL: uir.func @ConstCondIf
uir.func @ConstCondIf(%flag: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // CHECK: hir.constant_int 1 {{.*}} {{.*}}pa.phase = "float"
  %c1 = hir.constant_int 1 : %t
  // CHECK: hir.constant_int 0 {{.*}} {{.*}}pa.phase = "float"
  %c0 = hir.constant_int 0 : %t
  // CHECK: uir.if {{.*}} {
  %r = uir.if %flag : %t {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %c1 : %t
  } else {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %c0 : %t
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %r -> (%t)
}

//===----------------------------------------------------------------------===//
// uir.loop with break.

// CHECK-LABEL: uir.func @LoopOp
uir.func @LoopOp(%cond: 0, %val: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    uir.if %cond {
      // CHECK: uir.break {{.*}} {{.*}}pa.phase = "0"
      uir.break %val : %t
    } else {
      uir.unreachable
    }
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Side-effecting ops nested in structured CF: inherit parent phase.

// CHECK-LABEL: uir.func @SideEffectsInCF
uir.func @SideEffectsInCF(%cond: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  uir.if %cond {
    // CHECK: func.call @dummyA() {{.*}}pa.phase = "0"
    func.call @dummyA() : () -> ()
    uir.yield
  } else {
    // CHECK: func.call @dummyB() {{.*}}pa.phase = "0"
    func.call @dummyB() : () -> ()
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Early return inside uir.if.

// CHECK-LABEL: uir.func @EarlyReturn
uir.func @EarlyReturn(%cond: 0, %val: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  uir.if %cond {
    // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
    uir.return %val -> (%t)
  } else {
    uir.yield
  }
  uir.return %val -> (%t)
}

//===----------------------------------------------------------------------===//
// uir.unreachable after exhaustive early returns.

// CHECK-LABEL: uir.func @AllBranchesReturn
uir.func @AllBranchesReturn(%cond: 0, %val: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  uir.if %cond {
    uir.return %val -> (%t)
  } else {
    uir.return %val -> (%t)
  }
  // CHECK: uir.unreachable {{.*}}pa.phase = "0"
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// uir.call with all-zero phase offsets.

uir.func @CallTarget(%x: 0, %y: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// CHECK-LABEL: uir.func @CallAllZero
uir.func @CallAllZero(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  // CHECK: uir.call @CallTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @CallTarget(%a, %a) : (%ta, %ta) -> (%ta) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  uir.return %r -> (%ta)
}

//===----------------------------------------------------------------------===//
// Chained calls: result of one call feeds into the next.

uir.func @SingleArgTarget(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// CHECK-LABEL: uir.func @ChainedCalls
uir.func @ChainedCalls(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.call @SingleArgTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %inner = uir.call @SingleArgTarget(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  %ta2 = hir.int_type
  %tr2 = hir.int_type
  // CHECK: uir.call @SingleArgTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %outer = uir.call @SingleArgTarget(%inner) : (%ta2) -> (%tr2) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %outer -> (%tr2)
}

//===----------------------------------------------------------------------===//
// Call with floating (constant) args: literals satisfy any phase constraint.

// CHECK-LABEL: uir.func @CallWithLiterals
uir.func @CallWithLiterals() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %c10 = hir.constant_int 10 : %t
  %c20 = hir.constant_int 20 : %t
  %ta = hir.int_type
  %tb = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.call @CallTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @CallTarget(%c10, %c20) : (%ta, %tb) -> (%tr) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  uir.return %r -> (%tr)
}

//===----------------------------------------------------------------------===//
// Const arg passed to all-zero-offset call: arg at -1 is "over-available".

// CHECK-LABEL: uir.func @CallWithConstArg
uir.func @CallWithConstArg(%a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.call @SingleArgTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @SingleArgTarget(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r -> (%tr)
}

//===----------------------------------------------------------------------===//
// Sequential calls with pin (let bindings): call → pin → call → pin → call.

// CHECK-LABEL: uir.func @SequentialCalls
uir.func @SequentialCalls(%a: 0, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %ta1 = hir.int_type
  %tb1 = hir.int_type
  %tr1 = hir.int_type
  // CHECK: uir.call @CallTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %call1 = uir.call @CallTarget(%a, %b) : (%ta1, %tb1) -> (%tr1) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %x = uir.pin %call1, 0 : !hir.any

  %ta2 = hir.int_type
  %tb2 = hir.int_type
  %tr2 = hir.int_type
  // CHECK: uir.call @CallTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %call2 = uir.call @CallTarget(%x, %a) : (%ta2, %tb2) -> (%tr2) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %y = uir.pin %call2, 0 : !hir.any

  %ta3 = hir.int_type
  %tb3 = hir.int_type
  %tr3 = hir.int_type
  // CHECK: uir.call @CallTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %call3 = uir.call @CallTarget(%y, %b) : (%ta3, %tb3) -> (%tr3) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  uir.return %call3 -> (%tr3)
}

//===----------------------------------------------------------------------===//
// uir.call with const arg (offset -1).

uir.func @ConstArgTarget(%n: -1, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// CHECK-LABEL: uir.func @CallConstArg
uir.func @CallConstArg(%n: -1, %a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> (%t1)
} {
  %t = hir.int_type
  // Call at phase 0, arg 0 needs phase 0+(-1) = -1, arg 1 needs 0+0 = 0.
  // %n is at -1 (satisfies -1), %a is at 0 (satisfies 0).
  // CHECK: uir.call @ConstArgTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @ConstArgTarget(%n, %a) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  uir.return %r -> (%t)
}

//===----------------------------------------------------------------------===//
// uir.if without else region.

// CHECK-LABEL: uir.func @IfNoElse
uir.func @IfNoElse(%cond: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  uir.if %cond {
    // CHECK: func.call @dummyA() {{.*}}pa.phase = "0"
    func.call @dummyA() : () -> ()
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// uir.loop with continue.

// CHECK-LABEL: uir.func @LoopContinue
uir.func @LoopContinue(%c1: 0, %c2: 0) -> () {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> ()
} {
  uir.loop {
    uir.if %c1 {
      // CHECK: uir.continue {{.*}}pa.phase = "0"
      uir.continue
    } else {
      uir.unreachable
    }
    uir.if %c2 {
      uir.break
    } else {
      uir.unreachable
    }
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Nested pinned expr inside uir.if.

// CHECK-LABEL: uir.func @NestedExprInIf
uir.func @NestedExprInIf(%cond: 0, %a: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.if %cond : %t {
    %inner = uir.expr pin -1 : %t {
      // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
      %sum = hir.add %a, %a : %t
      uir.yield %sum : %t
    // CHECK: } {{.*}}pa.phase = "-1"
    }
    uir.yield %inner : %t
  } else {
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Type operand -1 constraint: hir.type_of on a const arg gets phase -1, which
// satisfies the -2 requirement from the add at phase -1. (type_of is pure,
// earliest = -1, constraint = -1-1 = -2, -1 > -2... actually this fails!)
// Let's test with hir.uint_type which takes a value operand.

// CHECK-LABEL: uir.func @TypeOperandConstraint
uir.func @TypeOperandConstraint(%n: -1) -> (result: 0) {
  // hir.uint_type %n is pure, operand at -1, so earliest = -1.
  // Signature pushes latest = argPhase - 1 = -1 - 1 = -2 for type of %n.
  // But uint_type is at -1 > -2, so this fails. Use hir.int_type instead.
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  // The add is pure, operands at -1, so earliest = -1. Type operand is
  // constant (floats), satisfies -1 - 1 = -2. All good.
  %t = hir.int_type
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
  %0 = hir.add %n, %n : %t
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Value used by both a pinned root AND a floating consumer.

// CHECK-LABEL: uir.func @PinAndFloatingConsumer
uir.func @PinAndFloatingConsumer(%a: 0) -> (result: 0) {
  %ty = hir.int_type
  uir.signature (%ty) -> (%ty)
} {
  %t = hir.int_type
  // %sum is used by both a pin (root) and the return (terminator).
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %sum = hir.add %a, %a : %t
  // Pin processes first (root), resolveValue skips on second visit (return).
  // CHECK: uir.pin {{.*}}, 0 {{.*}} {{.*}}pa.phase = "0"
  %p = uir.pin %sum, 0 : !hir.any
  uir.return %p -> (%t)
}

//===----------------------------------------------------------------------===//
// Pure op with zero operands floats.

// CHECK-LABEL: uir.func @PureZeroOperands
uir.func @PureZeroOperands() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  // hir.int_type is ConstantLike, floats to any phase.
  // CHECK: hir.int_type {{.*}}pa.phase = "float"
  %t = hir.int_type
  uir.return %t -> (%t)
}

//===----------------------------------------------------------------------===//
// Nested loop with const block containing continue.

// CHECK-LABEL: uir.func @NestedLoopConstContinue
uir.func @NestedLoopConstContinue(%c1: 0, %c2: 0) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> ()
} {
  uir.loop {
    uir.loop {
      uir.if %c1 {
        // CHECK: uir.continue {{.*}}pa.phase = "0"
        uir.continue
      } else {
        uir.unreachable
      }
      uir.if %c2 {
        uir.break
      } else {
        uir.unreachable
      }
      uir.yield
    }
    uir.if %c2 {
      uir.break
    } else {
      uir.unreachable
    }
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Zero-use uir.call (expression statement): arg offsets must be respected.

uir.func @VoidTarget(%n: -1, %x: 0) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> ()
} {
  uir.return -> ()
}

// CHECK-LABEL: uir.func @ZeroUseCall
uir.func @ZeroUseCall(%n: -1, %a: 0) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> ()
} {
  %ta = hir.int_type
  %tb = hir.int_type
  // Zero-use call at blockPhase 0. Arg 0 at 0+(-1)=-1, arg 1 at 0+0=0.
  // CHECK: uir.call @VoidTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  uir.call @VoidTarget(%n, %a) : (%ta, %tb) -> () (!hir.any, !hir.any) -> () [-1, 0] -> []
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Chained hir.type_of: type_of(type_of(x)) produces p(x) - 2.

// CHECK-LABEL: uir.func @ChainedTypeOf
uir.func @ChainedTypeOf(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  // type_of(%a at 0) = -1.
  // CHECK: hir.type_of %a {{.*}}pa.phase = "-1"
  %ta = hir.type_of %a
  // type_of(%ta at -1) = -2.
  // CHECK: hir.type_of {{.*}} {{.*}}pa.phase = "-2"
  %tta = hir.type_of %ta
  // Use %tta as the return type operand (needs phase 0-1 = -1; %tta at -2
  // satisfies since -2 <= -1).
  uir.return %a -> (%tta)
}

//===----------------------------------------------------------------------===//
// Pure op earliest scheduling respects type operand -1 rule.
// hir.uint_type %a is at phase -1 (pure, operand at -1). hir.add uses it as
// type operand. Without the fix, the add would float to -1, but the type at -1
// needs the op at -1+1=0 minimum. With the fix, earliest = max(-1, -1, -1+1)
// = 0.

// CHECK-LABEL: uir.func @PureTypeOperandEarliest
uir.func @PureTypeOperandEarliest(%a: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  // hir.uint_type %a: pure, operand at -1, earliest = -1.
  // CHECK: hir.uint_type {{.*}} {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %a
  // hir.add: pure. Value operands at -1, type operand at -1.
  // earliest = max(-1, -1, -1+1) = 0 (type operand contributes +1).
  // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
  %0 = hir.add %a, %a : %ut
  %rt = hir.int_type
  uir.return %0 -> (%rt)
}

//===----------------------------------------------------------------------===//
// Multi-result call with different result offsets. Result 0 at callPhase + 0,
// result 1 at callPhase + 1. Two consumers tighten the call from different
// results.

uir.func @MultiResultTarget(%x: 0) -> (r0: 0, r1: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0) -> (%t1, %t2)
} {
  %t = hir.int_type
  uir.return %x, %x -> (%t, %t)
}

// CHECK-LABEL: uir.func @MultiResultCall
uir.func @MultiResultCall(%a: -1) -> (r0: 0, r1: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0) -> (%t1, %t2)
} {
  %ta = hir.int_type
  %tr0 = hir.int_type
  %tr1 = hir.int_type
  // Return needs r0 at phase 0 and r1 at phase 0.
  // r0 = callPhase + 0, r1 = callPhase + 1.
  // For r1 at 0: callPhase + 1 = 0 → callPhase = -1.
  // For r0 at 0: callPhase + 0 = 0 → callPhase = 0.
  // Tightest: callPhase = min(0, -1) = -1.
  // So r0 = -1 + 0 = -1, r1 = -1 + 1 = 0.
  // Arg %a at -1, arg offset 0: constraint = -1 + 0 = -1. -1 <= -1: ok.
  // CHECK: uir.call @MultiResultTarget({{.*}}) {{.*}} {{.*}}pa.phase = "-1"
  %r0, %r1 = uir.call @MultiResultTarget(%a) : (%ta) -> (%tr0, %tr1) (!hir.any) -> (!hir.any, !hir.any) [0] -> [0, 1]
  %t = hir.int_type
  uir.return %r0, %r1 -> (%t, %t)
}

//===----------------------------------------------------------------------===//
// hir.type_of used by two consumers at different phases. The tightest
// constraint wins.

// CHECK-LABEL: uir.func @TypeOfMultiConsumer
uir.func @TypeOfMultiConsumer(%a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  // type_of(%a at 0) = -1. Used by two adds at phase 0: both push type
  // constraint latest = 0 - 1 = -1. type_of at -1 satisfies both.
  // CHECK: hir.type_of %a {{.*}}pa.phase = "-1"
  %ta = hir.type_of %a
  // CHECK: hir.add %a, %a : {{.*}} {{.*}}pa.phase = "0"
  %sum1 = hir.add %a, %a : %ta
  // CHECK: hir.add {{.*}}, %a : {{.*}} {{.*}}pa.phase = "0"
  %sum2 = hir.add %sum1, %a : %ta
  %t = hir.int_type
  uir.return %sum2 -> (%t)
}

//===----------------------------------------------------------------------===//
// Call with non-empty typeOfResults. The type-of-result operand gets
// callPhase + resultOffset - 1.

uir.func @TypedResultTarget(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x -> (%t2)
}

// CHECK-LABEL: uir.func @CallWithTypeOfResults
uir.func @CallWithTypeOfResults(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  // typeOfResults gets callPhase + 0 - 1 = -1. hir.int_type (float) satisfies.
  %tr = hir.int_type
  // CHECK: uir.call @TypedResultTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @TypedResultTarget(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r -> (%tr)
}

//===----------------------------------------------------------------------===//
// Multi-result uir.if: all results share the if's phase.

// CHECK-LABEL: uir.func @MultiResultIf
uir.func @MultiResultIf(%cond: 0, %a: 0, %b: 0) -> (r0: 0, r1: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  %t4 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3, %t4)
} {
  %t = hir.int_type
  // CHECK: uir.if {{.*}} {
  %r0, %r1 = uir.if %cond : %t, %t {
    uir.yield %a, %b : %t, %t
  } else {
    uir.yield %b, %a : %t, %t
  // CHECK: } {{.*}}pa.phase = "0"
  }
  uir.return %r0, %r1 -> (%t, %t)
}

//===----------------------------------------------------------------------===//
// Pure op floating inside uir.if body.

// CHECK-LABEL: uir.func @PureFloatInIf
uir.func @PureFloatInIf(%cond: 0, %a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0)
} {
  %t = hir.int_type
  %0 = uir.if %cond : %t {
    // Pure op, operands at -1. Floats to -1 even though the if body is at 0.
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %a : %t
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %sum : %t
  } else {
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// uir.return inside a floating uir.expr at body phase (should succeed).

// CHECK-LABEL: uir.func @ReturnInPinnedExpr
uir.func @ReturnInPinnedExpr(%cond: 0, %val: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0)
} {
  %t = hir.int_type
  // Pinned expr at phase 0. Return inside is at blockPhase = 0 =
  // funcBodyPhase, so it's allowed.
  uir.if %cond {
    uir.expr pin {
      // CHECK: uir.return {{.*}} -> ({{.*}}) {{.*}}pa.phase = "0"
      uir.return %val -> (%t)
    }
    uir.yield
  } else {
    uir.yield
  }
  uir.return %val -> (%t)
}

//===----------------------------------------------------------------------===//
// hir.constant_int type operand gets -1 constraint.

// CHECK-LABEL: uir.func @ConstantIntTypeOperand
uir.func @ConstantIntTypeOperand() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  // hir.int_type is constant (float). constant_int's type operand gets
  // phase - 1 = 0 - 1 = -1. hir.int_type (float) satisfies.
  %t = hir.int_type
  // CHECK: hir.constant_int {{.*}} {{.*}}pa.phase = "float"
  %c = hir.constant_int 42 : %t
  uir.return %c -> (%t)
}

//===----------------------------------------------------------------------===//
// hir.inferrable: not Pure, but isEffectivelyPure. Should float (no operands).

// CHECK-LABEL: uir.func @InferrableFloats
uir.func @InferrableFloats() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  // CHECK: hir.inferrable {{.*}}pa.phase = "float"
  %inf = hir.inferrable
  %t = hir.int_type
  uir.return %inf -> (%t)
}

//===----------------------------------------------------------------------===//
// hir.unify: has MemWrite (to prevent DCE), but isEffectivelyPure.
// Should float to earliest = max(operand phases).

// CHECK-LABEL: uir.func @UnifyPhase
uir.func @UnifyPhase(%a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  // unify: effectively pure, earliest = max(-2, -1) = -1.
  // CHECK: hir.unify {{.*}} {{.*}}pa.phase = "-1"
  %t = hir.unify %ta, %tb
  uir.return %a -> (%t)
}

//===----------------------------------------------------------------------===//
// hir.coerce_type: operand 1 ($typeOperand) gets -1 constraint.

// CHECK-LABEL: uir.func @CoerceTypePhase
uir.func @CoerceTypePhase(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  // The coerce_type is not pure, so it stays at the consumer's latest (0).
  // Its type operand (hir.int_type) gets 0 - 1 = -1 (float satisfies).
  %ty = hir.int_type
  // CHECK: hir.coerce_type {{.*}} {{.*}}pa.phase = "0"
  %c = hir.coerce_type %a, %ty
  %t = hir.int_type
  uir.return %c -> (%t)
}

//===----------------------------------------------------------------------===//
// Call with non-constant typeOfArgs: uint_type(%n) as arg type.

uir.func @TypedArgTarget(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x -> (%t2)
}

// CHECK-LABEL: uir.func @CallNonConstTypeOfArgs
uir.func @CallNonConstTypeOfArgs(%n: -1, %a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // hir.uint_type %n: pure, operand at -1, earliest = -1.
  // Call at phase 0. typeOfArgs[0] needs callPhase + argOffset - 1 = 0 + 0 - 1
  // = -1. uint_type at -1 satisfies.
  // CHECK: hir.uint_type {{.*}} {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %n
  %tr = hir.int_type
  // CHECK: uir.call @TypedArgTarget({{.*}}) {{.*}} {{.*}}pa.phase = "0"
  %r = uir.call @TypedArgTarget(%a) : (%ut) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r -> (%tr)
}
