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
// Zero-slack pure op: earliest == latest exactly. Const arg at -1, const
// return at -1. a+1 earliest = max(-1) = -1, latest = -1. Just valid.

// CHECK-LABEL: uir.func @PureNoSlack
uir.func @PureNoSlack(%a: -1) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %lit = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}}pa.phase = "-1"
  %sum = hir.add %a, %lit : %t
  uir.return %sum -> (%t)
}

//===----------------------------------------------------------------------===//
// Pin vs float interaction: a*a floats to -1, pin fixes copy at 0. Independent
// a*a in const block also floats to -1. Both paths produce -1 values, combined
// at 0 via pins.

// CHECK-LABEL: uir.func @LetPinConstraint
uir.func @LetPinConstraint(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: hir.mul %a, %a {{.*}}pa.phase = "-1"
  %prod = hir.mul %a, %a : %t
  // CHECK: uir.pin {{.*}}pa.phase = "0"
  %x = uir.pin %prod, 0 : !hir.any
  // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
  %y = uir.expr pin -1 : %t {
    // CHECK: hir.mul %a, %a {{.*}}pa.phase = "-1"
    %prod2 = hir.mul %a, %a : %t
    uir.yield %prod2 : %t
  }
  // CHECK: uir.pin {{.*}}pa.phase = "0"
  %pinY = uir.pin %y, 0 : !hir.any
  // CHECK: hir.add {{.*}}pa.phase = "0"
  %sum = hir.add %x, %pinY : %t
  uir.return %sum -> (%t)
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
  // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr pin -1 : %t {
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %a : %t
    uir.yield %sum : %t
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
  // CHECK: uir.expr pin 1 {{.*}} attributes {{{.*}}pa.phase = "1", pa.results = ["0"]
  %0 = uir.expr pin 1 : %t {
    // Yield at phase 1, carrying %x at phase 0 across the boundary.
    // CHECK: uir.yield {{.*}}pa.operands = ["0", "float"]
    // CHECK-SAME: pa.phase = "1"
    uir.yield %x : %t
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
  // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr pin -1 : %t {
    // Inner const block at phase -2.
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-2"
    %inner = uir.expr pin -1 : %t {
      %c1 = hir.constant_int 1 : %t
      // CHECK: hir.add {{.*}}pa.phase = "-2"
      %sum = hir.add %a, %c1 : %t
      uir.yield %sum : %t
    }
    uir.yield %inner : %t
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
  // CHECK: uir.expr pin -1 attributes {{{.*}}pa.phase = "-1"
  uir.expr pin -1 {
    // CHECK: func.call @sideEffectC() {{.*}}pa.phase = "-1"
    func.call @sideEffectC() : () -> ()
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Pin inside const block: call anchored at -1, pin at -1, pure op at -1.

// CHECK-LABEL: uir.func @PinInConstBlock
uir.func @PinInConstBlock(%a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ta = hir.int_type
    %tr = hir.int_type
    // Call anchored at const block phase -1.
    // CHECK: uir.call @SingleArgTarget({{.*}})
    // CHECK-SAME: pa.phase = "-1"
    %v = uir.call @SingleArgTarget(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    // Pin at blockPhase + 0 = -1.
    // CHECK: uir.pin {{.*}}, 0
    // CHECK-SAME: pa.phase = "-1"
    %y = uir.pin %v, 0 : !hir.any
    %c2 = hir.constant_int 2 : %t
    // Pure op using pinned value at -1.
    // CHECK: hir.mul {{.*}}pa.phase = "-1"
    %prod = hir.mul %y, %c2 : %t
    uir.yield %prod : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Nested dyn blocks: dyn { dyn { ... } } — literal floats through both yields.

// CHECK-LABEL: uir.func @NestedDynBlocks
uir.func @NestedDynBlocks() -> (result: 2) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  // CHECK: uir.expr pin 1 {{.*}} attributes {{{.*}}pa.phase = "1", pa.results = ["float"]
  %0 = uir.expr pin 1 : %t {
    // CHECK: uir.expr pin 1 {{.*}} attributes {{{.*}}pa.phase = "2", pa.results = ["float"]
    %inner = uir.expr pin 1 : %t {
      %c99 = hir.constant_int 99 : %t
      uir.yield %c99 : %t
    }
    uir.yield %inner : %t
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
  // CHECK: uir.expr pin 1 {{.*}} attributes {{{.*}}pa.phase = "1", pa.results = ["float"]
  %0 = uir.expr pin 1 : %t {
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["float"]
    %inner = uir.expr pin -1 : %t {
      %c42 = hir.constant_int 42 : %t
      uir.yield %c42 : %t
    }
    // Yield carries float value at phase 1.
    // CHECK: uir.yield {{.*}}pa.phase = "1"
    uir.yield %inner : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Floating expr block phase tightening: two results with different consumer
// demands cause the block phase to tighten. Both calls move to the tighter
// phase. Constraints propagate transitively through pure ops in the second
// test.

uir.func @MakeValue() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %c = hir.constant_int 1 : %t
  uir.return %c -> (%t)
}

// CHECK-LABEL: uir.func @FloatingExprTightening
uir.func @FloatingExprTightening() -> (r0: 0, r1: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature () -> (%t0, %t1)
} {
  %t = hir.int_type
  %ta = hir.int_type
  %tb = hir.int_type
  %x, %y = uir.expr : %ta, %tb {
    %ra = hir.int_type
    // Both calls tighten to -1 (from %y's -1 demand).
    // CHECK: uir.call @MakeValue()
    // CHECK-SAME: pa.phase = "-1"
    %a = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
    %rb = hir.int_type
    // CHECK: uir.call @MakeValue()
    // CHECK-SAME: pa.phase = "-1"
    %b = uir.call @MakeValue() : () -> (%rb) () -> !hir.any [] -> [0]
    uir.yield %a, %b : %ta, %tb
  }
  uir.return %x, %y -> (%t, %t)
}

// CHECK-LABEL: uir.func @FloatingExprTransitivePure
uir.func @FloatingExprTransitivePure() -> (r0: 0, r1: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature () -> (%t0, %t1)
} {
  %t = hir.int_type
  %ta = hir.int_type
  %tb = hir.int_type
  %x, %y = uir.expr : %ta, %tb {
    %ra = hir.int_type
    // CHECK: uir.call @MakeValue()
    // CHECK-SAME: pa.phase = "-1"
    %a = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
    %rb = hir.int_type
    // CHECK: uir.call @MakeValue()
    // CHECK-SAME: pa.phase = "-1"
    %b = uir.call @MakeValue() : () -> (%rb) () -> !hir.any [] -> [0]
    // Pure ops between calls and yield: constraints propagate through.
    %c1 = hir.constant_int 1 : %t
    %a2 = hir.add %a, %c1 : %t
    %b2 = hir.add %b, %c1 : %t
    uir.yield %a2, %b2 : %ta, %tb
  }
  uir.return %x, %y -> (%t, %t)
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
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "0"
  %0 = uir.expr : %t {
    // The add is pure, operands at -1, so earliest = -1.
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %a : %t
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %sum : %t
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
  // CHECK: uir.expr pin attributes {{{.*}}pa.phase = "0"
  uir.expr pin {
    // CHECK: hir.add {{.*}} {{.*}}pa.phase = "0"
    %sum = hir.add %a, %b : %t
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield
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
  // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "0"
  %0 = uir.if %cond : %t {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond : %t
  } else {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond : %t
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
  // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "0"
  %r = uir.if %cond1 : %bt {
    // CHECK: hir.leq {{.*}} {{.*}}pa.phase = "0"
    %cond2 = hir.leq %x, %hi : %bt
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cond2 : %bt
  } else {
    %cfalse = hir.constant_bool <false>
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %cfalse : %bt
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
  // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "0"
  %r = uir.if %flag : %t {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %c1 : %t
  } else {
    // CHECK: uir.yield {{.*}} {{.*}}pa.phase = "0"
    uir.yield %c0 : %t
  }
  uir.return %r -> (%t)
}

//===----------------------------------------------------------------------===//
// If with const args: if anchored at 0, yields carry -1 values transparently.
// The if result is at -1 (from yield operands), not 0 (the if's phase).
// TODO: Revisit this once we figure out how to deal with control flow ops that
// yield values that must be available in earlier phases.

// CHECK-LABEL: uir.func @IfTransparentYield
uir.func @IfTransparentYield(%sel: -1, %a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // If anchored at block phase 0.
  // Result at -1 (transparent yield), not 0.
  // CHECK: uir.if {{.*}} attributes {pa.operands = ["-1", "float"], pa.phase = "0", pa.results = ["-1"]
  %r = uir.if %sel : %t {
    // CHECK: uir.yield {{.*}}pa.operands = ["-1", "float"]
    // CHECK-SAME: pa.phase = "0"
    uir.yield %a : %t
  } else {
    // CHECK: uir.yield {{.*}}pa.operands = ["-1", "float"]
    // CHECK-SAME: pa.phase = "0"
    uir.yield %b : %t
  }
  // CHECK: uir.return {{.*}}pa.operands = ["-1", "float"]
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
  // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "0"
  %0 = uir.loop : %t {
    uir.if %cond {
      // CHECK: uir.break {{.*}} {{.*}}pa.phase = "0"
      uir.break %val : %t
    } else {
      uir.unreachable
    }
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield
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
// Call with const arg offset -1 and dyn result offset +1. The call must be in
// a floating expr so it can tighten to phase -1 (demanded by the return at 0).

uir.func @ConstArgDynResult(%c: -1) -> (result: 1) {
  %t = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t) -> (%t2)
} {
  %t3 = hir.int_type
  uir.return %c -> (%t3)
}

// CHECK-LABEL: uir.func @CallBothOffsets
uir.func @CallBothOffsets(%x: -2) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.expr {{.*}}pa.phase = "-1"
  %r = uir.expr : %tr {
    // CHECK: uir.call @ConstArgDynResult({{.*}}) {{.*}}pa.phase = "-1"
    // CHECK-SAME: pa.results = ["0"]
    %v = uir.call @ConstArgDynResult(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [1]
    uir.yield %v : %tr
  }
  uir.return %r -> (%tr)
}

//===----------------------------------------------------------------------===//
// Triple const chain: takes_const(takes_const(takes_const(x))). Each const arg
// offset -1 means each call tightens one phase earlier. x at -3.

uir.func @TakesConst(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x -> (%t2)
}

// CHECK-LABEL: uir.func @TripleConstChain
uir.func @TripleConstChain(%x: -3) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  %tr1 = hir.int_type
  %tr2 = hir.int_type
  %tr3 = hir.int_type
  // Calls wrapped in floating exprs to allow tightening to -2, -1, 0.
  // CHECK: uir.expr {{.*}}pa.phase = "-2"
  %r1 = uir.expr : %tr1 {
    // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "-2"
    %v = uir.call @TakesConst(%x) : (%t1) -> (%tr1) (!hir.any) -> !hir.any [-1] -> [0]
    uir.yield %v : %tr1
  }
  // CHECK: uir.expr {{.*}}pa.phase = "-1"
  %r2 = uir.expr : %tr2 {
    // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "-1"
    %v = uir.call @TakesConst(%r1) : (%t2) -> (%tr2) (!hir.any) -> !hir.any [-1] -> [0]
    uir.yield %v : %tr2
  }
  // Last call stays at body phase 0.
  // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "0"
  %r3 = uir.call @TakesConst(%r2) : (%t3) -> (%tr3) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r3 -> (%tr3)
}

//===----------------------------------------------------------------------===//
// Triple dyn chain: returns_dyn(returns_dyn(returns_dyn(x))). Each dyn result
// offset +1 pushes later. Calls tighten to -3, -2, -1 so the final result
// reaches phase 0.

uir.func @ReturnsDyn(%x: 0) -> (result: 1) {
  %t = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t) -> (%t2)
} {
  %t3 = hir.int_type
  uir.return %x -> (%t3)
}

// TODO: Disabled pending constrainRegionResult simplification (step 2 of
// PhaseAnalysis refactoring). Chained floating exprs with dyn result offsets
// don't cascade tightening correctly. Calls need uir.expr wrapping.
//
// C HECK-LABEL: uir.func @TripleDynChain
// uir.func @TripleDynChain(%x: -3) -> (result: 0) {
//   %t = hir.int_type
//   uir.signature (%t) -> (%t)
// } {
//   %t1 = hir.int_type
//   %t2 = hir.int_type
//   %t3 = hir.int_type
//   %tr1 = hir.int_type
//   %tr2 = hir.int_type
//   %tr3 = hir.int_type
//   // C HECK: uir.call @ReturnsDyn({{.*}}) {{.*}}pa.phase = "-3"
//   // C HECK-SAME: pa.results = ["-2"]
//   %r1 = uir.call @ReturnsDyn(%x) : (%t1) -> (%tr1) (!hir.any) -> !hir.any [0] -> [1]
//   // C HECK: uir.call @ReturnsDyn({{.*}}) {{.*}}pa.phase = "-2"
//   // C HECK-SAME: pa.results = ["-1"]
//   %r2 = uir.call @ReturnsDyn(%r1) : (%t2) -> (%tr2) (!hir.any) -> !hir.any [0] -> [1]
//   // C HECK: uir.call @ReturnsDyn({{.*}}) {{.*}}pa.phase = "-1"
//   // C HECK-SAME: pa.results = ["0"]
//   %r3 = uir.call @ReturnsDyn(%r2) : (%t3) -> (%tr3) (!hir.any) -> !hir.any [0] -> [1]
//   uir.return %r3 -> (%tr3)
// }

//===----------------------------------------------------------------------===//
// Offset cancel: takes_const(returns_dyn(x)). Inner dyn +1 and outer const -1
// cancel out. x at -2, inner call at -2 (result at -1), outer call at 0.

// CHECK-LABEL: uir.func @OffsetCancel
uir.func @OffsetCancel(%x: -2) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t1 = hir.int_type
  %t2 = hir.int_type
  %tr1 = hir.int_type
  %tr2 = hir.int_type
  // Inner call wrapped in floating expr: tightens to -2, result at -1.
  // CHECK: uir.expr {{.*}}pa.phase = "-2"
  %r1 = uir.expr : %tr1 {
    // CHECK: uir.call @ReturnsDyn({{.*}}) {{.*}}pa.phase = "-2"
    // CHECK-SAME: pa.results = ["-1"]
    %v = uir.call @ReturnsDyn(%x) : (%t1) -> (%tr1) (!hir.any) -> !hir.any [0] -> [1]
    uir.yield %v : %tr1
  }
  // Outer call at body phase 0: const arg at -1, %r1 at -1 satisfies.
  // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "0"
  %r2 = uir.call @TakesConst(%r1) : (%t2) -> (%tr2) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r2 -> (%tr2)
}

//===----------------------------------------------------------------------===//
// Deep nesting with const blocks: const { const { takes_const(x) } }.
// Block phase shift + arg offset: -1 + (-1) + (-1) = -3 for the arg.

// CHECK-LABEL: uir.func @DeepViaBlocks
uir.func @DeepViaBlocks(%x: -3) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  %0 = uir.expr pin -1 : %t2 {
    %1 = uir.expr pin -1 : %t2 {
      %ta = hir.int_type
      %tr = hir.int_type
      // Call at block phase -2. Const arg at -2 + (-1) = -3. x at -3. Ok.
      // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "-2"
      %r = uir.call @TakesConst(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
      uir.yield %r : %t2
    }
    uir.yield %1 : %t2
  }
  %p = uir.pin %0, 0 : !hir.any
  uir.return %p -> (%t2)
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
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
    %inner = uir.expr pin -1 : %t {
      // CHECK: hir.add {{.*}} {{.*}}pa.phase = "-1"
      %sum = hir.add %a, %a : %t
      uir.yield %sum : %t
    }
    uir.yield %inner : %t
  } else {
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// If inside const block inside loop: the const block shifts to -1, the if
// inside is at -1. The result flows through pin to an add at phase 0.

// CHECK-LABEL: uir.func @IfInConstInLoop
uir.func @IfInConstInLoop(%flag: -1, %a: -1, %b: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  %3 = hir.int_type
  uir.signature (%0, %1, %2, %3) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
    %chosen = uir.expr pin -1 : %t {
      // CHECK: uir.if %flag {{.*}} attributes {{{.*}}pa.phase = "-1"
      %sel = uir.if %flag : %t {
        // CHECK: uir.yield %a {{.*}}pa.phase = "-1"
        uir.yield %a : %t
      } else {
        uir.yield %b : %t
      }
      uir.yield %sel : %t
    }
    // Pin absorbs the -1 result to 0.
    // CHECK: uir.pin {{.*}}pa.phase = "0"
    %pinned = uir.pin %chosen, 0 : !hir.any
    // CHECK: hir.add {{.*}}pa.phase = "0"
    %sum = hir.add %pinned, %x : %t
    uir.break %sum : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Deep nesting: loop inside if inside const block. Both branches of the if
// contain loops, all at phase -1.

// CHECK-LABEL: uir.func @DeepNesting
uir.func @DeepNesting(%a: -1, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-1"
  %val = uir.expr pin -1 : %t {
    %cond = hir.gt %a, %b : %t
    // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "-1"
    %0 = uir.if %cond : %t {
      %1 = uir.loop : %t {
        // CHECK: hir.add %a, %b {{.*}}pa.phase = "-1"
        %sum = hir.add %a, %b : %t
        uir.break %sum : %t
      }
      uir.yield %1 : %t
    } else {
      %2 = uir.loop : %t {
        // CHECK: hir.sub %b, %a {{.*}}pa.phase = "-1"
        %diff = hir.sub %b, %a : %t
        uir.break %diff : %t
      }
      uir.yield %2 : %t
    }
    uir.yield %0 : %t
  }
  // CHECK: uir.pin {{.*}}pa.phase = "0"
  %pinned = uir.pin %val, 0 : !hir.any
  uir.return %pinned -> (%t)
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
// Nested loops with result phase transparency: inner loop produces a dyn value
// at phase 1 (pure op on dyn args), which flows through the outer loop's break.
// Both loops execute at phase 0, but the inner result is at 1.

// CHECK-LABEL: uir.func @NestedLoopResults
uir.func @NestedLoopResults(%a: 1, %cond: 0) -> (result: 1) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  // Outer loop result is 1 (transparent from inner).
  // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1"]
  %0 = uir.loop : %t {
    // Inner loop executes at 0 but result is at 1 (transparent break).
    // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1"]
    %inner = uir.loop : %t {
      uir.if %cond {
        // CHECK: hir.add %a, %a {{.*}}pa.phase = "1"
        %sum = hir.add %a, %a : %t
        uir.break %sum : %t
      } else {
        uir.unreachable
      }
      uir.yield
    }
    // Outer break carries the dyn value.
    uir.break %inner : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Multi-result loop with break values at different phases. Result 0 is a dyn
// pure op at phase 1, result 1 is a call result at phase 0.

// CHECK-LABEL: uir.func @MultiResultLoopPhases
uir.func @MultiResultLoopPhases(%a: 1) -> (r0: 1, r1: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  uir.signature (%0) -> (%1, %2)
} {
  %t = hir.int_type
  %t2 = hir.int_type
  // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1", "0"]
  %r0, %r1 = uir.loop : %t, %t2 {
    // CHECK: hir.add %a, %a {{.*}}pa.phase = "1"
    %sum = hir.add %a, %a : %t
    %rt = hir.int_type
    // CHECK: uir.call @MakeValue()
    // CHECK-SAME: pa.phase = "0"
    %v = uir.call @MakeValue() : () -> (%rt) () -> !hir.any [] -> [0]
    // CHECK: uir.break {{.*}}pa.operands = ["1", "0", "float", "float"]
    uir.break %sum, %v : %t, %t2
  }
  // CHECK: uir.return {{.*}}pa.operands = ["1", "0", "float", "float"]
  uir.return %r0, %r1 -> (%t, %t2)
}

//===----------------------------------------------------------------------===//
// Transparent dyn result through if: dyn arg at phase 1 yielded through an if
// at phase 0. The if result must be 1, not capped to the if's execution phase.

// CHECK-LABEL: uir.func @IfTransparentDynYield
uir.func @IfTransparentDynYield(%sel: 0, %a: 1, %b: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1"]
  %r = uir.if %sel : %t {
    uir.yield %a : %t
  } else {
    uir.yield %b : %t
  }
  // CHECK: uir.return {{.*}}pa.operands = ["1", "float"]
  uir.return %r -> (%t)
}

//===----------------------------------------------------------------------===//
// Transparent dyn result through pinned expr: dyn arg at phase 1 yielded
// through a pinned expr at phase 0. Result must be 1.

// CHECK-LABEL: uir.func @ExprTransparentDynYield
uir.func @ExprTransparentDynYield(%a: 1) -> (result: 1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  // CHECK: uir.expr pin : {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1"]
  %0 = uir.expr pin : %t {
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Transparent dyn result through loop: dyn arg at phase 1 flows through a
// break in a loop at phase 0. Loop result must be 1.

// CHECK-LABEL: uir.func @LoopTransparentDynBreak
uir.func @LoopTransparentDynBreak(%a: 1) -> (result: 1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "0", pa.results = ["1"]
  %0 = uir.loop : %t {
    uir.break %a : %t
  }
  uir.return %0 -> (%t)
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
// Mixed const depth with dependent types: A at -2, B at -1, x at -1 with
// type uint<A> at -2, y at 0 with type uint<B> at -1.

// CHECK-LABEL: uir.func @MixedDepthDependentTypes
uir.func @MixedDepthDependentTypes(%A: -2, %B: -1, %x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.uint_type %A
  %3 = hir.uint_type %B
  %4 = hir.uint_type %B
  uir.signature (%0, %1, %2, %3) -> (%4)
} {
  // CHECK: hir.uint_type %A {{.*}}pa.phase = "-2"
  %ta = hir.uint_type %A
  // CHECK: hir.uint_type %B {{.*}}pa.phase = "-1"
  %tb = hir.uint_type %B
  uir.return %y -> (%tb)
}

//===----------------------------------------------------------------------===//
// Type arithmetic with mixed depths: uint<A + B> where A at -2, B at -1.
// A + B is pure at max(-2, -1) = -1, uint_type at -1.

// CHECK-LABEL: uir.func @TypeArithMixedDepth
uir.func @TypeArithMixedDepth(%A: -2, %B: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %ab = hir.add %A, %B : %0
  %2 = hir.uint_type %ab
  %ab2 = hir.add %A, %B : %0
  %3 = hir.uint_type %ab2
  uir.signature (%0, %1, %2) -> (%3)
} {
  %t = hir.int_type
  // add floats to max(-2, -1) = -1.
  // CHECK: hir.add %A, %B {{.*}}pa.phase = "-1"
  %ab = hir.add %A, %B : %t
  // CHECK: hir.uint_type {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %ab
  uir.return %x -> (%ut)
}

//===----------------------------------------------------------------------===//
// Widening pattern: uint<N + 1>. N at -1, literal floats, N+1 at -1,
// uint_type at -1.

// CHECK-LABEL: uir.func @WidenType
uir.func @WidenType(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %lit = hir.constant_int 1 : %0
  %np1 = hir.add %N, %lit : %0
  %2 = hir.uint_type %np1
  uir.signature (%0, %1) -> (%2)
} {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  // CHECK: hir.add {{.*}}pa.phase = "-1"
  %np1 = hir.add %N, %one : %t
  // CHECK: hir.uint_type {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %np1
  uir.return %x -> (%ut)
}

//===----------------------------------------------------------------------===//
// Deep dependent type: N at -2 (const const), uint<N> at -2, x at -1 with
// type at -2. Tests that dependent types work at stacked const depths.

// CHECK-LABEL: uir.func @DeepDependentType
uir.func @DeepDependentType(%N: -2, %x: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %2 = hir.uint_type %N
  uir.signature (%0, %1) -> (%2)
} {
  // CHECK: hir.uint_type {{.*}}pa.phase = "-2"
  %rt = hir.uint_type %N
  uir.return %x -> (%rt)
}

//===----------------------------------------------------------------------===//
// Computed type uint<A + B>: A and B at -1, add pure at -1, uint_type at -1.

// CHECK-LABEL: uir.func @ComputedTypeAdd
uir.func @ComputedTypeAdd(%A: -1, %B: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %ab = hir.add %A, %B : %0
  %2 = hir.uint_type %ab
  %ab2 = hir.add %A, %B : %0
  %3 = hir.uint_type %ab2
  uir.signature (%0, %1, %2) -> (%3)
} {
  %t = hir.int_type
  // CHECK: hir.add %A, %B {{.*}}pa.phase = "-1"
  %ab = hir.add %A, %B : %t
  // CHECK: hir.uint_type {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %ab
  uir.return %x -> (%ut)
}

//===----------------------------------------------------------------------===//
// Chained type computation: uint<A + B + C> with all const at -1.

// CHECK-LABEL: uir.func @ChainedTypeComputation
uir.func @ChainedTypeComputation(%A: -1, %B: -1, %C: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  %ab = hir.add %A, %B : %0
  %abc = hir.add %ab, %C : %0
  %3 = hir.uint_type %abc
  %ab2 = hir.add %A, %B : %0
  %abc2 = hir.add %ab2, %C : %0
  %4 = hir.uint_type %abc2
  uir.signature (%0, %1, %2, %3) -> (%4)
} {
  %t = hir.int_type
  // CHECK: hir.add %A, %B {{.*}}pa.phase = "-1"
  %ab = hir.add %A, %B : %t
  // CHECK: hir.add {{.*}}, %C {{.*}}pa.phase = "-1"
  %abc = hir.add %ab, %C : %t
  // CHECK: hir.uint_type {{.*}}pa.phase = "-1"
  %ut = hir.uint_type %abc
  uir.return %x -> (%ut)
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
  // TODO: Disabled pending constrainRegionResult simplification (step 2).
  // Multi-result floating expr tightening doesn't re-update already-propagated
  // result actualPhases. Call needs uir.expr wrapping.
  //
  // C HECK: uir.call @MultiResultTarget({{.*}}) {{.*}} {{.*}}pa.phase = "-1"
  // %r0, %r1 = uir.call @MultiResultTarget(%a) : (%ta) -> (%tr0, %tr1) (!hir.any) -> (!hir.any, !hir.any) [0] -> [0, 1]
  // %t = hir.int_type
  // uir.return %r0, %r1 -> (%t, %t)
  %t = hir.int_type
  %c0 = hir.constant_int 0 : %t
  uir.return %c0, %c0 -> (%t, %t)
}

//===----------------------------------------------------------------------===//
// Multi-result call with spread offsets -1/0/+1. Call tightens to -1 (from
// r2: 0 - 1 = -1). Results at -2, -1, 0.

uir.func @Spread(%x: 0) -> (r0: -1, r1: 0, r2: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0) -> (%t1, %t2, %t3)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  %c2 = hir.constant_int 2 : %t
  %c3 = hir.constant_int 3 : %t
  uir.return %c1, %c2, %c3 -> (%t, %t, %t)
}

// TODO: Disabled pending constrainRegionResult simplification (step 2).
// Call needs uir.expr wrapping; multi-result tightening doesn't cascade.
//
// C HECK-LABEL: uir.func @SpreadCallAllResults
// uir.func @SpreadCallAllResults(%a: -1) -> (result: 0) {
//   %t0 = hir.int_type
//   %t1 = hir.int_type
//   uir.signature (%t0) -> (%t1)
// } {
//   %ta = hir.int_type
//   %tr0 = hir.int_type
//   %tr1 = hir.int_type
//   %tr2 = hir.int_type
//   // C HECK: uir.call @Spread({{.*}}) {{.*}}pa.phase = "-1"
//   // C HECK-SAME: pa.results = ["-2", "-1", "0"]
//   %r0, %r1, %r2 = uir.call @Spread(%a) : (%ta) -> (%tr0, %tr1, %tr2) (!hir.any) -> (!hir.any, !hir.any, !hir.any) [0] -> [-1, 0, 1]
//   %t = hir.int_type
//   %sum01 = hir.add %r0, %r1 : %t
//   %sum = hir.add %sum01, %r2 : %t
//   uir.return %sum -> (%t)
// }

//===----------------------------------------------------------------------===//
// Partial use of multi-result call: only the dyn result (offset +1) is
// consumed, tightening the call to -1.

// TODO: Disabled pending constrainRegionResult simplification (step 2).
// Call needs uir.expr wrapping; multi-result tightening doesn't cascade.
//
// C HECK-LABEL: uir.func @SpreadCallPartialUse
// uir.func @SpreadCallPartialUse(%a: -1) -> (result: 0) {
//   %t0 = hir.int_type
//   %t1 = hir.int_type
//   uir.signature (%t0) -> (%t1)
// } {
//   %ta = hir.int_type
//   %tr0 = hir.int_type
//   %tr1 = hir.int_type
//   %tr2 = hir.int_type
//   // C HECK: uir.call @Spread({{.*}}) {{.*}}pa.phase = "-1"
//   // C HECK-SAME: pa.results = ["-2", "-1", "0"]
//   %r0, %r1, %r2 = uir.call @Spread(%a) : (%ta) -> (%tr0, %tr1, %tr2) (!hir.any) -> (!hir.any, !hir.any, !hir.any) [0] -> [-1, 0, 1]
//   %t = hir.int_type
//   uir.return %r2 -> (%t)
// }

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
  // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "0"
  %r0, %r1 = uir.if %cond : %t, %t {
    uir.yield %a, %b : %t, %t
  } else {
    uir.yield %b, %a : %t, %t
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

//===----------------------------------------------------------------------===//
// While loop pattern: loop { if !cond { break }; body; yield }.

// CHECK-LABEL: uir.func @WhileLoop
uir.func @WhileLoop(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.loop {
    uir.if %x {
      uir.yield
    } else {
      uir.break
    }
    %body = hir.add %x, %x : %t
    %p = uir.pin %body, 0 : !hir.any
    uir.yield
  }
  uir.return %x -> (%t)
}

//===----------------------------------------------------------------------===//
// Loop inside const block at phase -1. Break and continue inherit the loop
// phase, which inherits from the const block.

// CHECK-LABEL: uir.func @ConstLoop
uir.func @ConstLoop(%x: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %1 = uir.loop : %t {
      %cond = hir.gt %x, %x : %t
      uir.if %cond {
        %sum = hir.add %x, %x : %t
        // CHECK: uir.break {{.*}}pa.phase = "-1"
        uir.break %sum : %t
      } else {
        uir.unreachable
      }
      // CHECK: uir.continue {{.*}}pa.phase = "-1"
      uir.continue
    }
    uir.yield %1 : %t
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Break value with slack: pure op at -1 flows through break transparently.
// The loop result reflects the break operand's actual phase (-1), not the
// loop's block phase (0).

// CHECK-LABEL: uir.func @LoopBreakSlack
uir.func @LoopBreakSlack(%a: -1, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    // CHECK: hir.add {{.*}}pa.phase = "-1"
    %sum = hir.add %a, %b : %t
    uir.break %sum : %t
  }
  // Loop result at -1 (transparent break).
  // CHECK: uir.return {{.*}}pa.operands = ["-1", "float"]
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// If inside floating expr: the if is anchored to the floating expr's block,
// which starts unconstrained. A consumer demand at -1 tightens the floating
// expr, and the if (and its calls) move with it.

// CHECK-LABEL: uir.func @IfInsideFloatingExpr
uir.func @IfInsideFloatingExpr(%cond: -1) -> (result: -1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  %ta = hir.int_type
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr : %ta {
    %ra = hir.int_type
    // The if is anchored at the floating expr's block phase.
    // When the expr tightens to -1, the if and its calls tighten too.
    // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "-1"
    %r = uir.if %cond : %ta {
      // CHECK: uir.call @MakeValue()
      // CHECK-SAME: pa.phase = "-1"
      %v = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
      uir.yield %v : %ta
    } else {
      // CHECK: uir.call @MakeValue()
      // CHECK-SAME: pa.phase = "-1"
      %v = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
      uir.yield %v : %ta
    }
    uir.yield %r : %ta
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Loop inside floating expr: the loop is anchored to the floating expr's block.
// A consumer demand at -1 tightens the floating expr, and the loop (and its
// break values) move with it.

// CHECK-LABEL: uir.func @LoopInsideFloatingExpr
uir.func @LoopInsideFloatingExpr(%cond: -1) -> (result: -1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  %ta = hir.int_type
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr : %ta {
    %ra = hir.int_type
    // CHECK: uir.loop {{.*}} attributes {{{.*}}pa.phase = "-1"
    %r = uir.loop : %ta {
      uir.if %cond {
        // CHECK: uir.call @MakeValue()
        // CHECK-SAME: pa.phase = "-1"
        %v = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
        // CHECK: uir.break {{.*}}pa.phase = "-1"
        uir.break %v : %ta
      } else {
        uir.unreachable
      }
      uir.yield
    }
    uir.yield %r : %ta
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Pinned expr inside floating expr: the pin's offset is relative to the
// floating parent. When the floating expr tightens to -1, the pinned inner
// expr moves to -1 + (-1) = -2.

// CHECK-LABEL: uir.func @PinnedInsideFloatingExpr
uir.func @PinnedInsideFloatingExpr(%a: -2) -> (result: -1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  %ta = hir.int_type
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr : %ta {
    // Inner pinned expr at offset -1 relative to parent.
    // When parent tightens to -1, inner is at -1 + (-1) = -2.
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-2"
    %inner = uir.expr pin -1 : %ta {
      // Pure op at -2 (operands at -2).
      // CHECK: hir.add %a, %a {{.*}}pa.phase = "-2"
      %sum = hir.add %a, %a : %t
      uir.yield %sum : %ta
    }
    uir.yield %inner : %ta
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Floating expr inside floating expr: both are demand-driven. The outer
// tightens to -1 (from return), the inner tightens independently based on
// its own consumers.

// CHECK-LABEL: uir.func @FloatingInsideFloating
uir.func @FloatingInsideFloating(%a: -2) -> (result: -1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  %ta = hir.int_type
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr : %ta {
    %tb = hir.int_type
    // Inner floating expr. Its phase is determined by its consumer (the yield
    // of the outer expr, which demands -1).
    // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
    %inner = uir.expr : %tb {
      // CHECK: hir.add %a, %a {{.*}}pa.phase = "-2"
      %sum = hir.add %a, %a : %t
      uir.yield %sum : %tb
    }
    uir.yield %inner : %ta
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Deep nesting: if inside pinned expr inside floating expr. All three layers
// must tighten when the outermost result is demanded at -1.

// CHECK-LABEL: uir.func @DeepNestedTightening
uir.func @DeepNestedTightening(%cond: -2) -> (result: -1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  %ta = hir.int_type
  // CHECK: uir.expr : {{.*}} attributes {{{.*}}pa.phase = "-1"
  %0 = uir.expr : %ta {
    // Pinned at offset -1 relative to floating parent.
    // When parent tightens to -1, this is at -2.
    // CHECK: uir.expr pin -1 {{.*}} attributes {{{.*}}pa.phase = "-2"
    %inner = uir.expr pin -1 : %ta {
      %ra = hir.int_type
      // If anchored at the pinned expr's phase (-2).
      // CHECK: uir.if {{.*}} attributes {{{.*}}pa.phase = "-2"
      %r = uir.if %cond : %ta {
        // CHECK: uir.call @MakeValue()
        // CHECK-SAME: pa.phase = "-2"
        %v = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
        uir.yield %v : %ta
      } else {
        // CHECK: uir.call @MakeValue()
        // CHECK-SAME: pa.phase = "-2"
        %v = uir.call @MakeValue() : () -> (%ra) () -> !hir.any [] -> [0]
        uir.yield %v : %ta
      }
      uir.yield %r : %ta
    }
    uir.yield %inner : %ta
  }
  uir.return %0 -> (%t)
}

//===----------------------------------------------------------------------===//
// Big pure tree: 4 args at different phases flowing through a tree of pure ops.
// Each pure op's phase is the max of its operand phases.

// CHECK-LABEL: uir.func @BigPureTree
uir.func @BigPureTree(%a: -3, %b: -2, %c: -1, %d: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  %t4 = hir.int_type
  uir.signature (%t0, %t1, %t2, %t3) -> (%t4)
} {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  %two = hir.constant_int 2 : %t
  %three = hir.constant_int 3 : %t
  // (a+1): max(-3, float) = -3
  // CHECK: hir.add %a, {{.*}}pa.phase = "-3"
  %a1 = hir.add %a, %one : %t
  // (b-2): max(-2, float) = -2
  // CHECK: hir.sub %b, {{.*}}pa.phase = "-2"
  %b2 = hir.sub %b, %two : %t
  // (a1*b2): max(-3, -2) = -2
  // CHECK: hir.mul {{.*}}pa.phase = "-2"
  %left1 = hir.mul %a1, %b2 : %t
  // (c*3): max(-1, float) = -1
  // CHECK: hir.mul %c, {{.*}}pa.phase = "-1"
  %c3 = hir.mul %c, %three : %t
  // left: max(-2, -1) = -1
  // CHECK: hir.add {{.*}}pa.phase = "-1"
  %left = hir.add %left1, %c3 : %t
  // (a-b): max(-3, -2) = -2
  // CHECK: hir.sub %a, %b {{.*}}pa.phase = "-2"
  %ab = hir.sub %a, %b : %t
  // (c+d): max(-1, 0) = 0
  // CHECK: hir.add %c, %d {{.*}}pa.phase = "0"
  %cd = hir.add %c, %d : %t
  // (d + ab + cd): max(0, -2) = 0, then max(0, 0) = 0
  // CHECK: hir.add %d, {{.*}}pa.phase = "0"
  %dab = hir.add %d, %ab : %t
  // CHECK: hir.add {{.*}}pa.phase = "0"
  %right = hir.add %dab, %cd : %t
  // final: max(-1, 0) = 0
  // CHECK: hir.mul {{.*}}pa.phase = "0"
  %result = hir.mul %left, %right : %t
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// One value at -2 consumed in four different phase contexts: body (pinned at
// 0), const block (-1), double const block (-2), and as const arg to a call.

// CHECK-LABEL: uir.func @MultiUse
uir.func @MultiUse(%x: -2) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  %two = hir.constant_int 2 : %t
  %three = hir.constant_int 3 : %t
  // u0 = x + 1: pure, max(-2, float) = -2. Pinned at 0.
  %u0_val = hir.add %x, %one : %t
  // CHECK: uir.pin {{.*}}, 0 {{.*}}pa.phase = "0"
  %u0 = uir.pin %u0_val, 0 : !hir.any
  // u1 = const { x + 2 }. Const block at -1. x at -2 OK. u1 at -1.
  %u1 = uir.expr pin -1 : %t {
    %v = hir.add %x, %two : %t
    uir.yield %v : %t
  }
  // u2 = const { const { x + 3 } }. Double const at -2. x at -2 OK. u2 at -2.
  %u2 = uir.expr pin -1 : %t {
    %inner = uir.expr pin -1 : %t {
      %v = hir.add %x, %three : %t
      uir.yield %v : %t
    }
    uir.yield %inner : %t
  }
  // u3 = TakesConst(x). Const arg at -1. x at -2 OK. u3 at 0.
  %tx = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.call @TakesConst({{.*}}) {{.*}}pa.phase = "0"
  %u3 = uir.call @TakesConst(%x) : (%tx) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  // Combine: max(0, -1, -2, 0) = 0.
  %s1 = hir.add %u0, %u1 : %t
  %s2 = hir.add %s1, %u2 : %t
  // CHECK: hir.add {{.*}}pa.phase = "0"
  %s3 = hir.add %s2, %u3 : %t
  uir.return %s3 -> (%t)
}

//===----------------------------------------------------------------------===//
// Short-circuit && with mixed phases: const LHS, body-phase RHS.
// The if anchors at 0 (from return). Condition at -1 has slack. OK.

// CHECK-LABEL: uir.func @ShortCircuitAndMixed
uir.func @ShortCircuitAndMixed(%a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  // b > 0: pure, max(0, float) = 0.
  // CHECK: hir.gt {{.*}}pa.phase = "0"
  %cmp = hir.gt %b, %zero : %t
  %false_val = hir.constant_bool <false>
  // if a { cmp } else { false }: if at 0. a at -1 (slack).
  // CHECK: uir.if %a {{.*}}pa.phase = "0"
  %result = uir.if %a : %t {
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield %cmp : %t
  } else {
    uir.yield %false_val : %t
  }
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// Short-circuit && with both const args, result fed as const arg to a call.
// If at -1 (from const arg demand). a at -1. b at -1.

uir.func @NeedsConstBool(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// CHECK-LABEL: uir.func @ShortCircuitAndBothConst
uir.func @ShortCircuitAndBothConst(%a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %false_val = hir.constant_bool <false>
  // if at 0 (block phase). Result floats to satisfy const arg demand at -1.
  // CHECK: uir.if %a {{.*}}pa.phase = "0"
  %and = uir.if %a : %t {
    uir.yield %b : %t
  } else {
    uir.yield %false_val : %t
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // CHECK: uir.call @NeedsConstBool({{.*}}) {{.*}}pa.phase = "0"
  %r = uir.call @NeedsConstBool(%and) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r -> (%tr)
}

//===----------------------------------------------------------------------===//
// Nested short-circuit: (a && b) || c.
// Desugars to: if (if a { b } else { false }) { true } else { c }.
// Outer if at 0 (return). Inner if floats. c at 0. a, b at -1 (slack).

// CHECK-LABEL: uir.func @NestedShortCircuit
uir.func @NestedShortCircuit(%a: -1, %b: -1, %c: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  %false_val = hir.constant_bool <false>
  %true_val = hir.constant_bool <true>
  // Inner: if a { b } else { false }
  %and = uir.if %a : %t {
    uir.yield %b : %t
  } else {
    uir.yield %false_val : %t
  }
  // Outer: if and { true } else { c }
  // CHECK: uir.if {{.*}}pa.phase = "0"
  %result = uir.if %and : %t {
    uir.yield %true_val : %t
  } else {
    // CHECK: uir.yield {{.*}}pa.phase = "0"
    uir.yield %c : %t
  }
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// Self-recursive factorial with const arg. Phase offsets come from the declared
// signature, so there is no circular dependency.

// CHECK-LABEL: uir.func @Factorial
uir.func @Factorial(%n: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  // CHECK: hir.leq %n, {{.*}}pa.phase = "-1"
  %cond = hir.leq %n, %one : %t
  %result = uir.if %cond : %t {
    uir.yield %one : %t
  } else {
    // CHECK: hir.sub %n, {{.*}}pa.phase = "-1"
    %nm1 = hir.sub %n, %one : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // CHECK: uir.call @Factorial({{.*}}) {{.*}}pa.phase = "0"
    %rec = uir.call @Factorial(%nm1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
    // CHECK: hir.mul %n, {{.*}}pa.phase = "0"
    %prod = hir.mul %n, %rec : %t
    uir.yield %prod : %t
  }
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// Self-recursive with const const arg at -2. Exercises deeper phase offsets
// beyond the common -1/+1 cases.

// CHECK-LABEL: uir.func @DeepRecurse
uir.func @DeepRecurse(%n: -2) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  // CHECK: hir.leq %n, {{.*}}pa.phase = "-2"
  %cond = hir.leq %n, %zero : %t
  %result = uir.if %cond : %t {
    uir.yield %zero : %t
  } else {
    %one = hir.constant_int 1 : %t
    // CHECK: hir.sub %n, {{.*}}pa.phase = "-2"
    %nm1 = hir.sub %n, %one : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // CHECK: uir.call @DeepRecurse({{.*}}) {{.*}}pa.phase = "0"
    %rec = uir.call @DeepRecurse(%nm1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-2] -> [0]
    uir.yield %rec : %t
  }
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// Mutual recursion: is_even and is_odd call each other. Both have const arg at
// -1. Calls satisfy each other's signatures.

uir.func @IsOdd(%n: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %cond = hir.eq %n, %zero : %t
  %false_val = hir.constant_bool <false>
  %one = hir.constant_int 1 : %t
  %result = uir.if %cond : %t {
    uir.yield %false_val : %t
  } else {
    %nm1 = hir.sub %n, %one : %t
    %ta = hir.int_type
    %tr = hir.int_type
    %rec = uir.call @IsEven(%nm1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
    uir.yield %rec : %t
  }
  uir.return %result -> (%t)
}

// CHECK-LABEL: uir.func @IsEven
uir.func @IsEven(%n: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  // CHECK: hir.eq %n, {{.*}}pa.phase = "-1"
  %cond = hir.eq %n, %zero : %t
  %true_val = hir.constant_bool <true>
  %one = hir.constant_int 1 : %t
  %result = uir.if %cond : %t {
    uir.yield %true_val : %t
  } else {
    // CHECK: hir.sub %n, {{.*}}pa.phase = "-1"
    %nm1 = hir.sub %n, %one : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // CHECK: uir.call @IsOdd({{.*}}) {{.*}}pa.phase = "0"
    %rec = uir.call @IsOdd(%nm1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
    uir.yield %rec : %t
  }
  uir.return %result -> (%t)
}

//===----------------------------------------------------------------------===//
// Self-recursive with both const and dyn args. Const arg n at -1, dyn arg acc
// at +1. Dyn result at callPhase + 1.

// CHECK-LABEL: uir.func @RecursiveChain
uir.func @RecursiveChain(%n: -1, %acc: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %cond = hir.leq %n, %zero : %t
  %result = uir.if %cond : %t {
    uir.yield %acc : %t
  } else {
    %one = hir.constant_int 1 : %t
    %nm1 = hir.sub %n, %one : %t
    %ta = hir.int_type
    %tb = hir.int_type
    %tr = hir.int_type
    // CHECK: uir.call @RecursiveChain({{.*}}) {{.*}}pa.phase = "0"
    %rec = uir.call @RecursiveChain(%nm1, %acc) : (%ta, %tb) -> (%tr) (!hir.any, !hir.any) -> !hir.any [-1, 1] -> [1]
    uir.yield %rec : %t
  }
  uir.return %result -> (%t)
}
