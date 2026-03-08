// RUN: silicon-opt --test-phase-analysis %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//
// Single-phase function: all side-effecting ops inherit phase 0.

// CHECK-LABEL: hir.unified_func @SinglePhase
hir.unified_func @SinglePhase() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: func.call @dummyA() {phase = 0 : si16}
  func.call @dummyA() : () -> ()
  // CHECK: hir.return : () -> () {phase = "float"}
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Const argument: type_of on a const arg gets the arg's phase (-1); type_of on
// a runtime arg gets phase 0; unify of the two gets max(0, -1) = 0.

// CHECK-LABEL: hir.unified_func @ConstArg
hir.unified_func @ConstArg(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  hir.unified_signature (%0, %1) -> (%2)
} {
  // CHECK: hir.type_of %a {phase = -1 : si16}
  %ta = hir.type_of %a
  // CHECK: hir.type_of %b {phase = 0 : si16}
  %tb = hir.type_of %b
  // CHECK: hir.unify {{.*}} {phase = 0 : si16}
  %t = hir.unify %ta, %tb
  // CHECK: hir.add {{.*}} {phase = 0 : si16}
  %0 = hir.add %a, %b : %t
  // CHECK: hir.type_of {{.*}} {phase = 0 : si16}
  %t0 = hir.type_of %0
  // CHECK: hir.return {{.*}} : () -> ({{.*}}) {phase = 0 : si16}
  hir.return %0 : () -> (%t0)
}

//===----------------------------------------------------------------------===//
// ExprOp phaseShift: an ExprOp with phaseShift -1 shifts its contents to
// phase -1. Side-effecting ops inside the ExprOp inherit that shifted phase.

// CHECK-LABEL: hir.unified_func @ExprPhaseShift
hir.unified_func @ExprPhaseShift() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: func.call @dummyA() {phase = 0 : si16}
  func.call @dummyA() : () -> ()
  // CHECK: hir.expr -1 attributes {phase = -1 : si16}
  hir.expr -1 {
    // CHECK: func.call @dummyB() {phase = -1 : si16}
    func.call @dummyB() : () -> ()
    // CHECK: hir.yield {phase = -1 : si16}
    hir.yield
  }
  // CHECK: hir.return : () -> () {phase = "float"}
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Pure ops with no operands float (INT16_MIN). Pure ops with operands inherit
// the max phase of their operands.

// CHECK-LABEL: hir.unified_func @PureFloating
hir.unified_func @PureFloating() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: hir.constant_int 42 {phase = "float"}
  %0 = hir.constant_int 42
  // CHECK: hir.constant_int 100 {phase = "float"}
  %1 = hir.constant_int 100
  // CHECK: hir.type_of {{.*}} {phase = "float"}
  %t0 = hir.type_of %0
  // CHECK: hir.type_of {{.*}} {phase = "float"}
  %t1 = hir.type_of %1
  // CHECK: hir.unify {{.*}} {phase = "float"}
  %t = hir.unify %t0, %t1
  // CHECK: hir.add {{.*}} {phase = "float"}
  %2 = hir.add %0, %1 : %t
  // CHECK: hir.return : () -> () {phase = "float"}
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Phase pulling via call arg requirements: a unified_call requires a const arg
// at an earlier phase; the ExprOp producing the argument value is pulled back.

hir.unified_func @Adder(%a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t = hir.unify %ta, %tb
  %r = hir.add %a, %b : %t
  hir.return %r : () -> (%t)
}

hir.unified_func @CallerConstArg(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t = hir.unify %ta, %tb
  %0 = hir.add %a, %b : %t
  %t0 = hir.type_of %0
  hir.return %0 : () -> (%t0)
}

// The ExprOp wrapping the call to @Adder should be pulled from phase 0 to -1
// since it is passed as the const arg of @CallerConstArg.

// CHECK-LABEL: hir.unified_func @PullExpr
hir.unified_func @PullExpr(%y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  // CHECK: hir.constant_int 19 {phase = "float"}
  %c19 = hir.constant_int 19
  // CHECK: hir.constant_int 23 {phase = "float"}
  %c23 = hir.constant_int 23
  // CHECK: hir.expr 0 {{.*}}attributes {phase = -1 : si16}
  %key = hir.expr 0 : !hir.any {
    // CHECK: hir.unified_call @Adder({{.*}}{phase = -1 : si16}
    %t0 = hir.int_type
    %t1 = hir.int_type
    %ti = hir.inferrable
    %t = hir.unified_call @Adder(%c19, %c23) : (%t0, %t1) -> (%ti) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
    hir.yield %t : !hir.any
  }
  %kt = hir.type_of %key
  %yt = hir.type_of %y
  %rt = hir.inferrable
  // CHECK: hir.unified_call @CallerConstArg({{.*}}{phase = 0 : si16}
  %r = hir.unified_call @CallerConstArg(%key, %y) : (%kt, %yt) -> (%rt) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %rrt = hir.type_of %r
  hir.return %r : () -> (%rrt)
}

//===----------------------------------------------------------------------===//
// Side-effecting ops nested inside an ExprOp inherit the ExprOp's phase, not
// the function's phase.

// CHECK-LABEL: hir.unified_func @NestedSideEffects
hir.unified_func @NestedSideEffects() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: hir.expr -1 attributes {phase = -1 : si16}
  hir.expr -1 {
    // CHECK: func.call @dummyA() {phase = -1 : si16}
    func.call @dummyA() : () -> ()
    // CHECK: hir.yield {phase = -1 : si16}
    hir.yield
  }
  // CHECK: func.call @dummyB() {phase = 0 : si16}
  func.call @dummyB() : () -> ()
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Three-phase function: args at phases -2, -1, 0. Pure ops depending on args
// from different phases get the max of their operand phases.

// CHECK-LABEL: hir.unified_func @ThreePhase
hir.unified_func @ThreePhase(%a: -2, %b: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
  // CHECK: hir.type_of %a {phase = -2 : si16}
  %ta = hir.type_of %a
  // CHECK: hir.type_of %b {phase = -1 : si16}
  %tb = hir.type_of %b
  // CHECK: hir.unify {{.*}} {phase = -1 : si16}
  %t0 = hir.unify %ta, %tb
  // CHECK: hir.add %a, %b : {{.*}} {phase = -1 : si16}
  %0 = hir.add %a, %b : %t0
  // CHECK: hir.type_of {{.*}} {phase = -1 : si16}
  %t0b = hir.type_of %0
  // CHECK: hir.type_of %c {phase = 0 : si16}
  %tc = hir.type_of %c
  // CHECK: hir.unify {{.*}} {phase = 0 : si16}
  %t1 = hir.unify %t0b, %tc
  // CHECK: hir.add {{.*}}, %c : {{.*}} {phase = 0 : si16}
  %1 = hir.add %0, %c : %t1
  %t1b = hir.type_of %1
  hir.return %1 : () -> (%t1b)
}

//===----------------------------------------------------------------------===//
// Nested ExprOps: inner at -1 inside outer at 0. The inner ExprOp's nested ops
// should be at phase -1, outer's at phase 0.

// CHECK-LABEL: hir.unified_func @NestedExpr
hir.unified_func @NestedExpr() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: hir.expr 0 attributes {phase = 0 : si16}
  hir.expr 0 {
    // CHECK: hir.expr -1 attributes {phase = -1 : si16}
    hir.expr -1 {
      // CHECK: func.call @dummyA() {phase = -1 : si16}
      func.call @dummyA() : () -> ()
      hir.yield
    }
    // CHECK: func.call @dummyB() {phase = 0 : si16}
    func.call @dummyB() : () -> ()
    hir.yield
  }
  hir.return : () -> ()
}
