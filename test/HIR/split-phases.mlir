// RUN: silicon-opt --split-phases %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @SinglePhase.const0
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
hir.unified_func @SinglePhase [] -> [] attributes {argNames = []} {
  hir.unified_signature () -> ()
} {
  func.call @dummyA() : () -> ()
  hir.unified_return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.const1
// CHECK-NEXT: hir.expr
// CHECK-NEXT:   func.call @dummyA
// CHECK-NEXT:   hir.yield
// CHECK-NEXT: }
// CHECK-NEXT: hir.return

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.const0
// CHECK-NEXT: func.call @dummyB
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
hir.unified_func @TwoUnrelatedPhases [] -> [] attributes {argNames = []} {
  hir.unified_signature () -> ()
} {
  func.call @dummyB() : () -> ()
  hir.expr attributes {const = -1} {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.unified_return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const1
// CHECK-NEXT: [[C42:%.+]] = hir.constant_int 42
// CHECK-NEXT: [[TMP:%.+]] = hir.expr
// CHECK-NEXT:   hir.constant_int 1337
// CHECK-NEXT:   hir.yield
// CHECK-NEXT: }
// CHECK-NEXT: hir.binary [[C42]], [[TMP]]
// CHECK-NEXT: hir.return

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const0
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
hir.unified_func @ValueUseAcrossPhases [] -> [] attributes {argNames = []} {
  hir.unified_signature () -> ()
} {
  %0 = hir.constant_int 42
  %1 = hir.expr : !hir.any attributes {const = -1} {
    %3 = hir.constant_int 1337
    hir.yield %3 : !hir.any
  }
  %2 = hir.binary %0, %1
  hir.unified_return
}

//===----------------------------------------------------------------------===//
// Constness-aware split: a const argument flows from the const phase to the
// runtime phase.

// CHECK-LABEL: hir.func private @ConstArg.const1
// CHECK-NEXT: ^bb0([[A:%.+]]: !hir.any):
// CHECK-NEXT: hir.return [[A]]

// CHECK-LABEL: hir.func private @ConstArg.const0
// CHECK-NEXT: ^bb0([[B:%.+]]: !hir.any, [[A:%.+]]: !hir.any):
// CHECK-NEXT: [[R:%.+]] = hir.binary [[A]], [[B]]
// CHECK-NEXT: hir.return [[R]]

// CHECK-NOT: hir.unified_func
hir.unified_func @ConstArg [-1, 0] -> [0] attributes {argNames = ["a", "b"]} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  hir.unified_signature (%0, %1) -> (%2)
} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.binary %a, %b
  hir.unified_return %0
}

//===----------------------------------------------------------------------===//
// Three-phase split: const const arg at phase -2, const arg at phase -1,
// runtime arg at phase 0. Values thread through adjacent phases.

// CHECK-LABEL: hir.func private @ThreePhase.const2
// CHECK-NEXT: ^bb0([[A:%.+]]: !hir.any):
// CHECK-NEXT: hir.return [[A]]

// CHECK-LABEL: hir.func private @ThreePhase.const1
// CHECK-NEXT: ^bb0([[B:%.+]]: !hir.any, [[A:%.+]]: !hir.any):
// CHECK-NEXT: [[TMP:%.+]] = hir.binary [[A]], [[B]]
// CHECK-NEXT: hir.return [[TMP]]

// CHECK-LABEL: hir.func private @ThreePhase.const0
// CHECK-NEXT: ^bb0([[C:%.+]]: !hir.any, [[AB:%.+]]: !hir.any):
// CHECK-NEXT: [[RES:%.+]] = hir.binary [[AB]], [[C]]
// CHECK-NEXT: hir.return [[RES]]

// CHECK-NOT: hir.unified_func
hir.unified_func @ThreePhase [-2, -1, 0] -> [0] attributes {argNames = ["a", "b", "c"]} {
^bb0(%a: !hir.any, %b: !hir.any, %c: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
^bb0(%a: !hir.any, %b: !hir.any, %c: !hir.any):
  %0 = hir.binary %a, %b
  %1 = hir.binary %0, %c
  hir.unified_return %1
}

//===----------------------------------------------------------------------===//
// Three-phase call rewriting: a unified_call to a 3-phase function gets
// rewritten into 3 chained hir.call ops. The phase -2 and -1 arguments are
// constants (available at any phase), and the phase 0 argument is a func arg.

// CHECK-LABEL: hir.func private @ThreePhaseCaller.const0
// CHECK: hir.call @ThreePhase.const2(
// CHECK: hir.call @ThreePhase.const1(
// CHECK: hir.call @ThreePhase.const0(
// CHECK: hir.return

// CHECK-NOT: hir.unified_func
hir.unified_func @ThreePhaseCaller [0] -> [0] attributes {argNames = ["z"]} {
^bb0(%z: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
^bb0(%z: !hir.any):
  %a = hir.constant_int 10
  %b = hir.constant_int 20
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %t3 = hir.inferrable
  %r = hir.unified_call @ThreePhase(%a, %b, %z) : (%t0, %t1, %t2) -> (%t3) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  hir.unified_return %r
}
