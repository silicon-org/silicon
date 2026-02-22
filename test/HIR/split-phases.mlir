// RUN: silicon-opt --split-phases %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @SinglePhase.const0
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unchecked_func
hir.unchecked_func @SinglePhase {
  hir.unchecked_signature () -> ()
} {
  func.call @dummyA() : () -> ()
  hir.unchecked_return
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

// CHECK-NOT: hir.unchecked_func
hir.unchecked_func @TwoUnrelatedPhases {
  hir.unchecked_signature () -> ()
} {
  func.call @dummyB() : () -> ()
  hir.expr attributes {const = 1} {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.unchecked_return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const1
// CHECK-NEXT: [[TMP:%.+]] = hir.expr
// CHECK-NEXT:   hir.constant_int 1337
// CHECK-NEXT:   hir.yield
// CHECK-NEXT: }
// CHECK-NEXT: hir.return [[TMP]]

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const0
// CHECK-NEXT: ^bb0([[TMP1:%.+]]: !hir.any):
// CHECK-NEXT: [[TMP0:%.+]] = hir.constant_int 42
// CHECK-NEXT: hir.binary [[TMP0]], [[TMP1]]
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unchecked_func
hir.unchecked_func @ValueUseAcrossPhases {
  hir.unchecked_signature () -> ()
} {
  %0 = hir.constant_int 42
  %1 = hir.expr : !hir.any attributes {const = 1} {
    %3 = hir.constant_int 1337
    hir.yield %3 : !hir.any
  }
  %2 = hir.binary %0, %1
  hir.unchecked_return
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

// CHECK-NOT: hir.unchecked_func
hir.unchecked_func @ConstArg {
  %0 = hir.int_type
  %1 = hir.unchecked_arg "a", %0, 1
  %2 = hir.int_type
  %3 = hir.unchecked_arg "b", %2, 0
  %4 = hir.int_type
  hir.unchecked_signature (%1, %3) -> (%4)
} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.binary %a, %b
  hir.unchecked_return %0
}
