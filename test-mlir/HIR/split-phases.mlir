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
// CHECK-NEXT: ^bb0([[TMP1:%.+]]: !hir.value):
// CHECK-NEXT: [[TMP0:%.+]] = hir.constant_int 42
// CHECK-NEXT: hir.binary [[TMP0]], [[TMP1]]
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unchecked_func
hir.unchecked_func @ValueUseAcrossPhases {
  hir.unchecked_signature () -> ()
} {
  %0 = hir.constant_int 42
  %1 = hir.expr : !hir.value attributes {const = 1} {
    %3 = hir.constant_int 1337
    hir.yield %3 : !hir.value
  }
  %2 = hir.binary %0, %1
  hir.unchecked_return
}
