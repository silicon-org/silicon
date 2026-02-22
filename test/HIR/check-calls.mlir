// RUN: silicon-opt --check-calls %s | FileCheck %s

// CHECK-LABEL: hir.unchecked_func @Empty
hir.unchecked_func @Empty {
  // CHECK: hir.unchecked_signature () -> ()
  hir.unchecked_signature () -> ()
} {
  // CHECK: hir.unchecked_return
  hir.unchecked_return
}

// CHECK-LABEL: hir.unchecked_func @SimpleFoo
hir.unchecked_func @SimpleFoo {
  hir.unchecked_signature () -> ()
} {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[TMP_TY:%.+]] = hir.type_of [[TMP]]
  // CHECK: [[UNI_TY:%.+]] = hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: hir.checked_call @SimpleBar([[TMP]]) : ([[UNI_TY]]) -> ([[RET_TY]]) -> (!hir.any) [0] [0]
  %1 = hir.unchecked_call @SimpleBar(%0) : (!hir.any) -> (!hir.any)
  hir.unchecked_return
}

// CHECK-LABEL: hir.unchecked_func @SimpleBar
hir.unchecked_func @SimpleBar {
  %0 = hir.int_type {a}
  %1 = hir.unchecked_arg "a", %0, 0
  %2 = hir.int_type {b}
  hir.unchecked_signature (%1) -> (%2)
} {
^bb0(%arg0: !hir.any):
  hir.unchecked_return %arg0
}

// CHECK-LABEL: hir.unchecked_func @NestedFoo
hir.unchecked_func @NestedFoo {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[TMP_TY:%.+]] = hir.type_of [[TMP]]
  // CHECK: [[UNI_TY:%.+]] = hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: hir.checked_call @NestedBar([[TMP]]) : ([[UNI_TY]]) -> ([[RET_TY]]) -> (!hir.any) [0] [0]
  %1 = hir.unchecked_call @NestedBar(%0) : (!hir.any) -> (!hir.any)
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}

// CHECK-LABEL: hir.unchecked_func @NestedBar
hir.unchecked_func @NestedBar {
  %0 = hir.int_type {a}
  %1 = hir.unchecked_arg "a", %0, 0
  %2 = hir.int_type {b}
  hir.unchecked_signature (%1) -> (%2)
} {
^bb0(%arg0: !hir.any):
  hir.unchecked_return %arg0
}
