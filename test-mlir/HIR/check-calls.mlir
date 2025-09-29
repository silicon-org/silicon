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
  // CHECK: hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.value
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[RET:%.+]] = hir.unchecked_call @SimpleBar
  %1 = hir.unchecked_call @SimpleBar(%0) : (!hir.value) -> (!hir.value)
  // CHECK: [[RET_COERCED:%.+]] = hir.coerce_type [[RET]], [[RET_TY]]
  hir.unchecked_return
}

// CHECK-LABEL: hir.unchecked_func @SimpleBar
hir.unchecked_func @SimpleBar {
  %0 = hir.int_type {a}
  %1 = hir.unchecked_arg "a", %0 : !hir.type
  %2 = hir.int_type {b}
  hir.unchecked_signature (%1 : !hir.value) -> (%2 : !hir.type)
} {
^bb0(%arg0: !hir.value):
  hir.unchecked_return %arg0 : !hir.value
}

// CHECK-LABEL: hir.unchecked_func @NestedFoo
hir.unchecked_func @NestedFoo {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[TMP_TY:%.+]] = hir.type_of [[TMP]]
  // CHECK: hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.value
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[RET:%.+]] = hir.unchecked_call @NestedBar
  %1 = hir.unchecked_call @NestedBar(%0) : (!hir.value) -> (!hir.value)
  // CHECK: [[RET_COERCED:%.+]] = hir.coerce_type [[RET]], [[RET_TY]]
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}

// CHECK-LABEL: hir.unchecked_func @NestedBar
hir.unchecked_func @NestedBar {
  %0 = hir.int_type {a}
  %1 = hir.unchecked_arg "a", %0 : !hir.type
  %2 = hir.int_type {b}
  hir.unchecked_signature (%1 : !hir.value) -> (%2 : !hir.type)
} {
^bb0(%arg0: !hir.value):
  hir.unchecked_return %arg0 : !hir.value
}
