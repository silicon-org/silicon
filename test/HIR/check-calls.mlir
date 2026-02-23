// RUN: silicon-opt --check-calls %s | FileCheck %s

// CHECK-LABEL: hir.unified_func @Empty
hir.unified_func @Empty [] -> [] {
  // CHECK: hir.unified_signature () -> ()
  hir.unified_signature () -> ()
} {
  // CHECK: hir.unified_return
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @SimpleFoo
hir.unified_func @SimpleFoo [] -> [] {
  hir.unified_signature () -> ()
} {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[TMP_TY:%.+]] = hir.type_of [[TMP]]
  // CHECK: [[UNI_TY:%.+]] = hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: hir.checked_call @SimpleBar([[TMP]]) : ([[UNI_TY]]) -> ([[RET_TY]]) -> (!hir.any) [0] [0]
  %1 = hir.unified_call @SimpleBar(%0) : (!hir.any) -> (!hir.any) [0] -> [0]
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @SimpleBar
hir.unified_func @SimpleBar [0] -> [0] {
  %0 = hir.int_type {a}
  %1 = hir.unified_arg "a", %0
  %2 = hir.int_type {b}
  hir.unified_signature (%1) -> (%2)
} {
^bb0(%arg0: !hir.any):
  hir.unified_return %arg0
}

// CHECK-LABEL: hir.unified_func @NestedFoo
hir.unified_func @NestedFoo [] -> [] {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[TMP_TY:%.+]] = hir.type_of [[TMP]]
  // CHECK: [[UNI_TY:%.+]] = hir.unify [[ARG_TY]], [[TMP_TY]]
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: hir.checked_call @NestedBar([[TMP]]) : ([[UNI_TY]]) -> ([[RET_TY]]) -> (!hir.any) [0] [0]
  %1 = hir.unified_call @NestedBar(%0) : (!hir.any) -> (!hir.any) [0] -> [0]
  hir.unified_signature () -> ()
} {
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @NestedBar
hir.unified_func @NestedBar [0] -> [0] {
  %0 = hir.int_type {a}
  %1 = hir.unified_arg "a", %0
  %2 = hir.int_type {b}
  hir.unified_signature (%1) -> (%2)
} {
^bb0(%arg0: !hir.any):
  hir.unified_return %arg0
}
