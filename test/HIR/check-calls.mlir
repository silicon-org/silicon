// RUN: silicon-opt --check-calls %s | FileCheck %s

// CHECK-LABEL: hir.unified_func @Empty
hir.unified_func @Empty() -> () {
  // CHECK: hir.unified_signature () -> ()
  hir.unified_signature () -> ()
} {
  // CHECK: hir.unified_return{{$}}
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @SimpleFoo
hir.unified_func @SimpleFoo() -> () {
  hir.unified_signature () -> ()
} {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: hir.unified_call @SimpleBar([[TMP]]) : ([[ARG_TY]]) -> ([[RET_TY]]) (!hir.any) -> !hir.any [0] -> [0]
  %infer0 = hir.inferrable
  %infer1 = hir.inferrable
  %1 = hir.unified_call @SimpleBar(%0) : (%infer0) -> (%infer1) (!hir.any) -> !hir.any [0] -> [0]
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @SimpleBar
hir.unified_func @SimpleBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.unified_signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.unified_return([[COERCED]]) : ([[UNI]])
  %t0 = hir.type_of %a
  hir.unified_return (%a) : (%t0)
}

//===----------------------------------------------------------------------===//
// Dependent types: result type depends on argument value.

// CHECK-LABEL: hir.unified_func @DepTypeCaller
hir.unified_func @DepTypeCaller() -> () {
  hir.unified_signature () -> ()
} {
  %int_type = hir.int_type
  %val = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: [[INT_TY:%.+]] = hir.int_type
  // CHECK: [[TYPE_TYPE:%.+]] = hir.type_type
  // CHECK: hir.unified_call @Identity([[INT_TY]], {{%.+}}) : ([[TYPE_TYPE]], [[INT_TY]]) -> ([[INT_TY]])
  %infer0 = hir.inferrable
  %infer1 = hir.inferrable
  %infer2 = hir.inferrable
  %r = hir.unified_call @Identity(%int_type, %val) : (%infer0, %infer1) -> (%infer2) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @Identity
hir.unified_func @Identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.unified_signature (%type_type, %T) -> (%T)
} {
  // CHECK: [[TT:%.+]] = hir.type_type
  // CHECK: [[CT:%.+]] = hir.coerce_type %T, [[TT]]
  // CHECK: [[CX:%.+]] = hir.coerce_type %x, %T
  // CHECK: [[UNI:%.+]] = hir.unify [[CT]], %T
  // CHECK: hir.unified_return([[CX]]) : ([[UNI]])
  hir.unified_return (%x) : (%T)
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.unified_func @NestedFoo
hir.unified_func @NestedFoo() -> () {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: hir.unified_call @NestedBar([[TMP]]) : ([[ARG_TY]]) -> ([[RET_TY]]) (!hir.any) -> !hir.any [0] -> [0]
  %infer2 = hir.inferrable
  %infer3 = hir.inferrable
  %1 = hir.unified_call @NestedBar(%0) : (%infer2) -> (%infer3) (!hir.any) -> !hir.any [0] -> [0]
  hir.unified_signature () -> ()
} {
  hir.unified_return
}

// CHECK-LABEL: hir.unified_func @NestedBar
hir.unified_func @NestedBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.unified_signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.unified_return([[COERCED]]) : ([[UNI]])
  %t0 = hir.type_of %a
  hir.unified_return (%a) : (%t0)
}
