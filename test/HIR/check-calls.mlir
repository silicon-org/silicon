// RUN: silicon-opt --check-calls %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Multi-block signature: the entry block computes the type in a separate block
// and branches to the exit block where the unified_signature terminator lives.

// CHECK-LABEL: hir.unified_func private @MultiBlockSig
hir.unified_func private @MultiBlockSig(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.const_br ^exit(%0, %0 : !hir.any, !hir.any)
^exit(%arg_ty: !hir.any, %ret_ty: !hir.any):
  hir.unified_signature (%arg_ty) -> (%ret_ty)
} {
  // The body gets the cloned signature blocks as a preamble. The entry block
  // computes the type and branches to an intermediate block, which forwards
  // the type values as block arguments to the body block.
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: hir.const_br ^bb1([[T]], [[T]] : !hir.any, !hir.any)
  // CHECK: ^bb1({{%.+}}: !hir.any, {{%.+}}: !hir.any):
  // CHECK: hir.const_br ^bb2({{%.+}}, {{%.+}} : !hir.any, !hir.any)
  // CHECK: ^bb2([[ARG_TY:%.+]]: !hir.any, [[RET_TY:%.+]]: !hir.any):
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.unified_return [[COERCED]] : ([[ARG_TY]]) -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.unified_return %a : () -> (%t0)
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.unified_func @Empty
hir.unified_func @Empty() -> () {
  // CHECK: hir.unified_signature () -> ()
  hir.unified_signature () -> ()
} {
  // CHECK: hir.unified_return : () -> (){{$}}
  hir.unified_return : () -> ()
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
  hir.unified_return : () -> ()
}

// CHECK-LABEL: hir.unified_func @SimpleBar
hir.unified_func @SimpleBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.unified_signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.unified_return [[COERCED]] : ([[ARG_TY]]) -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.unified_return %a : () -> (%t0)
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
  hir.unified_return : () -> ()
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
  // CHECK: hir.unified_return [[CX]] : ([[TT]], %T) -> ([[UNI]])
  hir.unified_return %x : () -> (%T)
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
  hir.unified_return : () -> ()
}

// CHECK-LABEL: hir.unified_func @NestedBar
hir.unified_func @NestedBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.unified_signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.unified_return [[COERCED]] : ([[ARG_TY]]) -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.unified_return %a : () -> (%t0)
}

//===----------------------------------------------------------------------===//
// Parameterized type in signature: `uint_type` depends on `constant_int`.
// The body should get its own cloned copies of both ops transitively.

// CHECK-LABEL: hir.unified_func private @UIntBody
hir.unified_func private @UIntBody(%a: 0) -> (result: 0) {
  %0 = hir.constant_int 42
  %1 = hir.uint_type %0
  %2 = hir.unit_type
  hir.unified_signature (%1) -> (%2)
} {
  // CHECK:       {
  // CHECK-DAG:     [[W:%.+]] = hir.constant_int 42
  // CHECK-DAG:     [[T:%.+]] = hir.uint_type [[W]]
  // CHECK:         hir.coerce_type %a, [[T]]
  %0 = hir.constant_unit
  %1 = hir.unit_type
  hir.unified_return %0 : () -> (%1)
}
