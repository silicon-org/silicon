// RUN: silicon-opt --check-calls %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Multi-block signature: the entry block computes the type in a separate block
// and branches to the exit block where the signature terminator lives.

// CHECK-LABEL: hir.unified_func private @MultiBlockSig
hir.unified_func private @MultiBlockSig(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  cf.br ^exit(%0, %0 : !hir.any, !hir.any)
^exit(%arg_ty: !hir.any, %ret_ty: !hir.any):
  hir.signature (%arg_ty) -> (%ret_ty)
} {
  // The body gets the cloned signature blocks as a preamble. The empty exit
  // block is eliminated, so the entry branches directly to the body block.
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: cf.br ^bb1([[T]], [[T]] : !hir.any, !hir.any)
  // CHECK: ^bb1([[ARG_TY:%.+]]: !hir.any, [[RET_TY:%.+]]: !hir.any):
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.return [[COERCED]] -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.return %a -> (%t0)
}

//===----------------------------------------------------------------------===//
// Calling a function with a 2-block signature. The signature blocks are
// inlined ahead of the call, with the empty exit block eliminated.

// CHECK-LABEL: hir.unified_func @CallMultiBlockSig
hir.unified_func @CallMultiBlockSig() -> () {
  hir.signature () -> ()
} {
  // The 2-block signature is inlined ahead of the call. The empty exit block
  // is eliminated, so the entry branches directly to the continuation block.
  // CHECK: [[VAL:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: cf.br ^bb1([[T]], [[T]] : !hir.any, !hir.any)
  // CHECK: ^bb1([[ARG_TY:%.+]]: !hir.any, [[RET_TY:%.+]]: !hir.any):
  // CHECK: hir.unified_call @MultiBlockSig([[VAL]]) : ([[ARG_TY]]) -> ([[RET_TY]])
  %0 = builtin.unrealized_conversion_cast to !hir.any
  %infer0 = hir.inferrable
  %infer1 = hir.inferrable
  %r = hir.unified_call @MultiBlockSig(%0) : (%infer0) -> (%infer1) (!hir.any) -> !hir.any [0] -> [0]
  hir.return -> ()
}

//===----------------------------------------------------------------------===//
// 3-block signature with multiple terminators. After consolidation, the
// exit block is eliminated by redirecting predecessor branches directly to
// the body block.

// CHECK-LABEL: hir.unified_func private @MultiTermSig
hir.unified_func private @MultiTermSig(%a: 0) -> (result: 0) {
  %cond = builtin.unrealized_conversion_cast to i1
  %t1 = hir.int_type {left}
  %t2 = hir.int_type {right}
  cf.cond_br %cond,
    ^left(%t1 : !hir.any),
    ^right(%t2 : !hir.any)
^left(%lt: !hir.any):
  hir.signature (%lt) -> (%lt)
^right(%rt: !hir.any):
  hir.signature (%rt) -> (%rt)
} {
  // After consolidation, both terminators are replaced with branches to a
  // consolidated exit block. The empty exit block is then eliminated, so
  // ^left and ^right branch directly to the body block.
  // CHECK: cf.cond_br {{%.+}}, ^bb1({{%.+}} : !hir.any), ^bb2({{%.+}} : !hir.any)
  // CHECK: ^bb1([[LT:%.+]]: !hir.any):
  // CHECK: cf.br ^bb3([[LT]], [[LT]] : !hir.any, !hir.any)
  // CHECK: ^bb2([[RT:%.+]]: !hir.any):
  // CHECK: cf.br ^bb3([[RT]], [[RT]] : !hir.any, !hir.any)
  // CHECK: ^bb3([[ARG_TY:%.+]]: !hir.any, [[RET_TY:%.+]]: !hir.any):
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.return [[COERCED]] -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.return %a -> (%t0)
}

// CHECK-LABEL: hir.unified_func @CallMultiTermSig
hir.unified_func @CallMultiTermSig() -> () {
  hir.signature () -> ()
} {
  // The 3-block signature with 2 terminators is consolidated and inlined.
  // The consolidated exit block is eliminated.
  // CHECK: [[VAL:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: cf.cond_br {{%.+}}, ^bb1({{%.+}} : !hir.any), ^bb2({{%.+}} : !hir.any)
  // CHECK: ^bb1([[LT:%.+]]: !hir.any):
  // CHECK: cf.br ^bb3([[LT]], [[LT]] : !hir.any, !hir.any)
  // CHECK: ^bb2([[RT:%.+]]: !hir.any):
  // CHECK: cf.br ^bb3([[RT]], [[RT]] : !hir.any, !hir.any)
  // CHECK: ^bb3([[ARG_TY:%.+]]: !hir.any, [[RET_TY:%.+]]: !hir.any):
  // CHECK: hir.unified_call @MultiTermSig([[VAL]]) : ([[ARG_TY]]) -> ([[RET_TY]])
  %0 = builtin.unrealized_conversion_cast to !hir.any
  %infer0 = hir.inferrable
  %infer1 = hir.inferrable
  %r = hir.unified_call @MultiTermSig(%0) : (%infer0) -> (%infer1) (!hir.any) -> !hir.any [0] -> [0]
  hir.return -> ()
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.unified_func @Empty
hir.unified_func @Empty() -> () {
  // CHECK: hir.signature () -> ()
  hir.signature () -> ()
} {
  // CHECK: hir.return -> (){{$}}
  hir.return -> ()
}

// CHECK-LABEL: hir.unified_func @SimpleFoo
hir.unified_func @SimpleFoo() -> () {
  hir.signature () -> ()
} {
  // CHECK: [[TMP:%.+]] = builtin.unrealized_conversion_cast
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  %0 = builtin.unrealized_conversion_cast to !hir.any
  // CHECK: hir.unified_call @SimpleBar([[TMP]]) : ([[ARG_TY]]) -> ([[RET_TY]]) (!hir.any) -> !hir.any [0] -> [0]
  %infer0 = hir.inferrable
  %infer1 = hir.inferrable
  %1 = hir.unified_call @SimpleBar(%0) : (%infer0) -> (%infer1) (!hir.any) -> !hir.any [0] -> [0]
  hir.return -> ()
}

// CHECK-LABEL: hir.unified_func @SimpleBar
hir.unified_func @SimpleBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.return [[COERCED]] -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.return %a -> (%t0)
}

//===----------------------------------------------------------------------===//
// Dependent types: result type depends on argument value.

// CHECK-LABEL: hir.unified_func @DepTypeCaller
hir.unified_func @DepTypeCaller() -> () {
  hir.signature () -> ()
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
  hir.return -> ()
}

// CHECK-LABEL: hir.unified_func @Identity
hir.unified_func @Identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.signature (%type_type, %T) -> (%T)
} {
  // CHECK: [[TT:%.+]] = hir.type_type
  // CHECK: [[CT:%.+]] = hir.coerce_type %T, [[TT]]
  // CHECK: [[CX:%.+]] = hir.coerce_type %x, %T
  // CHECK: [[UNI:%.+]] = hir.unify [[CT]], %T
  // CHECK: hir.return [[CX]] -> ([[UNI]])
  hir.return %x -> (%T)
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
  hir.signature () -> ()
} {
  hir.return -> ()
}

// CHECK-LABEL: hir.unified_func @NestedBar
hir.unified_func @NestedBar(%a: 0) -> (result: 0) {
  %0 = hir.int_type {a}
  %1 = hir.int_type {b}
  hir.signature (%0) -> (%1)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type {a}
  // CHECK: [[RET_TY:%.+]] = hir.int_type {b}
  // CHECK: [[COERCED:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[TYPEOF:%.+]] = hir.type_of [[COERCED]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TYPEOF]], [[RET_TY]]
  // CHECK: hir.return [[COERCED]] -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.return %a -> (%t0)
}

//===----------------------------------------------------------------------===//
// Parameterized type in signature: `uint_type` depends on `constant_int`.
// The body should get its own cloned copies of both ops transitively.

// CHECK-LABEL: hir.unified_func private @UIntBody
hir.unified_func private @UIntBody(%a: 0) -> (result: 0) {
  %tc0 = hir.int_type
  %0 = hir.constant_int 42 : %tc0
  %1 = hir.uint_type %0
  %2 = hir.unit_type
  hir.signature (%1) -> (%2)
} {
  // CHECK:       {
  // CHECK-DAG:     [[W:%.+]] = hir.constant_int 42 :
  // CHECK-DAG:     [[T:%.+]] = hir.uint_type [[W]]
  // CHECK:         hir.coerce_type %a, [[T]]
  %0 = hir.constant_unit
  %1 = hir.unit_type
  hir.return %0 -> (%1)
}

//===----------------------------------------------------------------------===//
// Duplicate calls: calling the same function twice with the same arguments.
// This tests that return type unification works correctly when a callee's
// signature is inlined multiple times. Previously, the hir.unify ops in return
// type positions were not recognized as resolvable types during HIR-to-MIR
// lowering.

// CHECK-LABEL: hir.unified_func private @DupCallee
hir.unified_func private @DupCallee(%a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[ARG_TY:%.+]] = hir.int_type
  // CHECK: [[CA:%.+]] = hir.coerce_type %a, [[ARG_TY]]
  // CHECK: [[CB:%.+]] = hir.coerce_type %b, [[ARG_TY]]
  // CHECK: [[TA:%.+]] = hir.type_of [[CA]]
  // CHECK: [[UNI:%.+]] = hir.unify [[TA]], [[ARG_TY]]
  // CHECK: hir.return [[CA]] -> ([[UNI]])
  %t0 = hir.type_of %a
  hir.return %a -> (%t0)
}

// CHECK-LABEL: hir.unified_func @DupCaller
hir.unified_func @DupCaller() -> () {
  hir.signature () -> ()
} {
  %v0 = builtin.unrealized_conversion_cast to !hir.any
  %v1 = builtin.unrealized_conversion_cast to !hir.any
  // First call: signature inlined, inferrables replaced with declared types.
  // CHECK: [[INT1:%.+]] = hir.int_type
  // CHECK: hir.unified_call @DupCallee({{.*}}) : ([[INT1]], [[INT1]]) -> ([[INT1]])
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r0 = hir.unified_call @DupCallee(%v0, %v1) : (%i0, %i1) -> (%i2) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  // Second call: separate inlined copy, also with unified return type.
  // CHECK: [[INT2:%.+]] = hir.int_type
  // CHECK: hir.unified_call @DupCallee({{.*}}) : ([[INT2]], [[INT2]]) -> ([[INT2]])
  %i3 = hir.inferrable
  %i4 = hir.inferrable
  %i5 = hir.inferrable
  %r1 = hir.unified_call @DupCallee(%v0, %v1) : (%i3, %i4) -> (%i5) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  hir.return -> ()
}
