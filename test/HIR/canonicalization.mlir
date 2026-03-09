// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @Unify
func.func @Unify() -> !hir.any {
  // CHECK: [[TY:%.+]] = hir.int_type
  %0 = hir.int_type
  %1 = hir.unify %0, %0
  // CHECK: return [[TY]]
  return %1 : !hir.any
}

// CHECK-LABEL: @TypeOfConstantUnit
func.func @TypeOfConstantUnit() -> !hir.any {
  // CHECK: [[TY:%.+]] = hir.unit_type
  %0 = hir.constant_unit
  %1 = hir.type_of %0
  // CHECK: return [[TY]]
  return %1 : !hir.any
}

// CHECK-LABEL: @TypeOfConstantBool
func.func @TypeOfConstantBool() -> !hir.any {
  // CHECK: [[TY:%.+]] = hir.bool_type
  %0 = hir.constant_bool <true>
  %1 = hir.type_of %0
  // CHECK: return [[TY]]
  return %1 : !hir.any
}

// CHECK-LABEL: @TypeOfConstantInt
func.func @TypeOfConstantInt() -> !hir.any {
  // CHECK: [[TY:%.+]] = hir.int_type
  %t = hir.int_type
  %0 = hir.constant_int 42 : %t
  %1 = hir.type_of %0
  // CHECK: return [[TY]]
  return %1 : !hir.any
}

// CHECK-LABEL: @CoerceTypeIdentity
func.func @CoerceTypeIdentity() -> !hir.any {
  %t = hir.int_type
  // CHECK: [[V:%.+]] = hir.constant_int 42
  %0 = hir.constant_int 42 : %t
  %1 = hir.coerce_type %0, %t
  // CHECK: return [[V]]
  return %1 : !hir.any
}

// CHECK-LABEL: @CoerceTypeNonIdentity
func.func @CoerceTypeNonIdentity() -> !hir.any {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %0 = hir.constant_int 42 : %t0
  // CHECK: hir.coerce_type
  %1 = hir.coerce_type %0, %t1
  return %1 : !hir.any
}

// CHECK-LABEL: @PackOfUnpack
func.func @PackOfUnpack(%arg0: !hir.any) -> !hir.any {
  %0:2 = hir.opaque_unpack %arg0 : !hir.any, !hir.any
  // CHECK-NOT: hir.opaque_pack
  %1 = hir.opaque_pack(%0#0, %0#1)
  // CHECK: return %arg0
  return %1 : !hir.any
}

// CHECK-LABEL: @PackOfUnpackPartial
func.func @PackOfUnpackPartial(%arg0: !hir.any) -> !hir.any {
  %0:3 = hir.opaque_unpack %arg0 : !hir.any, !hir.any, !hir.any
  // CHECK: hir.opaque_pack
  %1 = hir.opaque_pack(%0#0, %0#1)
  return %1 : !hir.any
}

// CHECK-LABEL: @PackOfUnpackReordered
func.func @PackOfUnpackReordered(%arg0: !hir.any) -> !hir.any {
  %0:2 = hir.opaque_unpack %arg0 : !hir.any, !hir.any
  // CHECK: hir.opaque_pack
  %1 = hir.opaque_pack(%0#1, %0#0)
  return %1 : !hir.any
}

// CHECK-LABEL: @PackOfDifferentUnpacks
func.func @PackOfDifferentUnpacks(%arg0: !hir.any, %arg1: !hir.any) -> !hir.any {
  %0:1 = hir.opaque_unpack %arg0 : !hir.any
  %1:1 = hir.opaque_unpack %arg1 : !hir.any
  // CHECK: hir.opaque_pack
  %2 = hir.opaque_pack(%0#0, %1#0)
  return %2 : !hir.any
}

hir.unified_func @dummy() -> (x: 0, y: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature () -> (%0, %1)
} {
  hir.return : () -> ()
}

// CHECK-LABEL: @TypeOfCallResults
func.func @TypeOfCallResults() -> (!hir.any, !hir.any) {
  // CHECK: [[INT_TY:%.+]] = hir.int_type
  %0 = hir.int_type
  %1 = hir.constant_int 42 : %0
  // CHECK: [[UINT_TY:%.+]] = hir.uint_type
  %2 = hir.uint_type %1
  %3:2 = hir.unified_call @dummy() : () -> (%0, %2) () -> (!hir.any, !hir.any) [] -> [0, 0]
  %4 = hir.type_of %3#0
  %5 = hir.type_of %3#1
  // CHECK: return [[INT_TY]], [[UINT_TY]]
  return %4, %5 : !hir.any, !hir.any
}
