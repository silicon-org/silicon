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

// CHECK-LABEL: @TypeOfConstantInt
func.func @TypeOfConstantInt() -> !hir.any {
  // CHECK: [[TY:%.+]] = hir.int_type
  %0 = hir.constant_int 42
  %1 = hir.type_of %0
  // CHECK: return [[TY]]
  return %1 : !hir.any
}

hir.unified_func @dummy() -> (x: 0, y: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature () -> (%0, %1)
} {
  hir.unified_return
}

// CHECK-LABEL: @TypeOfCallResults
func.func @TypeOfCallResults() -> (!hir.any, !hir.any) {
  // CHECK: [[INT_TY:%.+]] = hir.int_type
  %0 = hir.int_type
  %1 = hir.constant_int 42
  // CHECK: [[UINT_TY:%.+]] = hir.uint_type
  %2 = hir.uint_type %1
  %3:2 = hir.unified_call @dummy() : () -> (%0, %2) () -> (!hir.any, !hir.any) [] -> [0, 0]
  %4 = hir.type_of %3#0
  %5 = hir.type_of %3#1
  // CHECK: return [[INT_TY]], [[UINT_TY]]
  return %4, %5 : !hir.any, !hir.any
}
