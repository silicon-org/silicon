// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK-LABEL: @Unify
func.func @Unify() -> !hir.type {
  // CHECK: [[TY:%.+]] = hir.int_type
  %0 = hir.int_type
  %1 = hir.unify %0, %0 : !hir.type
  // CHECK: return [[TY]]
  return %1 : !hir.type
}

// CHECK-LABEL: @TypeOfConstant
func.func @TypeOfConstant() -> !hir.type {
  // CHECK: [[TY:%.+]] = hir.int_type
  %0 = hir.constant_int 42
  %1 = hir.type_of %0 : !hir.value
  // CHECK: return [[TY]]
  return %1 : !hir.type
}

// CHECK-LABEL: @TypeOfCallResults
func.func @TypeOfCallResults() -> (!hir.type, !hir.type) {
  // CHECK: [[INT_TY:%.+]] = hir.int_type
  %0 = hir.int_type
  %1 = hir.constant_int 42
  // CHECK: [[UINT_TY:%.+]] = hir.uint_type
  %2 = hir.uint_type %1
  %3:2 = hir.checked_call @dummy() : () -> (%0, %2 : !hir.type, !hir.type) [] []
  %4 = hir.type_of %3#0 : !hir.value
  %5 = hir.type_of %3#1 : !hir.value
  // CHECK: return [[INT_TY]], [[UINT_TY]]
  return %4, %5 : !hir.type, !hir.type
}
