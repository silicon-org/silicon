// RUN: silc --parse-only %s | FileCheck %s

// CHECK-LABEL: hir.unchecked_func @empty
// CHECK-NEXT: hir.unchecked_signature () -> ()
fn empty() {}

// CHECK-LABEL: hir.unchecked_func @simple
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[A:%.+]] = hir.unchecked_arg "a", [[INT_TYPE]]
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: hir.unchecked_signature ([[A]] : !hir.value) -> ([[INT_TYPE]] : !hir.type)
fn simple(a: int) -> int { 0 }

// CHECK-LABEL: hir.unchecked_func @dependent_types
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[A:%.+]] = hir.unchecked_arg "a", [[INT_TYPE]]
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[CONST_INT_TYPE:%.+]] = hir.const_type [[INT_TYPE]]
// CHECK-NEXT: [[B:%.+]] = hir.unchecked_arg "b", [[CONST_INT_TYPE]]
// CHECK-NEXT: [[UINT_TYPE:%.+]] = hir.uint_type [[B]]
// CHECK-NEXT: [[C:%.+]] = hir.unchecked_arg "c", [[UINT_TYPE]]
// CHECK-NEXT: hir.unchecked_signature ([[A]], [[B]], [[C]] : !hir.value, !hir.value, !hir.value) -> ()
fn dependent_types(a: int, b: const int, c: uint<b>) {}
