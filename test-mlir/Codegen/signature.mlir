// RUN: silc --parse-only %s | FileCheck %s

// CHECK-LABEL: hir.unchecked_func @empty
// CHECK-NEXT: hir.unchecked_signature () -> ()
fn empty() {}

// CHECK-LABEL: hir.unchecked_func @simple
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[A:%.+]] = hir.unchecked_arg "a", [[INT_TYPE]], 0
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: hir.unchecked_signature ([[A]] : !hir.value) -> ([[INT_TYPE]] : !hir.type)
fn simple(a: int) -> int { 0 }

// CHECK-LABEL: hir.unchecked_func @dependent_types
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[A:%.+]] = hir.unchecked_arg "a", [[INT_TYPE]], 0
// CHECK-NEXT: [[INT_TYPE:%.+]] = hir.int_type
// CHECK-NEXT: [[B:%.+]] = hir.unchecked_arg "b", [[INT_TYPE]], 1
// CHECK-NEXT: [[UINT_TYPE:%.+]] = hir.uint_type [[B]]
// CHECK-NEXT: [[C:%.+]] = hir.unchecked_arg "c", [[UINT_TYPE]], 0
// CHECK-NEXT: hir.unchecked_signature ([[A]], [[B]], [[C]] : !hir.value, !hir.value, !hir.value) -> ()
fn dependent_types(a: int, b: const int, c: uint<b>) {}
