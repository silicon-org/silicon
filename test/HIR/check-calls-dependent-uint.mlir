// RUN: silicon-opt --check-calls %s | FileCheck %s

// Test that CheckCalls correctly handles dependent uint<N> types in
// signatures. This exercises the fix for the dominance violation where
// cloned uint_type ops referenced coerce_type results defined later.

// CHECK-LABEL: hir.unified_func private @id
// CHECK: %[[INT_T:.*]] = hir.int_type
// CHECK: %[[UINT_T:.*]] = hir.uint_type %N
// CHECK: %[[UINT_T2:.*]] = hir.uint_type %N
// CHECK: %[[COERCE_N:.*]] = hir.coerce_type %N, %[[INT_T]]
// CHECK: %[[COERCE_X:.*]] = hir.coerce_type %x, %[[UINT_T]]
hir.unified_func private @id(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %2 = hir.uint_type %N
  hir.signature (%0, %1) -> (%2)
} {
  %0 = hir.type_of %x
  hir.return %x -> (%0)
}

// Two args sharing the same dependent type.
// CHECK-LABEL: hir.unified_func private @typed_add
// CHECK: hir.coerce_type %a
// CHECK: hir.coerce_type %b
hir.unified_func private @typed_add(%N: -1, %a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %2 = hir.uint_type %N
  %3 = hir.uint_type %N
  hir.signature (%0, %1, %2) -> (%3)
} {
  %0 = hir.type_of %a
  %1 = hir.type_of %b
  %2 = hir.add %a, %b : %0
  %3 = hir.type_of %2
  hir.return %2 -> (%3)
}
