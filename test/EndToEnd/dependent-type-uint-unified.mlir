// RUN: silc --ir-mir %s | FileCheck %s

// Test end-to-end pipeline from unified form through the full pass pipeline
// for dependent types where `hir.uint_type %N` has a block arg width. This
// exercises the path: CheckCalls → InferTypes → SplitPhases → HIRToMIR
// (mir.uint_type) → Interpret → SpecializeFuncs.

//===----------------------------------------------------------------------===//
// id(const N: int, x: uint<N>) -> uint<N>
//===----------------------------------------------------------------------===//

// CHECK: mir.evaluated_func {{.*}}@main.{{.*}} [#si.int<42> : !si.int]

hir.unified_func private @id(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1) -> (%1)
} {
  %0 = hir.uint_type %N
  %1 = hir.coerce_type %x, %0
  hir.return %1 -> (%0)
}

// main calls id(8, 42).
hir.unified_func @main() -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.uint_type %1
  hir.signature () -> (%2)
} {
  %tN = hir.int_type
  %N = hir.constant_int 8 : %tN
  %tX = hir.uint_type %N
  %x = hir.constant_int 42 : %tX
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %r = hir.unified_call @id(%N, %x) : (%t0, %t1) -> (%t2) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %tr = hir.type_of %r
  hir.return %r -> (%tr)
}
