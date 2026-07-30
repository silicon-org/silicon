// RUN: silicon-opt --sort-phases --thread-phases %s | FileCheck %s
// RUN: silicon-opt --sort-phases --thread-phases %s | silicon-opt --verify-roundtrip

// Sorting and threading compose: sorting produces jumping sorted form, and
// threading densifies it into the fully threaded contract. These pre-sort
// inputs exercise the two passes back to back.

//===----------------------------------------------------------------------===//
// A three-phase chain: the phase -2 value skips to phase 0 in sorted form, and
// threading routes it hop-by-hop through the phase -1 op.

// CHECK-LABEL: uir.func @Chain
uir.func @Chain(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: [[P2:%.+]] = uir.phase -2 {
  %sq = hir.mul %a, %a : %t
  // CHECK: [[P1:%.+]] = uir.phase -1 {
  // CHECK:   hir.write_var {{.*}}, [[P2]]
  // CHECK:   uir.phase_return yield [[P2]]
  // CHECK: }
  uir.expr pin -1 {
    %v = hir.var : %t
    hir.write_var %v, %sq : %t
    uir.yield
  }
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add [[P1]], %x
  %r = hir.add %sq, %x : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A call composes too: sorting decomposes it into per-phase uir.phase_call
// entries, and threading materializes the empty phases and threads the context.

// CHECK-LABEL: uir.func @Caller
uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: uir.phase_call @Chain2[-1]
  // CHECK: uir.phase_call @Chain2[0]
  %r = uir.call @Chain2(%x, %y) : (%t, %t) -> (%t) [-1, 0] -> [0]
  uir.return %r : %t
}

uir.func @Chain2(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  %r = hir.add %sq, %b : %t
  uir.return %r : %t
}
