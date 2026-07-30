// RUN: silicon-opt --sort-phases --thread-phases --split-funcs %s | FileCheck %s
// RUN: silicon-opt --sort-phases --thread-phases --split-funcs %s | silicon-opt --verify-roundtrip

// Sorting, threading, and splitting compose end to end: sorting produces the
// jumping sorted form, threading densifies it into the fully threaded contract,
// and splitting emits per-phase `hir.func` ops plus the structural
// `uir.split_func`. These pre-sort inputs exercise all three passes back to
// back on more than one function.

//===----------------------------------------------------------------------===//
// A three-phase chain whose phase -2 value skips to phase 0: threading routes
// it through the materialized gap phase -1 (which carries a receiving `hir.var`
// write), and splitting turns each phase into a context-threaded func.

// CHECK-LABEL: hir.func private @"Chain.-2"(%a) -> (ctx) {
// CHECK:         [[SQ:%.+]] = hir.mul %a, %a
// CHECK:         hir.opaque_pack([[SQ]])

// CHECK-LABEL: hir.func private @"Chain.-1"(%ctx) -> (ctx) {
// CHECK:       } {
// CHECK:         [[U:%.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.write_var {{%.+}}, [[U]]
// CHECK:         hir.opaque_pack([[U]])

// CHECK-LABEL: hir.func private @Chain.0(%x, %ctx) -> (result) {
// CHECK:       } {
// CHECK:         [[U:%.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.add [[U]], %x

// CHECK-LABEL: uir.split_func @Chain(%a: -2, %x: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    -2: @"Chain.-2",
// CHECK-NEXT:    -1: @"Chain.-1",
// CHECK-NEXT:    0: @Chain.0
// CHECK-NEXT:  ]
uir.func @Chain(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  uir.expr pin -1 {
    %v = hir.var : %t
    hir.write_var %v, %sq : %t
    uir.yield
  }
  %r = hir.add %sq, %x : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A caller: sorting decomposes the `uir.call` into per-phase `uir.phase_call`
// entries, and splitting rewrites each into an `hir.call` to the callee's
// concrete per-phase func. The callee context threads across the caller's own
// phase boundary.

// CHECK-LABEL: hir.func private @"Caller.-1"(%x) -> (ctx) {
// CHECK:       } {
// CHECK:         [[CC:%.+]] = hir.call @"Chain2.-1"(%x)
// CHECK:         hir.opaque_pack([[CC]])

// CHECK-LABEL: hir.func private @Caller.0(%y, %ctx) -> (result) {
// CHECK:       } {
// CHECK:         [[CC:%.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.call @Chain2.0(%y, [[CC]])

// CHECK-LABEL: uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.call @Chain2(%x, %y) : (%t, %t) -> (%t) [-1, 0] -> [0]
  uir.return %r : %t
}

// CHECK-LABEL: uir.split_func @Chain2(%a: -1, %b: 0) -> (result: 0) {
uir.func @Chain2(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  %r = hir.add %sq, %b : %t
  uir.return %r : %t
}
