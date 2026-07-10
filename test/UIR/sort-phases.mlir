// RUN: silicon-opt --split-phases2="sort-only=true annotate-phases=true" %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Dissolution: exprs are inlined, pins replaced by their inputs, and the
// interior ops keep their analyzed phases in the maintained phase map.

// CHECK-LABEL: uir.func @DissolveExprsAndPins
uir.func @DissolveExprsAndPins(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // CHECK-NOT: uir.expr
  // CHECK: hir.mul %a, %a {{.*}}pa.phase = "-1"
  %y = uir.expr pin -1 : %t {
    %prod = hir.mul %a, %a : %t
    uir.yield %prod : %t
  }
  // CHECK-NOT: uir.pin
  %pinY = uir.pin %y, 0 : !hir.any
  // The add consumes the former expr result directly.
  // CHECK: [[SUM:%.+]] = hir.add {{.*}}pa.operands = ["-1", "-1", "float"]{{.*}}pa.phase = "0"
  %sum = hir.add %pinY, %pinY : %t
  // CHECK: uir.return [[SUM]]
  uir.return %sum : %t
}

// Nested exprs dissolve inner-first; floating exprs dissolve like pinned ones.
// CHECK-LABEL: uir.func @DissolveNestedExprs
uir.func @DissolveNestedExprs(%a: -1) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // CHECK-NOT: uir.expr
  // CHECK: [[PROD:%.+]] = hir.mul %a, %a {{.*}}pa.phase = "-1"
  // CHECK: [[SUM:%.+]] = hir.add [[PROD]], %a {{.*}}pa.phase = "-1"
  %y = uir.expr pin -1 : %t {
    %inner = uir.expr : %t {
      %prod = hir.mul %a, %a : %t
      uir.yield %prod : %t
    }
    %sum = hir.add %inner, %a : %t
    uir.yield %sum : %t
  }
  // CHECK: uir.return [[SUM]]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Call decomposition: a call to a split callee becomes one hir.call per
// witness entry, chained through an opaque context value, each at its
// absolute phase.

uir.split_func @half(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} [
  -1: @half.0,
  0: @half.1
]

// CHECK-LABEL: uir.func @DecomposeCall
uir.func @DecomposeCall(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK-NOT: uir.call
  // CHECK: [[CTX:%.+]] = hir.call @half.0(%x) : (%{{.*}}) -> (%{{.*}}) {pa.operands = {{.*}}pa.phase = "-1", pa.results = ["-1"]}
  // CHECK: [[RES:%.+]] = hir.call @half.1(%y, [[CTX]]) : (%{{.*}}, %{{.*}}) -> (%{{.*}}) {pa.operands = {{.*}}pa.phase = "0", pa.results = ["0"]}
  %r = uir.call @half(%x, %y) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  // CHECK: uir.return [[RES]]
  uir.return %r : %t
}
