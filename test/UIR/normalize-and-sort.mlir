// RUN: silicon-opt %s --normalize-exits --sort-phases | FileCheck %s

//===----------------------------------------------------------------------===//
// The normalize -> analyze -> sort pipeline on an ordinary same-phase early
// exit: normalization leaves only same-phase value CF, so sorting wraps the
// whole body into one phase op, the floating choice_type and result-type
// inferrable stay at the top level, and the function result rides the
// phase return typed by the inferrable.

// CHECK-LABEL: uir.func @SamePhaseExit
uir.func @SamePhaseExit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[IF:%.+]]:2 = uir.if %c : [[ST]], [[CT]] {
  // CHECK:        [[P:%.+]] = hir.pick [[IF]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield [[IF]]#0
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:     hir.unify [[ST]], [[T]]
  // CHECK-NEXT:     hir.yield [[Y]]
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return [[P]] : [[ST]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}
