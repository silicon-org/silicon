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

//===----------------------------------------------------------------------===//
// The normalize -> analyze -> sort pipeline on a later-phase early exit: the
// exit decision (%e) is at phase 0 but the returned values are at phase 1.
// Normalization rebuilds the if with an exit-value slot and a payload-free tag;
// the value slot floors to phase 1 while the tag stays at phase 0. Sorting then
// choice-ifies both the tag pick and the phase-1 value slot through ordinary
// branch extraction: phase 0 decides the tag and the value-slot cf choice, and
// phase 1 replays two picks — one reconstructing the exit value (%x for the
// exit arm, poison for the fall-through) and one selecting the function result
// (the exit value or the phase-1 continuation) — returning from a single
// phase return. This is the case a value-keyed provider rule could not sort.

// CHECK-LABEL: uir.func @LaterPhaseExit
uir.func @LaterPhaseExit(%e: 0, %x: 1) -> (result: 1) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: [[P:%.+]]:2 = uir.phase 0 {
  // CHECK-NEXT:   [[IF:%.+]]:2 = uir.if %e
  // The exit arm delivers the value-slot cf choice and the tag choice.
  // CHECK:          hir.unify [[ST]], [[T]]
  // CHECK-NEXT:     hir.choice 0()
  // CHECK-NEXT:     hir.choice 0()
  // CHECK-NEXT:     uir.yield
  // CHECK-NEXT:   } else {
  // CHECK:          uir.yield
  // CHECK-NEXT:   }
  // The receiving pick reconstructs the phase-0 tag; the phase op yields both
  // the value-slot cf choice and the tag pick.
  // CHECK:        [[TAGPICK:%.+]] = hir.pick [[IF]]#0
  // CHECK:        uir.phase_return yield [[IF]]#1, [[TAGPICK]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // Reconstruct the exit value: %x on the exit arm, poison on the fall-through.
  // CHECK-NEXT:   [[VAL:%.+]] = hir.pick [[P]]#0 : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield %x : !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[U:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     hir.yield [[U]]
  // CHECK-NEXT:   }]
  // Select the function result: the exit value or the phase-1 continuation.
  // CHECK-NEXT:   [[RES:%.+]] = hir.pick [[P]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield [[VAL]] : !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:     hir.yield [[Y]]
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return [[RES]] : [[ST]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %t = hir.int_type
  uir.if %e {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}
