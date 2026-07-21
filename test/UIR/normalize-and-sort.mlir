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

//===----------------------------------------------------------------------===//
// The pipeline on a signature-region early exit: the signature is the
// region's return-like exit and normalizes exactly like a return, so a
// same-phase signature exit under an if wraps into one phase-0 op whose pick
// threads the argument- and result-type slots into the single trailing
// uir.signature. The argument and result sit at phase 1, so their type
// slots — selected by the phase-0 dispatch — satisfy their one-phase-earlier
// demand.

// CHECK-LABEL: uir.func @SamePhaseSignatureExit
uir.func @SamePhaseSignatureExit(%x: 1) -> (result: 1) {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[S0:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S1:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[C:%.+]] = hir.constant_int 1 : [[T]]
  // CHECK-NEXT: [[P0:%.+]]:2 = uir.phase 0 {
  // CHECK-NEXT:   [[IF:%.+]]:3 = uir.if [[C]] : [[S0]], [[S1]], [[CT]] {
  // CHECK-NEXT:     [[B:%.+]] = hir.bool_type
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield [[B]], [[B]], [[C0]] : [[S0]], [[S1]], [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK:          uir.yield {{.*}} : [[S0]], [[S1]], [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[PK:%.+]]:2 = hir.pick [[IF]]#2 : !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield [[IF]]#0, [[IF]]#1 : !hir.any, !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.yield [[T]], [[T]] : !hir.any, !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return yield [[PK]]#0, [[PK]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[P0]]#0) -> ([[P0]]#1)
  %int = hir.int_type
  %c = hir.constant_int 1 : %int
  uir.if %c {
    %b = hir.bool_type
    uir.signature (%b) -> (%b)
  }
  uir.signature (%int) -> (%int)
} {
  // CHECK: uir.phase 1 {
  // CHECK-NEXT: uir.phase_return %x
  %t = hir.int_type
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// The pipeline on a later-phase signature exit: the exit decision is at
// phase 0 but the exit's type values only materialize at phase 1. The tag
// and the phase-0 slot deliver at phase 0, while the phase-1 slots
// choice-ify through ordinary branch extraction: phase 1 replays a pick
// reconstructing the exit's type value and a pick dispatching on the tag,
// and the trailing uir.signature consumes crossings from both phases.

// CHECK-LABEL: uir.func @LaterPhaseSignatureExit
uir.func @LaterPhaseSignatureExit(%m: 1, %x: 2) -> (result: 2) {
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: [[P0:%.+]]:3 = uir.phase 0 {
  // CHECK-NEXT:   [[IF:%.+]]:3 = uir.if {{%.+}} : {{%.+}}, {{%.+}}, {{%.+}} {
  // The exit arm delivers the phase-0 type slot, the value-slot cf choice,
  // and the tag choice.
  // CHECK-NEXT:     hir.choice 0()
  // CHECK-NEXT:     hir.choice 0()
  // CHECK-NEXT:     uir.yield [[T]],
  // CHECK-NEXT:   } else {
  // CHECK:          uir.yield
  // CHECK-NEXT:   }
  // The phase-0 dispatch pick selects the phase-0 slot and the tag.
  // CHECK-NEXT:   [[PK0:%.+]]:2 = hir.pick [[IF]]#1 : !hir.any, !hir.any [0 (): {
  // CHECK:        uir.phase_return yield [[IF]]#2, [[PK0]]#0, [[PK0]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P1:%.+]]:2 = uir.phase 1 {
  // Reconstruct the exit's phase-1 type value on the exit arm, poison on the
  // fall-through.
  // CHECK-NEXT:   [[VAL:%.+]]:2 = hir.pick [[P0]]#0 : !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:     [[U:%.+]] = hir.uint_type %m
  // CHECK-NEXT:     hir.yield [[U]], [[U]] : !hir.any, !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[UV0:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     [[UV1:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     hir.yield [[UV0]], [[UV1]] : !hir.any, !hir.any
  // CHECK-NEXT:   }]
  // The replayed dispatch pick selects the phase-1 slots.
  // CHECK-NEXT:   [[PK1:%.+]]:2 = hir.pick [[P0]]#2 : !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield [[VAL]]#0, [[VAL]]#1 : !hir.any, !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.yield [[T]], [[T]] : !hir.any, !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return yield [[PK1]]#0, [[PK1]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[P0]]#1, [[P1]]#0) -> ([[P1]]#1)
  %int = hir.int_type
  %c = hir.constant_int 1 : %int
  uir.if %c {
    %u = hir.uint_type %m
    uir.signature (%int, %u) -> (%u)
  }
  uir.signature (%int, %int) -> (%int)
} {
  // CHECK: uir.phase 2 {
  // CHECK-NEXT: uir.phase_return %x
  %t = hir.int_type
  uir.return %x : %t
}
