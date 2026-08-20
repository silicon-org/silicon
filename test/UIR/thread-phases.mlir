// RUN: silicon-opt --thread-phases %s | FileCheck %s
// RUN: silicon-opt --thread-phases %s | silicon-opt --verify-roundtrip

// Threading normalization brings sorted phase form into the fully threaded
// shape the phase-blind splitter consumes: one uir.phase op per phase in a
// contiguous range, every crossing yielded by the immediately preceding phase
// op, and no floating ops left at the region top level. Body checks are
// anchored past the `} {` region separator so they do not match the
// signature region, which threads by the same rules.
//
// The one asymmetry is where the two regions' phase ranges land: a type
// ascribing a phase-P argument or result is phase P-1 content, so the
// signature's range is the declared span shifted one phase earlier. Every
// function whose arguments start at phase P therefore grows a signature phase
// P-1, below its declared span, that computes their types.

//===----------------------------------------------------------------------===//
// Gap phase materialized between two content phases, and a distance-2 crossing
// threaded through it: phase -2 yields the value, the materialized phase -1
// re-yields it, and phase 0 consumes the adjacent result.

// CHECK-LABEL: uir.func @Gap
uir.func @Gap(%a: -2, %x: 0) -> (result: 0) {
  // The signature's float is sunk once per delivery phase: phase -3 types the
  // phase -2 argument %a, phase -1 types the phase 0 argument %x and result.
  // The gap phase -2 in between is materialized empty like any other.
  // CHECK: [[S3:%.+]] = uir.phase -3 {
  // CHECK-NEXT: [[I3:%.+]] = hir.int_type
  // CHECK-NEXT: uir.phase_return yield [[I3]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase -2 {
  // CHECK-NEXT: uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S1:%.+]] = uir.phase -1 {
  // CHECK-NEXT: [[I1:%.+]] = hir.int_type
  // CHECK-NEXT: uir.phase_return yield [[I1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[S3]], [[S1]]) -> ([[S1]])
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: [[P2:%.+]] = uir.phase -2 {
  // CHECK:   [[SQ:%.+]] = hir.mul %a, %a
  // CHECK:   uir.phase_return yield [[SQ]]
  // CHECK: }
  %p = uir.phase -2 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return yield %sq
  }
  // CHECK: [[P1:%.+]] = uir.phase -1 {
  // CHECK-NEXT: uir.phase_return yield [[P2]]
  // CHECK-NEXT: }
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add [[P1]], %x
  uir.phase 0 {
    %r = hir.add %p, %x : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Argument consumed two phases after its declared phase: it enters the chain
// at its own phase (-2), gets threaded through -1, and is consumed at 0.

// CHECK-LABEL: uir.func @ArgCross
uir.func @ArgCross(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: [[P2:%.+]] = uir.phase -2 {
  // CHECK-NEXT: uir.phase_return yield %a
  // CHECK-NEXT: }
  // CHECK: [[P1:%.+]] = uir.phase -1 {
  // CHECK-NEXT: uir.phase_return yield [[P2]]
  // CHECK-NEXT: }
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add [[P1]], %x
  uir.phase 0 {
    %r = hir.add %a, %x : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A float used by two phases is cloned into each, and a float whose operand is
// itself a float is cloned as a whole tree.

// CHECK-LABEL: uir.func @Floats
uir.func @Floats(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  %two = hir.add %one, %one : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT: hir.int_type
  // CHECK-NEXT: [[ONE1:%.+]] = hir.constant_int 1
  // CHECK-NEXT: [[TWO1:%.+]] = hir.add [[ONE1]], [[ONE1]]
  // CHECK-NEXT: hir.mul %a, [[TWO1]]
  uir.phase -1 {
    %m = hir.mul %a, %two : %t
    uir.phase_return
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: hir.int_type
  // CHECK-NEXT: [[ONE2:%.+]] = hir.constant_int 1
  // CHECK-NEXT: [[TWO2:%.+]] = hir.add [[ONE2]], [[ONE2]]
  // CHECK-NEXT: hir.add %x, [[TWO2]]
  uir.phase 0 {
    %r = hir.add %x, %two : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A result-only phase stays, and the contentless span phase 0 (declared but
// with no content) is materialized empty.

// CHECK-LABEL: uir.func @Contentless
uir.func @Contentless(%a: -1) -> (c: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: uir.phase -1 {
  // CHECK:   uir.phase_return %a
  // CHECK: }
  uir.phase -1 {
    uir.phase_return %a : %t
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Signature region: the dependent signature threads its own phase result and
// sinks its float, delivering each type value as a result of the phase op one
// before the argument or result it ascribes.

// CHECK-LABEL: uir.func @DepSig
uir.func @DepSig(%N: -1, %x: 0) -> (result: 0) {
  // %N's type is a plain int computed at phase -2, one before %N's own phase;
  // %x's and the result's dependent uint type is computed at phase -1 from %N.
  // CHECK: [[S0:%.+]] = uir.phase -2 {
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: uir.phase_return yield [[INT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[S1:%.+]] = uir.phase -1 {
  // CHECK-NEXT: [[UI:%.+]] = hir.uint_type %N
  // CHECK-NEXT: uir.phase_return yield [[UI]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[S0]], [[S1]]) -> ([[S1]])
  %int = hir.int_type
  %p = uir.phase -1 {
    %ty = hir.uint_type %N
    uir.phase_return yield %ty
  }
  uir.signature (%int, %p) -> (%p)
} {
  // CHECK: } {
  // CHECK: uir.phase -1 {
  // CHECK:   [[Q:%.+]] = hir.uint_type %N
  // CHECK:   uir.phase_return yield [[Q]]
  // CHECK: }
  %q = uir.phase -1 {
    %ty = hir.uint_type %N
    uir.phase_return yield %ty
  }
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add %x, %x
  uir.phase 0 {
    %r = hir.add %x, %x : %q
    uir.phase_return %r : %q
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A pre-span phase op (earlier than the declared span {0}) participates in the
// chain identically: its crossing threads into phase 0.

// CHECK-LABEL: uir.func @PreSpan
uir.func @PreSpan(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  %one = hir.constant_int 1 : %t
  // CHECK: [[P:%.+]] = uir.phase -1 {
  // CHECK:   [[C:%.+]] = hir.add
  // CHECK:   uir.phase_return yield [[C]]
  // CHECK: }
  %p = uir.phase -1 {
    %c = hir.add %one, %one : %t
    uir.phase_return yield %c
  }
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add %x, [[P]]
  uir.phase 0 {
    %r = hir.add %x, %p : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A trailing internal phase (phase 2, later than the declared span {0}) is a
// content phase like any other; the gap phase 1 between it and phase 0 is
// materialized and threads the crossing.

// CHECK-LABEL: uir.func @TrailingInternal
uir.func @TrailingInternal(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: [[P0:%.+]] = uir.phase 0 {
  // CHECK:   [[SQ:%.+]] = hir.mul %a, %a
  // CHECK:   uir.phase_return [[SQ]] : {{.+}} yield [[SQ]]
  // CHECK: }
  %p = uir.phase 0 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return %sq : %t yield %sq
  }
  // CHECK: [[P1:%.+]] = uir.phase 1 {
  // CHECK-NEXT: uir.phase_return yield [[P0]]
  // CHECK-NEXT: }
  // CHECK: uir.phase 2 {
  // CHECK:   hir.write_var {{%.+}}, [[P1]]
  uir.phase 2 {
    %v = hir.var : %t
    hir.write_var %v, %p : %t
    uir.phase_return
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A uir.phase_call chain whose entries sit in non-adjacent phases: the opaque
// context value threads through the materialized intermediate phase like any
// other crossing.

// CHECK-LABEL: uir.func @NonAdjacentCall
uir.func @NonAdjacentCall(%x: -2, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: } {
  %t = hir.int_type
  // CHECK: [[C0:%.+]] = uir.phase -2 {
  // CHECK:   [[CTX:%.+]] = uir.phase_call @Callee[-1] (%x) {{.*}}context
  // CHECK:   uir.phase_return yield [[CTX]]
  // CHECK: }
  %ctx = uir.phase -2 {
    %c = uir.phase_call @Callee[-1] (%x) : (%t) -> () context
    uir.phase_return yield %c
  }
  // CHECK: [[C1:%.+]] = uir.phase -1 {
  // CHECK-NEXT: uir.phase_return yield [[C0]]
  // CHECK-NEXT: }
  // CHECK: uir.phase 0 {
  // CHECK:   uir.phase_call @Callee[0] (%y) context [[C1]]
  uir.phase 0 {
    %r = uir.phase_call @Callee[0] (%y) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

uir.func @Callee(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %p = uir.phase -1 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return yield %sq
  }
  uir.phase 0 {
    %r = hir.add %p, %b : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A multi-result float sunk into the signature keeps each result distinct: its
// op is cloned once, and each referenced result rides its own phase result
// rather than collapsing to result #0. Both arguments are declared at phase 0,
// so their types are delivered by a signature phase -1 below the span.

// CHECK-LABEL: uir.func @MultiResultFloatSig
uir.func @MultiResultFloatSig(%a: 0, %b: 0) -> (result: 0) {
  // CHECK: [[S:%.+]]:2 = uir.phase -1 {
  // CHECK:   [[U:%.+]]:2 = hir.opaque_unpack
  // CHECK:   uir.phase_return yield [[U]]#1, [[U]]#0
  // CHECK: }
  // CHECK: uir.signature ([[S]]#0, [[S]]#1) -> ([[S]]#1)
  %t = hir.int_type
  %pack = hir.opaque_pack(%t, %t)
  %u:2 = hir.opaque_unpack %pack : !hir.any, !hir.any
  uir.signature (%u#1, %u#0) -> (%u#0)
} {
  %t = hir.int_type
  uir.phase 0 {
    %r = hir.add %a, %b : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}
