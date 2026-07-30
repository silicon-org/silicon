// RUN: silicon-opt --split-funcs %s | FileCheck %s
// RUN: silicon-opt --split-funcs %s | silicon-opt --verify-roundtrip

// The phase-blind function splitter turns each threaded `uir.func` into one
// `hir.func` per `uir.phase` op, wires phase-op crossings into opaque context
// pack/unpack, rewrites `uir.phase_call` into `hir.call` to the callee's
// per-phase entry, reconstructs each per-phase signature, and combines the
// results under a `uir.split_func` (with `hir.eager_func` heading any pre-span
// chain). Inputs are the fully threaded form of thread-phases.mlir.

//===----------------------------------------------------------------------===//
// Case 1: two-phase split of a dependent-type identity function. The dependent
// type uint[N] is computed once in the phase -1 body and rides the opaque
// context into phase 0, unpacked both in the signature (as the type of %x and
// the result) and in the body (as the add type).

// CHECK-LABEL: hir.func private @"Id.-1"(%N) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    hir.signature ([[INT]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[TY]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Id.0(%x, %ctx) -> (result) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[TY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[TY]], [[O]]) -> ([[TY]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[TY2:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[R:%.+]] = hir.add %x, %x : [[TY2]]
// CHECK-NEXT:    hir.return [[R]] : [[TY2]]
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Id(%N: -1, %x: 0) -> (result: 0) {
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    uir.signature ([[INT]], [[TY]]) -> ([[TY]])
// CHECK-NEXT:  } [
// CHECK-NEXT:    -1: @"Id.-1",
// CHECK-NEXT:    0: @Id.0
// CHECK-NEXT:  ]
uir.func @Id(%N: -1, %x: 0) -> (result: 0) {
  %s0:2 = uir.phase -1 {
    %int = hir.int_type
    %ui = hir.uint_type %N
    uir.phase_return yield %ui, %int
  }
  %s1 = uir.phase 0 {
    uir.phase_return yield %s0#0
  }
  uir.signature (%s0#1, %s1) -> (%s1)
} {
  %p = uir.phase -1 {
    %ty = hir.uint_type %N
    uir.phase_return yield %ty
  }
  uir.phase 0 {
    %r = hir.add %x, %x : %p
    uir.phase_return %r : %p
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 2: pre-span eager chain. Phase -2 is argument-free (pre-span), so its
// func heads an `hir.eager_func` whose tail is the earliest span entry (phase
// -1). The eager_func's external signature is the phase -1 entry's declared
// argument %N and its context result — the tail minus its trailing pre-span
// context argument.

// CHECK-LABEL: hir.func private @"Pre.-2"() -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature () -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[T:%.+]] = hir.int_type
// CHECK-NEXT:    [[C:%.+]] = hir.constant_int 5 : [[T]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[C]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"Pre.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[PRE:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[W:%.+]] = hir.add %N, [[PRE]] : [[INT]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[W]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Pre.0(%x, %ctx) -> (result) {
// CHECK:       } {
// CHECK-NEXT:    [[V:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.add %x, [[V]] : [[INT]]
// CHECK-NEXT:    hir.return [[R]] : [[INT]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.eager_func @"Pre.-1"(N) -> (ctx) [
// CHECK-NEXT:    @"Pre.-2",
// CHECK-NEXT:    @"Pre.-1.tail"
// CHECK-NEXT:  ]

// CHECK-LABEL: uir.split_func @Pre(%N: -1, %x: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    -1: @"Pre.-1",
// CHECK-NEXT:    0: @Pre.0
// CHECK-NEXT:  ]
uir.func @Pre(%N: -1, %x: 0) -> (result: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm1, %s0) -> (%s0)
} {
  %pre = uir.phase -2 {
    %t = hir.int_type
    %c = hir.constant_int 5 : %t
    uir.phase_return yield %c
  }
  %v = uir.phase -1 {
    %int = hir.int_type
    %w = hir.add %N, %pre : %int
    uir.phase_return yield %w
  }
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.add %x, %v : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 3: pass-through gap phase. The materialized empty gap phase -1 becomes a
// context-forwarding func: it unpacks its incoming crossing and repacks it
// unchanged. It is a full span entry callers thread through lock-step.

// CHECK-LABEL: hir.func private @"Gap.-2"(%a) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[SQ:%.+]] = hir.mul %a, %a : [[INT]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[SQ]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"Gap.-1"(%ctx) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[SQ:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[SQ]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Gap.0(%x, %ctx) -> (result) {
// CHECK:       } {
// CHECK-NEXT:    [[SQ:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.add [[SQ]], %x : [[INT]]
// CHECK-NEXT:    hir.return [[R]] : [[INT]]
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Gap(%a: -2, %x: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    -2: @"Gap.-2",
// CHECK-NEXT:    -1: @"Gap.-1",
// CHECK-NEXT:    0: @Gap.0
// CHECK-NEXT:  ]
uir.func @Gap(%a: -2, %x: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.phase -1 {
    uir.phase_return
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %s0) -> (%s0)
} {
  %c0 = uir.phase -2 {
    %int = hir.int_type
    %sq = hir.mul %a, %a : %int
    uir.phase_return yield %sq
  }
  %c1 = uir.phase -1 {
    uir.phase_return yield %c0
  }
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.add %c1, %x : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 4: caller side. Each `uir.phase_call` becomes an `hir.call` to the
// callee's concrete per-phase entry, living in the caller's own per-phase func.
// The callee context returned by the phase -1 entry rides the caller's own
// context across the caller's phase boundary and is unpacked before the second
// call.

// CHECK-LABEL: hir.func private @"MultiPhase.-1"(%a) -> (ctx) {
// CHECK-LABEL: hir.func private @MultiPhase.0(%b, %ctx) -> (result) {
uir.func @MultiPhase(%a: -1, %b: 0) -> (result: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm1, %s0) -> (%s0)
} {
  %p = uir.phase -1 {
    %int = hir.int_type
    %sq = hir.mul %a, %a : %int
    uir.phase_return yield %sq
  }
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.add %p, %b : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

// CHECK-LABEL: hir.func private @"Caller.-1"(%x) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[CC:%.+]] = hir.call @"MultiPhase.-1"(%x) : ([[INT]]) -> ([[O]])
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[CC]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Caller.0(%y, %ctx) -> (result) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[CC:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.call @MultiPhase.0(%y, [[CC]]) : ([[INT]], [[O]]) -> ([[INT]])
// CHECK-NEXT:    hir.return [[R]] : [[INT]]
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm1, %s0) -> (%s0)
} {
  %ctx = uir.phase -1 {
    %t = hir.int_type
    %c = uir.phase_call @MultiPhase[-1] (%x) : (%t) -> () context
    uir.phase_return yield %c
  }
  uir.phase 0 {
    %t = hir.int_type
    %r = uir.phase_call @MultiPhase[0] (%y) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 5: split caller of a dependent-signature callee. The caller recomputes
// uint[N] itself from the const argument %N in its own phase -1 body and
// threads it through its own context as an ordinary sig-crossing, side by side
// with the callee's opaque context. The callee's context is never read for
// types; the caller supplies `typeOfArgs` for the phase-0 call independently.

// CHECK-LABEL: hir.func private @"Use.-1"(%N) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[XTY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    [[IDCTX:%.+]] = hir.call @"Id.-1"(%N) : ([[INT]]) -> ([[O]])
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[XTY]], [[IDCTX]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Use.0(%x, %ctx) -> (r) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    hir.signature ([[U]]#0, [[O]]) -> ([[U]]#0)
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U2:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    [[R:%.+]] = hir.call @Id.0(%x, [[U2]]#1) : ([[U2]]#0, [[O2]]) -> ([[U2]]#0)
// CHECK-NEXT:    hir.return [[R]] : [[U2]]#0
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Use(%N: -1, %x: 0) -> (r: 0) {
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[XTY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    uir.signature ([[INT]], [[XTY]]) -> ([[XTY]])
// CHECK-NEXT:  } [
// CHECK-NEXT:    -1: @"Use.-1",
// CHECK-NEXT:    0: @Use.0
// CHECK-NEXT:  ]
uir.func @Use(%N: -1, %x: 0) -> (r: 0) {
  %s0:2 = uir.phase -1 {
    %int = hir.int_type
    %ui = hir.uint_type %N
    uir.phase_return yield %ui, %int
  }
  %s1 = uir.phase 0 {
    uir.phase_return yield %s0#0
  }
  uir.signature (%s0#1, %s1) -> (%s1)
} {
  %c:2 = uir.phase -1 {
    %int = hir.int_type
    %xty = hir.uint_type %N
    %idctx = uir.phase_call @Id[-1] (%N) : (%int) -> () context
    uir.phase_return yield %xty, %idctx
  }
  uir.phase 0 {
    %r = uir.phase_call @Id[0] (%x) context %c#1 : (%c#0) -> (%c#0)
    uir.phase_return %r : %c#0
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 6: a dependent argument type the body never uses. The body threads no
// crossing between phase -1 and phase 0, but %y's type uint[N] is a signature
// crossing computed at phase -1. It is relocated into the phase -1 body and
// packed into the context so phase 0's signature can unpack it — the body and
// signature share one opaque context even when only the signature crosses.

// CHECK-LABEL: hir.func private @"F.-1"(%N) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[XTY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[XTY]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @F.0(%y, %ctx) -> () {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[XTY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[XTY]], [[O]]) -> ()
// CHECK-NEXT:  } {
// CHECK-NEXT:    hir.return
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @F(%N: -1, %y: 0) -> () {
uir.func @F(%N: -1, %y: 0) -> () {
  %s0:2 = uir.phase -1 {
    %int = hir.int_type
    %ui = hir.uint_type %N
    uir.phase_return yield %ui, %int
  }
  %s1 = uir.phase 0 {
    uir.phase_return yield %s0#0
  }
  uir.signature (%s0#1, %s1) -> ()
} {
  uir.phase -1 {
    uir.phase_return
  }
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}
