// RUN: silicon-opt --split-funcs %s | FileCheck %s
// RUN: silicon-opt --split-funcs %s | silicon-opt --verify-roundtrip

// The phase-blind function splitter turns each threaded `uir.func` into one
// `hir.func` per phase, merging the signature region's phase-P content with the
// body region's, wires phase-op crossings into opaque context pack/unpack,
// rewrites `uir.phase_call` into `hir.call` to the callee's per-phase entry,
// reconstructs each per-phase signature out of context unpacks alone, and fills
// in the phase list of the `uir.split_func` holding the declaration (with
// `hir.eager_func` heading any pre-span chain).
//
// Inputs are the fully threaded form of thread-phases.mlir, paired with the
// `uir.split_func` in the sorted state that phase sorting created for it: the
// flat signature is preserved from before sorting, never reconstructed here.
//
// A phase-P argument's or result's type is phase P-1 content, so every function
// declaring arguments at phase P carries a signature phase P-1 computing their
// types. That phase lies below the declared span, which is why nearly every
// function here ends up with a pre-span eager chain.

//===----------------------------------------------------------------------===//
// Case 1: three-phase split of a dependent-type identity function. %N's own
// plain int type is computed in the pre-span phase -2; the dependent type
// uint[N] is computed in phase -1 and rides the opaque context into phase 0,
// unpacked both in the signature (as the type of %x and the result) and in the
// body (as the add type). Each per-phase signature is nothing but unpacks.

// CHECK-LABEL: hir.func private @"Id.-2"() -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature () -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[INT]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// The phase -1 func holds the signature region's phase -1 content first, then
// the body's. Both compute uint[N]; the two coincide in one context slot, so
// the body's copy is left dead for the canonicalizer.
// CHECK-LABEL: hir.func private @"Id.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[INT]], [[O]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT2:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    hir.uint_type %N
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

// CHECK-LABEL: hir.eager_func @"Id.-1"(N) -> (ctx) [
// CHECK-NEXT:    @"Id.-2",
// CHECK-NEXT:    @"Id.-1.tail"
// CHECK-NEXT:  ]

// CHECK-LABEL: uir.split_func @Id(%N: -1, %x: 0) -> (result: 0) {
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    uir.signature ([[INT]], [[TY]]) -> ([[TY]])
// CHECK-NEXT:  } [
// CHECK-NEXT:    -1: @"Id.-1",
// CHECK-NEXT:    0: @Id.0
// CHECK-NEXT:  ]
uir.split_func @Id(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  %ty = hir.uint_type %N
  uir.signature (%int, %ty) -> (%ty)
} impl @Id.impl []

uir.func @Id.impl(%N: -1, %x: 0) -> (result: 0) {
  %s0 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase -1 {
    %ui = hir.uint_type %N
    uir.phase_return yield %ui
  }
  uir.signature (%s0, %s1) -> (%s1)
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
// context argument. The pre-span func here merges the body's constant with the
// signature's type computation for %N, and packs both into one context.

// CHECK-LABEL: hir.func private @"Pre.-2"() -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature () -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[NTY:%.+]] = hir.int_type
// CHECK-NEXT:    [[T:%.+]] = hir.int_type
// CHECK-NEXT:    [[C:%.+]] = hir.constant_int 5 : [[T]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[NTY]], [[C]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"Pre.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    hir.signature ([[U]]#0, [[O]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    [[XTY:%.+]] = hir.int_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[W:%.+]] = hir.add %N, [[U]]#1 : [[INT]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[XTY]], [[W]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Pre.0(%x, %ctx) -> (result) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    hir.signature ([[U]]#0, [[O]]) -> ([[U]]#0)
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.add %x, [[U]]#1 : [[INT]]
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
uir.split_func @Pre(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @Pre.impl []

uir.func @Pre.impl(%N: -1, %x: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %sm1) -> (%sm1)
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
// Case 3: pass-through gap phase. The gap phase -1 carries no body content, so
// its func only forwards: it unpacks its incoming crossing and repacks it,
// alongside the signature's phase -1 type computation. It is a full span entry
// callers thread through lock-step. The type of %a, declared at phase -2, is
// computed one phase earlier still, so the chain reaches down to phase -3.

// CHECK-LABEL: hir.func private @"Gap.-3"() -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature () -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[INT]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"Gap.-2.tail"(%a, %ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[ATY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[ATY]], [[O]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[SQ:%.+]] = hir.mul %a, %a : [[INT]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[SQ]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"Gap.-1"(%ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature ([[O]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[SQ:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[XTY:%.+]] = hir.int_type
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[XTY]], [[SQ]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Gap.0(%x, %ctx) -> (result) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    hir.signature ([[U]]#0, [[O]]) -> ([[U]]#0)
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.add [[U]]#1, %x : [[INT]]
// CHECK-NEXT:    hir.return [[R]] : [[INT]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.eager_func @"Gap.-2"(a) -> (ctx) [
// CHECK-NEXT:    @"Gap.-3",
// CHECK-NEXT:    @"Gap.-2.tail"
// CHECK-NEXT:  ]

// CHECK-LABEL: uir.split_func @Gap(%a: -2, %x: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    -2: @"Gap.-2",
// CHECK-NEXT:    -1: @"Gap.-1",
// CHECK-NEXT:    0: @Gap.0
// CHECK-NEXT:  ]
uir.split_func @Gap(%a: -2, %x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @Gap.impl []

uir.func @Gap.impl(%a: -2, %x: 0) -> (result: 0) {
  %sm3 = uir.phase -3 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.phase -2 {
    uir.phase_return
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm3, %sm1) -> (%sm1)
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

// CHECK-LABEL: hir.func private @"MultiPhase.-1.tail"(%a, %ctx) -> (ctx) {
// CHECK-LABEL: hir.func private @MultiPhase.0(%b, %ctx) -> (result) {
// CHECK-LABEL: hir.eager_func @"MultiPhase.-1"(a) -> (ctx) [
uir.split_func @MultiPhase(%a: -1, %b: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @MultiPhase.impl []

uir.func @MultiPhase.impl(%a: -1, %b: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %sm1) -> (%sm1)
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

// The callee's phase -1 entry is its eager_func, which the call targets like
// any other function.
// CHECK-LABEL: hir.func private @"Caller.-1.tail"(%x, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[YTY:%.+]] = hir.int_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[CC:%.+]] = hir.call @"MultiPhase.-1"(%x) : ([[INT]]) -> ([[O]])
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[YTY]], [[CC]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @Caller.0(%y, %ctx) -> (result) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[U:%.+]]:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[R:%.+]] = hir.call @MultiPhase.0(%y, [[U]]#1) : ([[INT]], [[O]]) -> ([[INT]])
// CHECK-NEXT:    hir.return [[R]] : [[INT]]
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @Caller.impl []

uir.func @Caller.impl(%x: -1, %y: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %sm1) -> (%sm1)
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

// CHECK-LABEL: hir.func private @"Use.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[XTY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    hir.uint_type %N
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
uir.split_func @Use(%N: -1, %x: 0) -> (r: 0) {
  %int = hir.int_type
  %xty = hir.uint_type %N
  uir.signature (%int, %xty) -> (%xty)
} impl @Use.impl []

uir.func @Use.impl(%N: -1, %x: 0) -> (r: 0) {
  %s0 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase -1 {
    %ui = hir.uint_type %N
    uir.phase_return yield %ui
  }
  uir.signature (%s0, %s1) -> (%s1)
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
// crossing between phase -1 and phase 0, but %y's type uint[N] is signature
// content at phase -1. The phase -1 func is built from the signature region's
// content alone and packs it into the context so phase 0's signature can unpack
// it — the body and signature share one opaque context even when the body
// contributes nothing to it.

// CHECK-LABEL: hir.func private @"F.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[XTY:%.+]] = hir.uint_type %N
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[XTY]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @F.0(%y, %ctx) -> () {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[XTY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[XTY]], [[O]]) -> ()
// CHECK-NEXT:  } {
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.return
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @F(%N: -1, %y: 0) -> () {
uir.split_func @F(%N: -1, %y: 0) -> () {
  %int = hir.int_type
  %xty = hir.uint_type %N
  uir.signature (%int, %xty) -> ()
} impl @F.impl []

uir.func @F.impl(%N: -1, %y: 0) -> () {
  %s0 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase -1 {
    %ui = hir.uint_type %N
    uir.phase_return yield %ui
  }
  uir.signature (%s0, %s1) -> ()
} {
  uir.phase -1 {
    uir.phase_return
  }
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 7: per-phase function names are cosmetic. A pre-existing symbol occupies
// the callee's conventional phase name, so the splitter uniquifies it; the
// caller's `uir.phase_call` still resolves through the callee split_func's
// phase list, which is the authoritative mapping.

// The squatter, left untouched by the pass.
// CHECK-LABEL: hir.func private @"Cosmetic.-1"(%a) -> (r) {
hir.func private @"Cosmetic.-1"(%a) -> (r) {
  %int = hir.int_type
  hir.signature (%int) -> (%int)
} {
  %int = hir.int_type
  hir.return %a : %int
}

// CHECK-LABEL: hir.func private @"Cosmetic.-1.tail"(%a, %ctx) -> (ctx) {
// CHECK-LABEL: hir.func private @Cosmetic.0(%b, %ctx) -> (result) {
// CHECK-LABEL: hir.eager_func @"Cosmetic.-1_1"(a) -> (ctx) [
// CHECK-LABEL: uir.split_func @Cosmetic(%a: -1, %b: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    -1: @"Cosmetic.-1_1",
// CHECK-NEXT:    0: @Cosmetic.0
// CHECK-NEXT:  ]
uir.split_func @Cosmetic(%a: -1, %b: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @Cosmetic.impl []

uir.func @Cosmetic.impl(%a: -1, %b: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %sm1) -> (%sm1)
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

// CHECK-LABEL: hir.func private @"CosmeticCaller.-1.tail"(%x, %ctx) -> (ctx) {
// CHECK:         hir.call @"Cosmetic.-1_1"(%x)
// CHECK-LABEL: hir.func private @CosmeticCaller.0(%y, %ctx) -> (result) {
// CHECK:         hir.call @Cosmetic.0(%y,
uir.split_func @CosmeticCaller(%x: -1, %y: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @CosmeticCaller.impl []

uir.func @CosmeticCaller.impl(%x: -1, %y: 0) -> (result: 0) {
  %sm2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm2, %sm1) -> (%sm1)
} {
  %ctx = uir.phase -1 {
    %t = hir.int_type
    %c = uir.phase_call @Cosmetic[-1] (%x) : (%t) -> () context
    uir.phase_return yield %c
  }
  uir.phase 0 {
    %t = hir.int_type
    %r = uir.phase_call @Cosmetic[0] (%y) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 8: a `uir.split_func` in the declared state — an interface whose
// implementation has not been compiled, e.g. from a precompiled library — has
// no implementation to split and passes through untouched.

// CHECK-LABEL: uir.split_func @Declared(%a: -1, %b: 0) -> (result: 0) {
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    uir.signature ([[INT]], [[INT]]) -> ([[INT]])
// CHECK-NEXT:  } []
uir.split_func @Declared(%a: -1, %b: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} []

//===----------------------------------------------------------------------===//
// Case 9: a dependent argument type computed by a call in the signature. Since
// splitting merges the signature's phase content into the per-phase func rather
// than relocating it, a `uir.phase_call` in the signature needs no special
// handling: it lands in the phase -1 func like any other content, and the type
// it produces rides the shared context into phase 0's signature. @Width spans
// [0, 0] and is called at offset -1, so it contributes a single entry.

// CHECK-LABEL: hir.func private @"sig_call.-2"() -> (ctx) {
// CHECK:       } {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[INT]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"sig_call.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[NTY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[NTY]], [[O]]) -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[W:%.+]] = hir.call @Width.0(%N) : ([[INT]]) -> ([[INT]])
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type [[W]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[TY]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @sig_call.0(%x, %ctx) -> (result) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[TY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[TY]], [[O]]) -> ([[TY]])
// CHECK-NEXT:  } {

// CHECK-LABEL: hir.eager_func @"sig_call.-1"(N) -> (ctx) [
// CHECK-NEXT:    @"sig_call.-2",
// CHECK-NEXT:    @"sig_call.-1.tail"
// CHECK-NEXT:  ]

// The flat signature keeps the `uir.call` verbatim, as sorting snapshotted it.
// CHECK-LABEL: uir.split_func @sig_call(%N: -1, %x: 0) -> (result: 0) {
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[W:%.+]] = uir.call @Width
// CHECK-NEXT:    hir.uint_type [[W]]
// CHECK:       } [
// CHECK-NEXT:    -1: @"sig_call.-1",
// CHECK-NEXT:    0: @sig_call.0
// CHECK-NEXT:  ]
uir.split_func @sig_call(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  %w = uir.call @Width(%N) : (%int) -> (%int) [0] -> [0]
  %ty = hir.uint_type %w
  uir.signature (%int, %ty) -> (%ty)
} impl @sig_call.impl []

uir.func @sig_call.impl(%N: -1, %x: 0) -> (result: 0) {
  %s2 = uir.phase -2 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase -1 {
    %int = hir.int_type
    %w = uir.phase_call @Width[0] (%N) : (%int) -> (%int)
    %ty = hir.uint_type %w
    uir.phase_return yield %ty
  }
  uir.signature (%s2, %s1) -> (%s1)
} {
  uir.phase -1 {
    uir.phase_return
  }
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.add %x, %x : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

// CHECK-LABEL: uir.split_func @Width(%N: 0) -> (w: 0) {
uir.split_func @Width(%N: 0) -> (w: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @Width.impl []

uir.func @Width.impl(%N: 0) -> (w: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm1) -> (%sm1)
} {
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.mul %N, %N : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 10: the same, one phase earlier. The type of the phase -1 argument %N is
// computed by a call in the pre-span phase -2, which is the eager chain's
// zero-argument head; the call is therefore argument-free by construction.

// CHECK-LABEL: hir.func private @"pre_sig_call.-2"() -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    hir.signature () -> ([[O]])
// CHECK-NEXT:  } {
// CHECK-NEXT:    [[O2:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[INT:%.+]] = hir.int_type
// CHECK-NEXT:    [[W:%.+]] = hir.call @DefaultWidth.0() : () -> ([[INT]])
// CHECK-NEXT:    [[TY:%.+]] = hir.uint_type [[W]]
// CHECK-NEXT:    [[CTX:%.+]] = hir.opaque_pack([[TY]])
// CHECK-NEXT:    hir.return [[CTX]] : [[O2]]
// CHECK-NEXT:  }

// CHECK-LABEL: hir.func private @"pre_sig_call.-1.tail"(%N, %ctx) -> (ctx) {
// CHECK-NEXT:    [[O:%.+]] = hir.opaque_type
// CHECK-NEXT:    [[TY:%.+]] = hir.opaque_unpack %ctx : !hir.any
// CHECK-NEXT:    hir.signature ([[TY]], [[O]]) -> ([[O]])
// CHECK-NEXT:  } {

// The callee's own phase list entry is its eager_func: its pre-span link
// computes the types, its tail is the runtime phase 0 body.
// CHECK-LABEL: hir.eager_func @DefaultWidth.0() -> (w) [
// CHECK-NEXT:    @"DefaultWidth.-1",
// CHECK-NEXT:    @DefaultWidth.0.tail
// CHECK-NEXT:  ]
uir.split_func @pre_sig_call(%N: -1, %x: 0) -> () {
  %int = hir.int_type
  %w = uir.call @DefaultWidth() : () -> (%int) [] -> [0]
  %ty = hir.uint_type %w
  uir.signature (%ty, %int) -> ()
} impl @pre_sig_call.impl []

uir.func @pre_sig_call.impl(%N: -1, %x: 0) -> () {
  %s2 = uir.phase -2 {
    %int = hir.int_type
    %w = uir.phase_call @DefaultWidth[0] () : () -> (%int)
    %ty = hir.uint_type %w
    uir.phase_return yield %ty
  }
  %s1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s2, %s1) -> ()
} {
  uir.phase -1 {
    uir.phase_return
  }
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}

// CHECK-LABEL: uir.split_func @DefaultWidth() -> (w: 0) {
uir.split_func @DefaultWidth() -> (w: 0) {
  %int = hir.int_type
  uir.signature () -> (%int)
} impl @DefaultWidth.impl []

uir.func @DefaultWidth.impl() -> (w: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature () -> (%sm1)
} {
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.constant_int 8 : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Case 11: a function with no arguments and no results. Its signature has
// nothing to deliver, so the signature region holds no phase op at all, the
// merged range is the bare span {0}, and the single func needs no context
// plumbing and no eager chain.

// CHECK-LABEL: hir.func private @Bare.0() -> () {
// CHECK-NEXT:    hir.signature () -> ()
// CHECK-NEXT:  } {
// CHECK-NEXT:    hir.return
// CHECK-NEXT:  }

// CHECK-LABEL: uir.split_func @Bare() -> () {
// CHECK-NEXT:    uir.signature () -> ()
// CHECK-NEXT:  } [
// CHECK-NEXT:    0: @Bare.0
// CHECK-NEXT:  ]
uir.split_func @Bare() -> () {
  uir.signature () -> ()
} impl @Bare.impl []

uir.func @Bare.impl() -> () {
  uir.signature () -> ()
} {
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}
