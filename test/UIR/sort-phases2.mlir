// RUN: silicon-opt --sort-phases %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Straight-line, single phase: all anchored content wraps into one uir.phase
// op whose phase return carries the function result; the body terminator
// becomes uir.unreachable. Floating constants stay at the region top level,
// consumed directly from inside the phase region. The signature has no
// anchored content and gets no phase op at all.

// CHECK-LABEL: uir.func @SinglePhase
uir.func @SinglePhase(%a: 0, %b: 0) -> (result: 0) {
  // CHECK-NEXT: [[SIGT:%.+]] = hir.int_type
  %0 = hir.int_type
  // CHECK-NEXT: uir.signature ([[SIGT]], [[SIGT]]) -> ([[SIGT]])
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[SUM:%.+]] = hir.add %a, %b : [[T]]
  %sum = hir.add %a, %b : %t
  // CHECK-NEXT:   uir.phase_return [[SUM]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %sum : %t
}

//===----------------------------------------------------------------------===//
// Straight-line, two phases: phase -1 computes from the const argument and
// yields the crossing value; phase 0 consumes it as an ordinary SSA value
// and returns the function result.

// CHECK-LABEL: uir.func @MultiPhase
uir.func @MultiPhase(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[SQ:%.+]] = hir.mul %a, %a : [[T]]
  %sq = hir.mul %a, %a : %t
  // CHECK-NEXT:   uir.phase_return yield [[SQ]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = hir.add [[P]], %b : [[T]]
  %r = hir.add %sq, %b : %t
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Skip crossing: a value is yielded once, from the phase computing it, and
// consumed directly at any later phase; here the phase -2 result is consumed
// at phase -1 and again at phase 0, skipping over the -1 op.

// CHECK-LABEL: uir.func @SkipCrossing
uir.func @SkipCrossing(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -2 {
  // CHECK-NEXT:   [[SQ:%.+]] = hir.mul %a, %a : [[T]]
  %sq = hir.mul %a, %a : %t
  // CHECK-NEXT:   uir.phase_return yield [[SQ]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase -1 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.write_var [[V]], [[P]] : [[T]]
  // CHECK-NEXT:   uir.phase_return{{$}}
  // CHECK-NEXT: }
  uir.expr pin -1 {
    %v = hir.var : %t
    hir.write_var %v, %sq : %t
    uir.yield
  }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = hir.add [[P]], %x : [[T]]
  %r = hir.add %sq, %x : %t
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Result-only phase: a phase return carrying function results is content, so
// a phase whose only job is returning a value still gets its phase op.

// CHECK-LABEL: uir.func @ResultOnlyPhase
uir.func @ResultOnlyPhase(%a: -1) -> (c: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase -1 {
  // CHECK-NEXT:   uir.phase_return %a : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %a : %t
}

//===----------------------------------------------------------------------===//
// No empty phases: an externally visible phase without content (the unused
// const argument's phase -1) gets no uir.phase op.

// CHECK-LABEL: uir.func @NoEmptyPhases
uir.func @NoEmptyPhases(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: hir.int_type
  %t = hir.int_type
  // CHECK-NOT: uir.phase -1
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add %x, %x
  %r = hir.add %x, %x : %t
  // CHECK: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Direct argument crossing: a function argument consumed past its declared
// phase is used directly; its home phase -1 stays absent since it has no
// content.

// CHECK-LABEL: uir.func @ArgCrossing
uir.func @ArgCrossing(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK-NOT: uir.phase -1
  // CHECK: uir.phase 0 {
  // CHECK:   hir.add %a, %x
  %r = hir.add %a, %x : %t
  // CHECK: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Phase grouping dissolves before sorting: expr bodies are inlined and pin
// outputs are replaced by their inputs, so the interior ops wrap directly.

// CHECK-LABEL: uir.func @DissolveExprsAndPins
uir.func @DissolveExprsAndPins(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NOT: uir.expr
  // CHECK-NEXT:   [[PROD:%.+]] = hir.mul %a, %a : [[T]]
  %y = uir.expr : %t {
    %prod = hir.mul %a, %a : %t
    uir.yield %prod : %t
  }
  // CHECK-NOT: uir.pin
  %p = uir.pin %y, 0 : !hir.any
  // CHECK-NEXT:   [[SUM:%.+]] = hir.add [[PROD]], [[PROD]] : [[T]]
  %sum = hir.add %p, %p : %t
  // CHECK-NEXT:   uir.phase_return [[SUM]] : [[T]]
  uir.return %sum : %t
}

//===----------------------------------------------------------------------===//
// Zero-use pure ops are anchored at their block phase by the analysis and
// wrap like any other content, in program order.

// CHECK-LABEL: uir.func @DeadContentAnchors
uir.func @DeadContentAnchors(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  %v = hir.var : %t
  // CHECK-NEXT:   hir.write_var [[V]], %x : [[T]]
  hir.write_var %v, %x : %t
  // CHECK-NEXT:   [[R:%.+]] = hir.read_var [[V]] : [[T]]
  %r = hir.read_var %v : %t
  // CHECK-NEXT:   hir.add [[R]], [[R]] : [[T]]
  %dead = hir.add %r, %r : %t
  // CHECK-NEXT:   uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A floating op (dead pure content no demand ever reached) that uses a value
// moving into a phase region sinks into that region after its operands;
// floating ops independent of anchored content stay at the top level.

// CHECK-LABEL: uir.func @DeadFloatSinks
uir.func @DeadFloatSinks(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  %v = hir.var : %t
  // CHECK-NEXT:   hir.write_var [[V]], %x : [[T]]
  hir.write_var %v, %x : %t
  // CHECK-NEXT:   [[R:%.+]] = hir.read_var [[V]] : [[T]]
  %r = hir.read_var %v : %t
  // The dissolved expr's interior is floating (its result is never
  // demanded), uses the anchored read result, and sinks after it.
  // CHECK-NEXT:   hir.add [[R]], [[R]] : [[T]]
  %y = uir.expr : %t {
    %m = hir.add %r, %r : %t
    uir.yield %m : %t
  }
  // CHECK-NEXT:   uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// Structured control flow entirely at one phase wraps wholesale into that
// phase's op; no rule inspects its interior beyond the phase checks.

// CHECK-LABEL: uir.func @SinglePhaseIf
uir.func @SinglePhaseIf(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[IF:%.+]] = uir.if %c : [[T]] {
  %r = uir.if %c : %t {
    // CHECK-NEXT: [[A:%.+]] = hir.add %x, %x : [[T]]
    %a = hir.add %x, %x : %t
    // CHECK-NEXT: uir.yield [[A]] : [[T]]
    uir.yield %a : %t
  } else {
    // CHECK: uir.yield %x : [[T]]
    uir.yield %x : %t
  }
  // CHECK: uir.phase_return [[IF]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Independent phases without value crossings each get their own phase op,
// in increasing phase order.

// CHECK-LABEL: uir.func @IndependentPhases
uir.func @IndependentPhases(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: uir.phase -1 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.write_var [[V]], %a : [[T]]
  // CHECK-NEXT:   uir.phase_return{{$}}
  // CHECK-NEXT: }
  uir.expr pin -1 {
    %v = hir.var : %t
    hir.write_var %v, %a : %t
    uir.yield
  }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = hir.add %x, %x : [[T]]
  %r = hir.add %x, %x : %t
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// The signature region follows the same rules: anchored content wraps into
// phase ops ahead of the uir.signature terminator. A body without content
// keeps only its uir.unreachable terminator; no empty phase op appears.

// CHECK-LABEL: uir.func @SignatureContent
uir.func @SignatureContent(%x: 0) -> () {
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  %0 = hir.int_type
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  %v = hir.var : %0
  // CHECK-NEXT:   hir.write_var [[V]], %x : [[T]]
  hir.write_var %v, %x : %0
  // CHECK-NEXT:   uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[T]]) -> ()
  uir.signature (%0) -> ()
} {
  // CHECK: {
  // CHECK-NEXT: uir.unreachable
  // CHECK-NEXT: }
  uir.return
}

//===----------------------------------------------------------------------===//
// Function results at different phases: each phase's phase return carries
// exactly the function results belonging to that phase. A value may double
// as a function result and a crossing (returned in `values`, yielded in
// `yields` of the same phase return).

// CHECK-LABEL: uir.func @ResultsPerPhase
uir.func @ResultsPerPhase(%a: -1, %b: 0) -> (c: -1, d: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0, %0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[SQ:%.+]] = hir.mul %a, %a : [[T]]
  %sq = hir.mul %a, %a : %t
  // CHECK-NEXT:   uir.phase_return [[SQ]] : [[T]] yield [[SQ]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = hir.add [[P]], %b : [[T]]
  %r = hir.add %sq, %b : %t
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %sq, %r : %t, %t
}

//===----------------------------------------------------------------------===//
// A crossing consumed as a later phase's function result: the value is
// yielded by its phase and rides the `values` of the later phase's return.

// CHECK-LABEL: uir.func @ResultCrossing
uir.func @ResultCrossing(%a: -1, %b: 0) -> (c: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[SQ:%.+]] = hir.mul %a, %a : [[T]]
  %sq = hir.mul %a, %a : %t
  // CHECK-NEXT:   uir.phase_return yield [[SQ]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  %v = hir.var : %t
  // CHECK-NEXT:   hir.write_var [[V]], %b : [[T]]
  hir.write_var %v, %b : %t
  // CHECK-NEXT:   uir.phase_return [[P]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %sq : %t
}

//===----------------------------------------------------------------------===//
// Internal phases: content past the last declared phase wraps into a
// uir.phase op like any other phase, even though callers cannot target it;
// crossings into the internal phase are ordinary phase results.

// CHECK-LABEL: uir.func @TrailingInternalPhase
uir.func @TrailingInternalPhase(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   [[SQ:%.+]] = hir.mul %a, %a : [[T]]
  %sq = hir.mul %a, %a : %t
  // CHECK-NEXT:   uir.phase_return [[SQ]] : [[T]] yield [[SQ]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.write_var [[V]], [[P]] : [[T]]
  // CHECK-NEXT:   uir.phase_return{{$}}
  // CHECK-NEXT: }
  uir.expr pin 1 {
    %v = hir.var : %t
    hir.write_var %v, %sq : %t
    uir.yield
  }
  // CHECK-NEXT: uir.unreachable
  uir.return %sq : %t
}

//===----------------------------------------------------------------------===//
// Signature-region phases: a dependent signature computes the phase-0
// argument and result type from the const argument in its own phase -1 op;
// the uir.signature terminator consumes the phase result directly. The body
// recomputes the type in its own phase -1 op, and the phase 0 return's type
// operand consumes that crossing as well.

// CHECK-LABEL: uir.func @DependentSignature
uir.func @DependentSignature(%N: -1, %x: 0) -> (result: 0) {
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  %int = hir.int_type
  // CHECK-NEXT: [[SP:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[TY:%.+]] = hir.uint_type %N
  %ty = hir.uint_type %N
  // CHECK-NEXT:   uir.phase_return yield [[TY]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.signature ([[INT]], [[SP]]) -> ([[SP]])
  uir.signature (%int, %ty) -> (%ty)
} {
  // CHECK: [[BP:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[TY2:%.+]] = hir.uint_type %N
  %ty2 = hir.uint_type %N
  // CHECK-NEXT:   uir.phase_return yield [[TY2]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = hir.add %x, %x : [[BP]]
  %r = hir.add %x, %x : %ty2
  // CHECK-NEXT:   uir.phase_return [[R]] : [[BP]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %r : %ty2
}

//===----------------------------------------------------------------------===//
// A dead floating op whose operands moved into two different phase regions
// sinks into the latest one; its earlier-phase operand crosses as an
// ordinary phase result.

// CHECK-LABEL: uir.func @DeadFloatAcrossPhases
uir.func @DeadFloatAcrossPhases(%a: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[V1:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.write_var [[V1]], %a : [[T]]
  // CHECK-NEXT:   [[RD1:%.+]] = hir.read_var [[V1]] : [[T]]
  // CHECK-NEXT:   uir.phase_return yield [[RD1]]
  // CHECK-NEXT: }
  %r1 = uir.expr pin -1 : %t {
    %v1 = hir.var : %t
    hir.write_var %v1, %a : %t
    %rd1 = hir.read_var %v1 : %t
    uir.yield %rd1 : %t
  }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V2:%.+]] = hir.var : [[T]]
  %v2 = hir.var : %t
  // CHECK-NEXT:   hir.write_var [[V2]], %x : [[T]]
  hir.write_var %v2, %x : %t
  // CHECK-NEXT:   [[RD2:%.+]] = hir.read_var [[V2]] : [[T]]
  %r2 = hir.read_var %v2 : %t
  // CHECK-NEXT:   hir.add [[P]], [[RD2]] : [[T]]
  %dead = uir.expr : %t {
    %m = hir.add %r1, %r2 : %t
    uir.yield %m : %t
  }
  // CHECK-NEXT:   uir.phase_return{{$}}
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A function without content or results gets no phase op at all; the bare
// return still becomes uir.unreachable.

// CHECK-LABEL: uir.func @NoContent
uir.func @NoContent(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // CHECK: uir.signature
  // CHECK-NOT: uir.phase
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A two-phase callee, reused by the call-decomposition cases below. Its
// declaration-derived external phases are -1 and 0.

// CHECK-LABEL: uir.func @Callee
uir.func @Callee(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  %r = hir.add %sq, %b : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Call decomposition: a call to the two-phase @Callee becomes one
// uir.phase_call per externally visible callee phase. The call sits at the
// caller's phase 0 (its own analysis phase, the frame shift), so entry [-1]
// lands in caller phase -1 and entry [0] in caller phase 0. The context value
// chaining the entries crosses phases like any other value.

// CHECK-LABEL: uir.func @Caller
uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[CTX:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[C:%.+]] = uir.phase_call @Callee[-1] (%x) : ([[T]]) -> () context
  // CHECK-NEXT:   uir.phase_return yield [[C]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = uir.phase_call @Callee[0] (%y) context [[CTX]] : ([[T]]) -> ([[T]])
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = uir.call @Callee(%x, %y) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Frame-shifted call: a dyn-shifted call (wrapped in a phase-1 expr) executes
// one phase later than the callee's frame, so its analysis phase is 1. That
// single shift moves the callee's [-1] entry into caller phase 0 and its [0]
// entry into caller phase 1. The calleePhase attribute stays in the callee's
// own frame.

// CHECK-LABEL: uir.func @ShiftedCall
uir.func @ShiftedCall(%k: 0, %x: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[CTX:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   [[C:%.+]] = uir.phase_call @Callee[-1] (%k) : ([[T]]) -> () context
  // CHECK-NEXT:   uir.phase_return yield [[C]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   [[R:%.+]] = uir.phase_call @Callee[0] (%x) context [[CTX]] : ([[T]]) -> ([[T]])
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = uir.expr pin 1 : %t {
    %c = uir.call @Callee(%k, %x) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
    uir.yield %c : %t
  }
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A three-external-phase callee: its declared arg and result phases yield the
// external set {-1, 0, 1}.

// CHECK-LABEL: uir.func @ThreePhaseCallee
uir.func @ThreePhaseCallee(%a: -1, %b: 0, %c: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  %t = hir.int_type
  %s0 = hir.add %a, %b : %t
  %s1 = hir.add %s0, %c : %t
  uir.return %s1 : %t
}

//===----------------------------------------------------------------------===//
// Structurally different from @Caller: a three-external-phase callee produces
// three entries, and the middle entry (phase 0) both consumes the previous
// context and produces the next one. Both context values cross forward one
// phase.

// CHECK-LABEL: uir.func @ThreePhaseCall
uir.func @ThreePhaseCall(%x: -1, %y: 0, %z: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[CTX0:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[C0:%.+]] = uir.phase_call @ThreePhaseCallee[-1] (%x) : ([[T]]) -> () context
  // CHECK-NEXT:   uir.phase_return yield [[C0]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[CTX1:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   [[C1:%.+]] = uir.phase_call @ThreePhaseCallee[0] (%y) context [[CTX0]] : ([[T]]) -> () context
  // CHECK-NEXT:   uir.phase_return yield [[C1]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   [[R:%.+]] = uir.phase_call @ThreePhaseCallee[1] (%z) context [[CTX1]] : ([[T]]) -> ([[T]])
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = uir.call @ThreePhaseCallee(%x, %y, %z) : (%t, %t, %t) -> (%t) (!hir.any, !hir.any, !hir.any) -> !hir.any [-1, 0, 1] -> [1]
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Self-recursion: a function may call itself. Deriving the entries from the
// callee's declaration alone means no callee-before-caller ordering is
// needed, so the recursive call decomposes like any other.

// CHECK-LABEL: uir.func @Recurse
uir.func @Recurse(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[CTX:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[C:%.+]] = uir.phase_call @Recurse[-1] (%x) : ([[T]]) -> () context
  // CHECK-NEXT:   uir.phase_return yield [[C]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[R:%.+]] = uir.phase_call @Recurse[0] (%y) context [[CTX]] : ([[T]]) -> ([[T]])
  // CHECK-NEXT:   uir.phase_return [[R]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = uir.call @Recurse(%x, %y) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Branch extraction, effects: a const if with dyn content in both arms. The
// rebuilt if yields the const result and a per-arm choice; both cross to
// phase 0, where a pick dispatches on the choice and the arm-local const
// value arrives as the choice payload.

// CHECK-LABEL: uir.func @IfChoiceEffects
uir.func @IfChoiceEffects(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[P:%.+]]:2 = uir.phase -1 {
  // CHECK-NEXT:   [[V:%.+]]:2 = uir.if %c : [[T]], [[CT]] {
  // CHECK-NEXT:     [[A:%.+]] = hir.add %c, %c : [[T]]
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0([[A]])
  // CHECK-NEXT:     uir.yield [[A]], [[C0]] : [[T]], [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield %c, [[C1]] : [[T]], [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[V]]#0, [[V]]#1
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[VA:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.pick [[P]]#1 [0 ([[ARG:%.+]]): {
  // CHECK-NEXT:     hir.write_var [[VA]], [[ARG]] : [[T]]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.write_var [[VA]], %x : [[T]]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   [[OUT:%.+]] = hir.add %x, [[P]]#0 : [[T]]
  // CHECK-NEXT:   uir.phase_return [[OUT]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %va = hir.var : %t
  %r = uir.expr pin -1 : %t {
    %v = uir.if %c : %t {
      %a = hir.add %c, %c : %t
      uir.expr pin 1 {
        hir.write_var %va, %a : %t
        uir.yield
      }
      uir.yield %a : %t
    } else {
      uir.expr pin 1 {
        hir.write_var %va, %x : %t
        uir.yield
      }
      uir.yield %c : %t
    }
    uir.yield %v : %t
  }
  %out = hir.add %x, %r : %t
  uir.return %out : %t
}

//===----------------------------------------------------------------------===//
// Branch extraction, dyn result: the if's dyn-phase result selects distinct
// later-phase values, so it is removed from the if and reconstructed as a
// result of the receiving pick, whose alternatives recompute the per-arm
// values. The if keeps only the choice result.

// CHECK-LABEL: uir.func @IfChoiceResult
uir.func @IfChoiceResult(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: hir.choice_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[CH:%.+]] = uir.if %c : [[CT:%.+]] {
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[CH]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.pick [[P]] : !hir.any [0 (): {
  // CHECK-NEXT:     [[S:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:     hir.yield [[S]] : !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[M:%.+]] = hir.mul %x, %x : [[T]]
  // CHECK-NEXT:     hir.yield [[M]] : !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return [[V]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = uir.expr pin -1 : %t {
    %v = uir.if %c : %t {
      %a = uir.expr pin 1 : %t {
        %s = hir.add %x, %x : %t
        uir.yield %s : %t
      }
      uir.yield %a : %t
    } else {
      %b = uir.expr pin 1 : %t {
        %m = hir.mul %x, %x : %t
        uir.yield %m : %t
      }
      uir.yield %b : %t
    }
    uir.yield %v : %t
  }
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Branch extraction into an internal phase: a phase-0 if with later-phase
// (phase 1) content and no results. The else-less if gets a materialized
// else region for the not-taken tag, and the pick lands in the trailing
// internal phase 1.

// CHECK-LABEL: uir.func @IfLaterContentInternal
uir.func @IfLaterContentInternal(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK: hir.int_type
  %t = hir.int_type
  // CHECK: [[P:%.+]]:2 = uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : %{{.+}}
  // CHECK-NEXT:   uir.if %c : %{{.+}} {
  // CHECK-NEXT:     hir.choice 0()
  // CHECK:       } else {
  // CHECK-NEXT:     hir.choice 1()
  // CHECK:       }
  // CHECK-NEXT:   uir.phase_return yield [[V]], %{{.+}}
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   hir.pick [[P]]#1 [0 (): {
  // CHECK-NEXT:     hir.write_var [[P]]#0, %x
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %v = hir.var : %t
  uir.if %c {
    uir.expr pin 1 {
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.yield
  }
  uir.return
}

//===----------------------------------------------------------------------===//
// Nested branch extraction: a const if whose else arm holds another const if,
// each leaf carrying dyn effects. Extraction runs innermost first, so the
// inner if's choice rides the outer choice payload and the receiving pick
// nests a pick per level.

// CHECK-LABEL: uir.func @NestedIfChoice
uir.func @NestedIfChoice(%c: -1, %d: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[OUT:%.+]] = uir.if %c : %{{.+}} {
  // CHECK-NEXT:     [[OC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield [[OC0]]
  // CHECK-NEXT:   } else {
  // CHECK:         [[IN:%.+]] = uir.if %d : %{{.+}} {
  // CHECK-NEXT:       [[IC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       uir.yield [[IC0]]
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[IC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[IC1]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[OC1:%.+]] = hir.choice 1([[IN]])
  // CHECK-NEXT:     uir.yield [[OC1]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[OUT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[VA:%.+]] = hir.var
  // CHECK-NEXT:   hir.pick [[P]] [0 (): {
  // CHECK-NEXT:     hir.write_var [[VA]], %x
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   } 1 ([[ARG:%.+]]): {
  // CHECK-NEXT:     hir.pick [[ARG]] [0 (): {
  // CHECK-NEXT:       hir.write_var [[VA]], %x
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[VA]], %x
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }]
  %va = hir.var : %t
  uir.expr pin -1 {
    uir.if %c {
      uir.expr pin 1 {
        hir.write_var %va, %x : %t
        uir.yield
      }
      uir.yield
    } else {
      uir.if %d {
        uir.expr pin 1 {
          hir.write_var %va, %x : %t
          uir.yield
        }
        uir.yield
      } else {
        uir.expr pin 1 {
          hir.write_var %va, %x : %t
          uir.yield
        }
        uir.yield
      }
      uir.yield
    }
    uir.yield
  }
  %r = hir.read_var %va : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Branch extraction with a call in an arm: the dyn call decomposes into a
// uir.phase_call, which is later-phase content of the const if and moves into
// the receiving pick alongside the write.

// CHECK-LABEL: uir.func @Helper
uir.func @Helper(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %r = hir.add %a, %a : %t
  uir.return %r : %t
}

// CHECK-LABEL: uir.func @IfArmCall
uir.func @IfArmCall(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: uir.phase -1 {
  // CHECK:   uir.if %c
  // CHECK: uir.phase 0 {
  // CHECK:   [[VA:%.+]] = hir.var
  // CHECK:   hir.pick %{{.+}} [0 (): {
  // CHECK-NEXT:     [[K:%.+]] = uir.phase_call @Helper[0] (%x) : (%{{.+}}) -> (%{{.+}})
  // CHECK-NEXT:     hir.write_var [[VA]], [[K]]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }]
  %va = hir.var : %t
  uir.expr pin -1 {
    uir.if %c {
      uir.expr pin 1 {
        %k = uir.call @Helper(%x) : (%t) -> (%t) (!hir.any) -> !hir.any [0] -> [0]
        hir.write_var %va, %k : %t
        uir.yield
      }
      uir.yield
    }
    uir.yield
  }
  %r = hir.read_var %va : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Loop records: a const loop with dyn content pushes one record per activation
// onto a hit list. The loop's break check is an exit-carrying divergence, so
// each record ends in an exit-path choice: the break path an empty exited()
// link, the continued path a link carrying the iteration argument the dyn
// content samples. The list crosses to phase 0, where a replicate replays the
// records and a pick dispatches each.

// CHECK-LABEL: uir.func @LoopRecords
uir.func @LoopRecords(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: [[ZERO:%.+]] = hir.constant_int 0
  %zero = hir.constant_int 0 : %t
  // CHECK-NEXT: [[ONE:%.+]] = hir.constant_int 1
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = [[ZERO]] : [[T]]) {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[I]], %n : [[T]]
  // CHECK-NEXT:     uir.if [[DONE]] {
  // CHECK-NEXT:       [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[R0:%.+]] = hir.opaque_pack([[C0]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[R0]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], [[ONE]] : [[T]]
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1([[I]])
  // CHECK-NEXT:     [[R1:%.+]] = hir.opaque_pack([[C1]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[R1]]
  // CHECK-NEXT:     uir.continue [[NEXT]] : [[T]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]]
  // CHECK-NEXT: }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      uir.expr pin 1 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.replicate ([[REC:%.+]]) in [[P]] {
  // CHECK-NEXT:     hir.pick [[REC]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SI:%.+]]): {
  // CHECK-NEXT:       [[CUR:%.+]] = hir.read_var [[ACC]] : [[T]]
  // CHECK-NEXT:       [[NEW:%.+]] = hir.add [[CUR]], [[SI]] : [[T]]
  // CHECK-NEXT:       hir.write_var [[ACC]], [[NEW]] : [[T]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[RD:%.+]] = hir.read_var [[ACC]] : [[T]]
  // CHECK-NEXT:   uir.phase_return [[RD]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Loop without a divergence: a const loop whose body runs dyn content
// unconditionally (no break) pushes one empty record per activation, and the
// receiving replicate takes no record fields. Structurally unlike @LoopRecords
// (no exit-path choice), it exercises the no-divergence record path.

// CHECK-LABEL: uir.func @LoopNoDivergence
uir.func @LoopNoDivergence(%n: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  %v = hir.var : %t
  // CHECK: [[P:%.+]]:2 = uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop {
  // CHECK-NEXT:     [[REC:%.+]] = hir.opaque_pack()
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[REC]]
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[V]], [[LIST]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   hir.replicate () in [[P]]#1 {
  // CHECK-NEXT:     hir.write_var [[P]]#0, %x : [[T]]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.loop {
    uir.expr pin 1 {
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.continue
  }
  uir.return
}

//===----------------------------------------------------------------------===//
// Exit-path prefix samples: dyn content runs before the break check, so every
// activation (break and continue alike) samples the iteration argument as a
// record prefix. Both exit-path choices carry empty payloads, since no dyn
// content follows the divergence.

// CHECK-LABEL: uir.func @LoopExitAfterContent
uir.func @LoopExitAfterContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[I]], %n
  // CHECK-NEXT:     uir.if [[DONE]] {
  // CHECK-NEXT:       [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[R0:%.+]] = hir.opaque_pack([[I]], [[C0]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[R0]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     [[R1:%.+]] = hir.opaque_pack([[I]], [[C1]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[R1]]
  // CHECK-NEXT:     uir.continue [[NEXT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[VV:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[SI:%.+]], [[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.write_var [[VV]], [[SI]]
  // CHECK-NEXT:     hir.pick [[REC]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %v, %i : %t
        uir.yield
      }
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Nested loops: nested hit lists fall out with no special casing. The inner
// loop's list is created per outer activation and crosses through the outer
// record's continued link like any other snapshot; the receiving replicate
// nests a replicate per level.

// CHECK-LABEL: uir.func @NestedLoops
uir.func @NestedLoops(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[OUTER:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK:          [[INNER:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:     uir.loop ([[J:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK:            hir.choice 1([[J]])
  // CHECK:          }
  // CHECK-NEXT:     [[OC1:%.+]] = hir.choice 1([[INNER]])
  // CHECK-NEXT:     [[OREC:%.+]] = hir.opaque_pack([[OC1]])
  // CHECK-NEXT:     hir.opaque_list_push [[OUTER]], [[OREC]]
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[OUTER]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[VACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[OREC2:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[OREC2]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[INNER2:%.+]]): {
  // CHECK-NEXT:       hir.replicate ([[IREC:%.+]]) in [[INNER2]] {
  // CHECK-NEXT:         hir.pick [[IREC]] [0 (): {
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         } 1 ([[SJ:%.+]]): {
  // CHECK-NEXT:           [[CUR:%.+]] = hir.read_var [[VACC]]
  // CHECK-NEXT:           [[NEW:%.+]] = hir.add [[CUR]], [[SJ]]
  // CHECK-NEXT:           hir.write_var [[VACC]], [[NEW]]
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         }]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %d0 = hir.geq %i, %n : %t
      uir.if %d0 { uir.break }
      %ni = hir.add %i, %one : %t
      uir.loop (%j = %zero : %t) {
        %d1 = hir.geq %j, %n : %t
        uir.if %d1 { uir.break }
        %nj = hir.add %j, %one : %t
        uir.expr pin 1 {
          %cur = hir.read_var %acc : %t
          %s = hir.add %cur, %j : %t
          hir.write_var %acc, %s : %t
          uir.yield
        }
        uir.continue %nj : %t
      }
      uir.continue %ni : %t
    }
    uir.yield
  }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}
