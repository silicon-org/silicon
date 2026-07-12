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
