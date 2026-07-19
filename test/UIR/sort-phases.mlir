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
// Dead pure ops derive their phase from their operands in the analysis and
// wrap like any other content, in program order: sorting is demand-blind.

// CHECK-LABEL: uir.func @DeadContentDerived
uir.func @DeadContentDerived(%x: 0) -> () {
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
// The interior of a dissolved dead expr derives its phase from its operands
// and is placed like live content; only operand-free dead trees stay
// floating at the top level.

// CHECK-LABEL: uir.func @DeadExprInteriorPlaced
uir.func @DeadExprInteriorPlaced(%x: 0) -> () {
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
  // The dissolved expr's interior is dead (its result is never demanded)
  // but derives phase 0 from the read result and wraps after it.
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
// A dead pure op whose operands moved into two different phase regions
// derives the latest operand phase and wraps there; its earlier-phase
// operand crosses as an ordinary phase result.

// CHECK-LABEL: uir.func @DeadContentAcrossPhases
uir.func @DeadContentAcrossPhases(%a: -1, %x: 0) -> () {
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
// Dead later-phase content nested inside earlier-phase control flow extracts
// exactly like live content: the branch choice-ifies, and the dead phase-1
// op lands in the receiving pick's matching arm.

// CHECK-LABEL: uir.func @DeadContentInBranch
uir.func @DeadContentInBranch(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  %m = uir.expr pin 1 : %t {
    %v = hir.var : %t
    hir.write_var %v, %x : %t
    %rd = hir.read_var %v : %t
    uir.yield %rd : %t
  }
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[CH:%.+]] = uir.phase 0 {
  // CHECK:   [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK:     [[C0:%.+]] = hir.choice 0()
  // CHECK:     uir.yield [[C0]] : [[CT]]
  // CHECK:   } else {
  // CHECK:     [[C1:%.+]] = hir.choice 1()
  // CHECK:     uir.yield [[C1]] : [[CT]]
  // CHECK:   }
  // CHECK:   uir.phase_return yield [[IF]]
  // CHECK: }
  uir.if %c {
    %y = uir.expr : %t {
      %dead = hir.add %m, %m : %t
      uir.yield %dead : %t
    }
    uir.yield
  }
  // CHECK: uir.phase 1 {
  // CHECK:   [[V:%.+]] = hir.var : [[T]]
  // CHECK:   hir.write_var [[V]], %x : [[T]]
  // CHECK:   [[RD:%.+]] = hir.read_var [[V]] : [[T]]
  // CHECK:   hir.pick [[CH]] [0 (): {
  // CHECK:     hir.add [[RD]], [[RD]] : [[T]]
  // CHECK:     hir.yield
  // CHECK:   } 1 (): {
  // CHECK:     hir.yield
  // CHECK:   }]
  // CHECK:   uir.phase_return
  // CHECK: }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A fully dead if inside a dead floating expr anchors at its derived phase
// (the condition's) and extracts exactly like live content: the dead result
// choice-ifies, and the later-phase arm values reconstruct in the receiving
// pick, unused there.

// CHECK-LABEL: uir.func @DeadIfInDeadExpr
uir.func @DeadIfInDeadExpr(%c: 0, %a: 1) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[CH:%.+]] = uir.phase 0 {
  // CHECK:   [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK:     [[C0:%.+]] = hir.choice 0()
  // CHECK:     uir.yield [[C0]] : [[CT]]
  // CHECK:   } else {
  // CHECK:     [[C1:%.+]] = hir.choice 1()
  // CHECK:     uir.yield [[C1]] : [[CT]]
  // CHECK:   }
  // CHECK:   uir.phase_return yield [[IF]]
  // CHECK: }
  // CHECK: uir.phase 1 {
  // CHECK:   hir.pick [[CH]] : !hir.any [0 (): {
  // CHECK:     [[V:%.+]] = hir.add %a, %a : [[T]]
  // CHECK:     hir.yield [[V]] : !hir.any
  // CHECK:   } 1 (): {
  // CHECK:     [[W:%.+]] = hir.mul %a, %a : [[T]]
  // CHECK:     hir.yield [[W]] : !hir.any
  // CHECK:   }]
  // CHECK:   uir.phase_return
  // CHECK: }
  %r = uir.expr : %t {
    %s = uir.if %c : %t {
      %v = hir.add %a, %a : %t
      uir.yield %v : %t
    } else {
      %w = hir.mul %a, %a : %t
      uir.yield %w : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A fully dead loop inside a dead expr sorts exactly like its live twin: the
// expr defaults to block phase 0 and dissolves, the loop wraps at phase 0
// pushing a per-activation record, and the later-phase add moves into a
// phase-1 pick that replays it. Dead content sorts demand-blind.

// CHECK-LABEL: uir.func @DeadLoopInDeadExpr
uir.func @DeadLoopInDeadExpr(%a: 1) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  %t = hir.int_type
  // CHECK: [[L:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   uir.loop
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   hir.pick [[L]]
  // CHECK: hir.add %a, %a
  %r = uir.expr : %t {
    %s = uir.loop : %t {
      %x = hir.add %a, %a : %t
      uir.break %x : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// Same for a dead if whose condition is a floating constant: the frame
// defaults to block phase 0 and the if is choice-ified there, with the
// later-phase arm content moving into a phase-1 pick — the live twin's shape.

// CHECK-LABEL: uir.func @DeadIfConstCondInDeadExpr
uir.func @DeadIfConstCondInDeadExpr(%a: 1) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  %t = hir.int_type
  %c = hir.constant_int 1 : %t
  // CHECK: [[CH:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   uir.if
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   hir.pick [[CH]]
  // CHECK: hir.add %a, %a
  %r = uir.expr : %t {
    %s = uir.if %c : %t {
      %v = hir.add %a, %a : %t
      uir.yield %v : %t
    } else {
      uir.yield %a : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A live conduit expr with a bystander dead loop: both sort as ordinary
// phase-0 content. The bystander loop is choice-ified into phase 0 with its
// later-phase add replayed by a phase-1 pick, alongside the conduit's own
// phase-0 result.

// CHECK-LABEL: uir.func @LiveConduitWithBystanderLoop
uir.func @LiveConduitWithBystanderLoop(%x: 0, %a: 1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   uir.loop
  // CHECK:        [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   uir.phase_return [[Y]] : [[T]] yield
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   hir.pick
  // CHECK:        hir.add %a, %a
  %r = uir.expr : %t {
    %s = uir.loop : %t {
      %v = hir.add %a, %a : %t
      uir.break %v : %t
    }
    %y = hir.add %x, %x : %t
    uir.yield %y : %t
  }
  // CHECK: uir.unreachable
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A dead frame nested inside live anchored content sorts like its live twin:
// the enclosing if wraps at phase 0, the dead inner if is choice-ified there
// (nested inside the outer choice), and both later-phase halves move into the
// phase-1 pick.

// CHECK-LABEL: uir.func @FloatFrameInLiveIf
uir.func @FloatFrameInLiveIf(%c: 0, %a: 1) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %fc = hir.constant_int 1 : %t
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   uir.if %c
  // CHECK:          uir.if
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   hir.pick
  // CHECK:        hir.pick
  // CHECK:        hir.add %a, %a
  uir.if %c {
    %dead = uir.expr : %t {
      %s = uir.if %fc : %t {
        %v = hir.add %a, %a : %t
        uir.yield %v : %t
      } else {
        uir.yield %a : %t
      }
      uir.yield %s : %t
    }
    uir.yield
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A dead expr holding a break that targets the enclosing loop is anchored at
// the loop's phase by the analysis (the break crosses the expr frame), so it
// is no longer a wholesale float: it dissolves and wraps in place at phase 0
// like any other content, with no extraction.

// CHECK-LABEL: uir.func @DeadExprOutBreakSorts
uir.func @DeadExprOutBreakSorts(%x: 0, %c: 0) -> (result: 0) {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> (%t0)
} {
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   [[L:%.+]] = uir.loop (%arg0 = %x
  // CHECK-NEXT:     uir.if %c
  // CHECK-NEXT:       uir.break %arg0
  // CHECK:          uir.yield %arg0
  // CHECK:        uir.continue %arg0
  // CHECK:   uir.phase_return [[L]]
  %r = uir.loop (%i = %x : %t) : %t {
    %dead = uir.expr : %t {
      %s = uir.if %c : %t {
        uir.break %i : %t
      } else {
        uir.yield %i : %t
      }
      uir.yield %s : %t
    }
    uir.continue %i : %t
  }
  // CHECK: uir.unreachable
  // CHECK-NOT: hir.pick
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A continue value bounds its frame through a pinned sub-expr: the phase-1
// value fed back into the loop's iteration argument forces the whole dead
// frame to phase 1, and it sorts in place there — identical to the same
// flow without the pin wrapper.

// CHECK-LABEL: uir.func @ContinueValueThroughPinSorts
uir.func @ContinueValueThroughPinSorts(%p0: 0, %p1: 1, %c: 0) -> () {
  %t0 = hir.int_type
  uir.signature (%t0, %t0, %t0) -> ()
} {
  %t = hir.int_type
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   uir.loop (%arg0 = %p0
  // CHECK-NEXT:     uir.if %c
  // CHECK-NEXT:       [[J:%.+]] = hir.add %arg0, %p1
  // CHECK-NEXT:       uir.continue [[J]]
  // CHECK:          uir.yield %arg0
  // CHECK:        uir.break
  // CHECK:   uir.phase_return
  %r = uir.expr : %t {
    %s = uir.loop (%i = %p0 : %t) : %t {
      %p = uir.expr pin : %t {
        %sel = uir.if %c : %t {
          %j = hir.add %i, %p1 : %t
          uir.continue %j : %t
        } else {
          uir.yield %i : %t
        }
        uir.yield %sel : %t
      }
      uir.break %p : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A nested floating expr locked to the outer frame by an escaping break
// anchors with it and sorts in place: the whole structure wraps at the
// loop's phase 0 with the nested expr dissolved.

// CHECK-LABEL: uir.func @NestedFrameExitToOuterLoopSorts
uir.func @NestedFrameExitToOuterLoopSorts(%p0: 0, %c: 0) -> () {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> ()
} {
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   uir.loop (%arg0 = %p0
  // CHECK-NEXT:     uir.if %c
  // CHECK-NEXT:       uir.break %arg0
  // CHECK:          uir.yield %arg0
  // CHECK:        uir.continue %arg0
  // CHECK:   uir.phase_return
  %r = uir.expr : %t {
    %s = uir.loop (%i = %p0 : %t) : %t {
      %n = uir.expr : %t {
        %sel = uir.if %c : %t {
          uir.break %i : %t
        } else {
          uir.yield %i : %t
        }
        uir.yield %sel : %t
      }
      uir.continue %i : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A nested dead expr derives independently: the inner frame anchors at the
// if condition's phase 1 while the outer stays float. Sorting extracts the
// inner if into phase 1 and the outer expr dissolves entirely (its yield of
// the inner result is a plain forward, leaving nothing behind).

// CHECK-LABEL: uir.func @NestedDeadExpr
uir.func @NestedDeadExpr(%d: 1, %x: 1) -> () {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> ()
} {
  %t = hir.int_type
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   uir.if %d
  // CHECK-NEXT:     uir.yield %x
  // CHECK:        uir.yield %x
  // CHECK:   uir.phase_return
  // CHECK: }
  %outer = uir.expr : %t {
    %inner = uir.expr : %t {
      %s = uir.if %d : %t {
        uir.yield %x : %t
      } else {
        uir.yield %x : %t
      }
      uir.yield %s : %t
    }
    uir.yield %inner : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// A dead frame mixing a phase-0 var read with the phase-1 arg defaults to
// block phase 0 (no internal-crossing rule sinks it). The loop wraps at
// phase 0, the var read crosses into phase 1 as a plain phase-op result, and
// the interior add is replayed by a phase-1 pick where it consumes the
// crossing — the same shape as a live loop with later-phase content.

// CHECK-LABEL: uir.func @DeadFrameDistributesInterior
uir.func @DeadFrameDistributesInterior(%x: 0, %a: 1) -> () {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> ()
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: [[PH0:%.+]]:2 = uir.phase 0 {
  // CHECK-NEXT:   [[V:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.write_var [[V]], %x : [[T]]
  // CHECK-NEXT:   [[R:%.+]] = hir.read_var [[V]] : [[T]]
  // CHECK-NEXT:   uir.loop
  // CHECK:        uir.phase_return yield [[R]]
  %v = hir.var : %t
  hir.write_var %v, %x : %t
  %r = hir.read_var %v : %t
  // CHECK: uir.phase 1 {
  // CHECK-NEXT:   hir.pick [[PH0]]#1
  // CHECK:          hir.add [[PH0]]#0, %a : [[T]]
  %dead = uir.expr : %t {
    %s = uir.loop : %t {
      %m = hir.add %r, %a : %t
      uir.break %m : %t
    }
    uir.yield %s : %t
  }
  // CHECK: uir.unreachable
  uir.return
}

//===----------------------------------------------------------------------===//
// Extracted content referencing a floating tree defined inside the arm gets
// a clone of the whole tree in front of it (arm-interior floats are not
// visible from the receiving pick); values defined outside the arm are used
// directly. Dead extracted content is cloned for exactly like live content.

// CHECK-LABEL: uir.func @ClonedFloatTree
uir.func @ClonedFloatTree(%c: 0, %a: 1) -> (result: 1) {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> (%t0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 1 {
  // CHECK:   hir.pick {{%.+}} : !hir.any [0 (): {
  // CHECK:     [[ONE:%.+]] = hir.constant_int 1 : [[T]]
  // CHECK:     [[TWO:%.+]] = hir.add [[ONE]], [[ONE]] : [[T]]
  // CHECK:     hir.mul [[TWO]], %a : [[T]]
  // CHECK:     [[LIVE:%.+]] = hir.add %a, %a : [[T]]
  // CHECK:     hir.yield [[LIVE]] : !hir.any
  // CHECK:   } 1 (): {
  // CHECK:     [[W:%.+]] = hir.mul %a, %a : [[T]]
  // CHECK:     hir.yield [[W]] : !hir.any
  // CHECK:   }]
  %r = uir.if %c : %t {
    %one = hir.constant_int 1 : %t
    %two = hir.add %one, %one : %t
    %dead = hir.mul %two, %a : %t
    %live = hir.add %a, %a : %t
    uir.yield %live : %t
  } else {
    %w = hir.mul %a, %a : %t
    uir.yield %w : %t
  }
  uir.return %r : %t
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
// Branch extraction, same-value result: both arms yield the same later-phase
// SSA value. Two providers are a selection regardless of the values'
// identity, so the result choice-ifies exactly like distinct values: the if
// yields a per-arm choice and the receiving pick reconstructs the value.

// CHECK-LABEL: uir.func @IfSameValueChoice
uir.func @IfSameValueChoice(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: uir.signature
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK-NEXT: hir.choice_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase 0 {
  // CHECK-NEXT:   [[CH:%.+]] = uir.if %c : [[CT:%.+]] {
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[CH]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase 1 {
  // CHECK-NEXT:   [[V:%.+]] = hir.pick [[P]] : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield %y : !hir.any
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.yield %y : !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return [[V]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  %v = uir.if %c : %t {
    uir.yield %y : %t
  } else {
    uir.yield %y : %t
  }
  uir.return %v : %t
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

//===----------------------------------------------------------------------===//
// Break-value choice: a const loop whose break carries a dyn value is rebuilt
// at phase -1 breaking with a choice of the value's snapshots instead; the
// choice crosses to phase 0, where a pick reconstructs the value from the
// snapshots. Matches @BreakValueChoice in sorted-form.mlir.

// CHECK-LABEL: uir.func @BreakValueChoice
uir.func @BreakValueChoice(%n: -1, %x: 0) -> (result: 0) {
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
  // CHECK-NEXT: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[P:%.+]] = uir.phase -1 {
  // CHECK-NEXT:   [[CH:%.+]] = uir.loop ([[K:%.+]] = [[ZERO]] : [[T]]) : [[CT]] {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[K]], %n : [[T]]
  // CHECK-NEXT:     uir.if [[DONE]] {
  // CHECK-NEXT:       [[C0:%.+]] = hir.choice 0([[K]])
  // CHECK-NEXT:       uir.break [[C0]] : [[CT]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[K]], [[ONE]] : [[T]]
  // CHECK-NEXT:     uir.continue [[NEXT]] : [[T]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[CH]]
  // CHECK-NEXT: }
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %s = hir.add %x, %k : %t
        uir.break %s : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[Y:%.+]] = hir.pick [[P]] : !hir.any [0 ([[SK:%.+]]): {
  // CHECK-NEXT:     [[S:%.+]] = hir.add %x, [[SK]] : [[T]]
  // CHECK-NEXT:     hir.yield [[S]] : !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.phase_return [[Y]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.unreachable
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Two break sites, each carrying a distinct dyn value: one choice, two tags,
// two pick alternatives. Structurally unlike the single-break case.

// CHECK-LABEL: uir.func @TwoBreakValues
uir.func @TwoBreakValues(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[CH:%.+]] = uir.loop (
  // CHECK:   hir.choice 0([[K0:%.+]])
  // CHECK:   uir.break
  // CHECK:   hir.choice 1([[K1:%.+]])
  // CHECK:   uir.break
  // CHECK: }
  // CHECK: uir.phase_return yield [[CH]]
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %s = hir.add %x, %k : %t
        uir.break %s : %t
      }
      %big = hir.geq %k, %one : %t
      uir.if %big {
        %m = hir.mul %x, %k : %t
        uir.break %m : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK: hir.pick %{{.+}} : !hir.any [0 ([[SK0:%.+]]): {
  // CHECK-NEXT:   hir.add %x, [[SK0]]
  // CHECK-NEXT:   hir.yield
  // CHECK-NEXT: } 1 ([[SK1:%.+]]): {
  // CHECK-NEXT:   hir.mul %x, [[SK1]]
  // CHECK-NEXT:   hir.yield
  // CHECK-NEXT: }]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Two break sites carrying the same dyn SSA value, defined outside the loop:
// two providers are a selection regardless of the values' identity, so the
// value choice-ifies exactly like distinct break values. With the value
// defined outside, the loop body holds no later-phase content at all — the
// later-phase result alone triggers the extraction, and the pick alternatives
// reference the value directly.

// CHECK-LABEL: uir.func @SameBreakValueChoice
uir.func @SameBreakValueChoice(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[CH:%.+]] = uir.loop (
  // CHECK:   hir.choice 0()
  // CHECK:   uir.break
  // CHECK:   hir.choice 1()
  // CHECK:   uir.break
  // CHECK: }
  // CHECK: uir.phase_return yield [[CH]]
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        uir.break %x : %t
      }
      %big = hir.geq %k, %one : %t
      uir.if %big {
        uir.break %x : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK: hir.pick %{{.+}} : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield %x : !hir.any
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.yield %x : !hir.any
  // CHECK-NEXT: }]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Mixed kept and later results with the result-only trigger: result 0 is a
// const value at the loop's own phase (kept on the rebuilt loop), result 1 a
// dyn value defined outside the loop (choice-ified). The body holds no
// later-phase content, so the later result alone triggers the extraction.

// CHECK-LABEL: uir.func @MixedKeptLater
uir.func @MixedKeptLater(%n: -1, %x: 0) -> (rc: -1, rd: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0, %0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[CH:%.+]]:2 = uir.loop (
  // CHECK:   hir.choice 0()
  // CHECK:   uir.break {{%.+}}, {{%.+}} :
  // CHECK:   hir.choice 1()
  // CHECK:   uir.break {{%.+}}, {{%.+}} :
  // CHECK: }
  // CHECK: uir.phase_return [[CH]]#0 : {{%.+}} yield [[CH]]#1
  %a, %b = uir.expr pin -1 : %t, %t {
    %r0, %r1 = uir.loop (%k = %zero : %t) : %t, %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        uir.break %zero, %x : %t, %t
      }
      %big = hir.geq %k, %one : %t
      uir.if %big {
        uir.break %one, %x : %t, %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r0, %r1 : %t, %t
  }
  // CHECK: hir.pick %{{.+}} : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield %x : !hir.any
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.yield %x : !hir.any
  // CHECK-NEXT: }]
  uir.return %a, %b : %t, %t
}

//===----------------------------------------------------------------------===//
// Nested loops with the result-only trigger cascading: the inner const loop
// breaks with a dyn value defined outside both loops (single break site, no
// later body content). Inner extraction plants a receiving pick inside the
// outer loop's body; the outer loop's break then carries that pick's dyn
// result, so the outer loop break-value-extracts with the inner pick as its
// slice, and the receiving picks nest at phase 0.

// CHECK-LABEL: uir.func @NestedResultOnly
uir.func @NestedResultOnly(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[OCH:%.+]] = uir.loop (
  // CHECK:   [[ICH:%.+]] = uir.loop (
  // CHECK:     hir.choice 0()
  // CHECK:     uir.break
  // CHECK:   }
  // CHECK:   [[OC0:%.+]] = hir.choice 0([[ICH]])
  // CHECK:   uir.break [[OC0]] :
  // CHECK: }
  // CHECK: uir.phase_return yield [[OCH]]
  %y = uir.expr pin -1 : %t {
    %o = uir.loop (%k = %zero : %t) : %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %inner = uir.loop (%j = %zero : %t) : %t {
          %jdone = hir.geq %j, %k : %t
          uir.if %jdone {
            uir.break %x : %t
          }
          %jnext = hir.add %j, %one : %t
          uir.continue %jnext : %t
        }
        uir.break %inner : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %o : %t
  }
  // CHECK: hir.pick %{{.+}} : !hir.any [0 ([[ARG:%.+]]): {
  // CHECK-NEXT:   [[IV:%.+]] = hir.pick [[ARG]] : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield %x : !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   hir.yield [[IV]] : !hir.any
  // CHECK-NEXT: }]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Break value composed with separate record content: the loop both accumulates
// a dyn var per iteration (record content) and breaks with a dyn value. The
// break value choice-ifies while the record content extracts into a replicate;
// the reconstructing pick runs after the replicate (all iterations first, then
// the final break value). Structurally unlike @BreakValueChoice.

// CHECK-LABEL: uir.func @BreakValueWithRecords
uir.func @BreakValueWithRecords(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // The phase -1 op yields both the hit list and the break-value choice.
  // CHECK: [[P:%.+]]:2 = uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   [[CH:%.+]] = uir.loop ([[K:%.+]] = %{{.+}} : [[T:%.+]]) : %{{.+}} {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[K]], %n : [[T]]
  // CHECK-NEXT:     uir.if [[DONE]] {
  // The break site pushes both a break-value choice and an exit-path record.
  // CHECK-NEXT:       [[BV:%.+]] = hir.choice 0([[K]])
  // CHECK-NEXT:       [[EX:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[RE:%.+]] = hir.opaque_pack([[K]], [[EX]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RE]]
  // CHECK-NEXT:       uir.break [[BV]] :
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[K]], %{{.+}} : [[T]]
  // CHECK-NEXT:     [[CO:%.+]] = hir.choice 1()
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[K]], [[CO]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[RC]]
  // CHECK-NEXT:     uir.continue [[NEXT]] : [[T]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]], [[CH]]
  // CHECK-NEXT: }
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      uir.expr pin 1 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %k : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %s = hir.add %x, %k : %t
        uir.break %s : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // The replicate replays the record content; the pick reconstructs the break
  // value afterwards, and the result combines both.
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var : [[T]]
  // CHECK-NEXT:   hir.replicate ([[SAMP:%.+]], [[CHAIN:%.+]]) in [[P]]#0 {
  // CHECK:          hir.pick [[CHAIN]] [0 (): {
  // CHECK:          } 1 (): {
  // CHECK:          }]
  // CHECK:        }
  // CHECK:        [[BVR:%.+]] = hir.pick [[P]]#1 : !hir.any [0 ([[SK:%.+]]): {
  // CHECK-NEXT:     hir.add %x, [[SK]] : [[T]]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }]
  // CHECK:        [[RD:%.+]] = hir.read_var [[ACC]] : [[T]]
  // CHECK-NEXT:   [[SUM:%.+]] = hir.add [[BVR]], [[RD]] : [[T]]
  // CHECK-NEXT:   uir.phase_return [[SUM]] : [[T]]
  // CHECK-NEXT: }
  %rd = hir.read_var %acc : %t
  %sum = hir.add %y, %rd : %t
  uir.return %sum : %t
}

//===----------------------------------------------------------------------===//
// Rejoining-arm content: a divergence with later-phase content in both arms
// carries a cf choice tagging which arm ran, with the arm's samples as its
// payload. The exit arm builds its cf (and the NOT_REACHED reach link) at its
// break; the rejoining arm yields its cf, which becomes the rebuilt branch's
// result and rides the back-edge record next to the REACHED reach link. The
// rejoining arm samples an arm-interior phase -1 value (%m), which its cf tag
// captures inside the arm.

// CHECK-LABEL: uir.func @RejoiningArmContent
uir.func @RejoiningArmContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[I]], %n
  // CHECK-NEXT:     hir.choice_type
  // CHECK-NEXT:     [[CF:%.+]] = uir.if [[DONE]] : %{{.+}} {
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0([[I]])
  // CHECK-NEXT:       [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[RE:%.+]] = hir.opaque_pack([[CF0]], [[NR]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RE]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[M:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[M]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:     [[R:%.+]] = hir.choice 1([[I]])
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[CF]], [[R]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[RC]]
  // CHECK-NEXT:     uir.continue [[NEXT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]]
  // CHECK-NEXT: }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.break
      } else {
        %m = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %m : %t
          uir.yield
        }
        uir.yield
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The receiving replicate replays by data alone: the cf pick dispatches the
  // arm that ran, then the reach pick gates the following segment.
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[SI:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SI]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SM:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SM]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SE:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SE]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[RD:%.+]] = hir.read_var [[ACC]]
  // CHECK-NEXT:   uir.phase_return [[RD]]
  // CHECK-NEXT: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// An exit-carrying divergence whose rejoining arm yields a demanded later-phase
// result. The result is the value the rejoining arm's later content produces,
// so the cf pick that replays that content is made valued: the rejoining arm's
// alternative yields the result, the exit arm's alternative yields an
// `uir.unreachable_value` (it provides none), and the following segment —
// gated by the reach pick, so it never consumes the unreachable value —
// writes the pick's result.

// CHECK-LABEL: uir.func @DivergenceLaterResult
uir.func @DivergenceLaterResult(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // The rebuilt if drops the later result and yields only the cf choice; the
  // exit arm pushes its own record and breaks.
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[I]], %n
  // CHECK-NEXT:     hir.choice_type
  // CHECK-NEXT:     [[CF:%.+]] = uir.if [[DONE]] : %{{.+}} {
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[RE:%.+]] = hir.opaque_pack([[CF0]], [[NR]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RE]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:     [[R:%.+]] = hir.choice 1()
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[CF]], [[R]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[RC]]
  // CHECK-NEXT:     uir.continue [[NEXT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]]
  // CHECK-NEXT: }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      %r = uir.if %done : %t {
        uir.break
      } else {
        %d = uir.expr pin 1 : %t {
          %s = hir.add %x, %x : %t
          uir.yield %s : %t
        }
        uir.yield %d : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %r : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The valued cf pick reconstructs the later result; the reach pick gates
  // the consuming segment.
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     [[RES:%.+]] = hir.pick [[CFR]] : !hir.any [0 (): {
  // CHECK-NEXT:       [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       hir.yield [[UV]] : !hir.any
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       [[S:%.+]] = hir.add %x, %x
  // CHECK-NEXT:       hir.yield [[S]] : !hir.any
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[RES]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[RD:%.+]] = hir.read_var [[ACC]]
  // CHECK-NEXT:   uir.phase_return [[RD]]
  // CHECK-NEXT: }
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

//===----------------------------------------------------------------------===//
// A demanded later result whose producer sits after a nested exit in the
// rejoining arm. The producer replays only on the arm's fall-through path, so
// the arm's reach pick is made valued: its NOT_REACHED alternative yields
// `uir.unreachable_value` (the activation exited at the nested exit) and its
// REACHED alternative forwards the producer's value. The arm's cf alternative
// yields the reach pick's result, so the value threads up to the divergence's
// cf pick and reconstructs the demanded result.

// CHECK-LABEL: uir.func @LaterResultAfterNestedExit
uir.func @LaterResultAfterNestedExit(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK-NEXT:     [[C:%.+]] = hir.geq [[I]], %n
  // CHECK-NEXT:     hir.choice_type
  // The exit arm pushes its own record and breaks.
  // CHECK-NEXT:     [[CF:%.+]] = uir.if [[C]] : %{{.+}} {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK-NEXT:       hir.choice 0()
  // CHECK-NEXT:       [[RA:%.+]] = hir.opaque_pack
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RA]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[D:%.+]] = hir.geq %n, [[I]]
  // The nested exit pushes a record: cf tag 1 wrapping a NOT_REACHED arm link,
  // then the divergence's NOT_REACHED link.
  // CHECK-NEXT:       uir.if [[D]] {
  // CHECK-NEXT:         [[IB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[ICF:%.+]] = hir.choice 1([[IB]])
  // CHECK-NEXT:         [[INR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[RB:%.+]] = hir.opaque_pack([[ICF]], [[INR]])
  // CHECK-NEXT:         hir.opaque_list_push [[LIST]], [[RB]]
  // CHECK-NEXT:         uir.break
  // CHECK-NEXT:       }
  // The arm's fall-through cf: tag 1 wrapping a REACHED arm link.
  // CHECK-NEXT:       [[RE:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[ACF:%.+]] = hir.choice 1([[RE]])
  // CHECK-NEXT:       uir.yield [[ACF]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:     [[BR:%.+]] = hir.choice 1()
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[CF]], [[BR]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]], [[RC]]
  // CHECK-NEXT:     uir.continue [[NEXT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   uir.phase_return yield [[LIST]]
  // CHECK-NEXT: }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      %r = uir.if %c : %t {
        uir.break
      } else {
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %v = uir.expr pin 1 : %t {
          %s = hir.add %x, %x : %t
          uir.yield %s : %t
        }
        uir.yield %v : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %r : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The valued cf pick reconstructs the later result. The rejoining arm's
  // alternative dispatches its nested reach pick — NOT_REACHED an
  // `uir.unreachable_value`, REACHED the producer — and yields the result.
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     [[RES:%.+]] = hir.pick [[CFR]] : !hir.any [0 (): {
  // CHECK-NEXT:       [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       hir.yield [[UV]] : !hir.any
  // CHECK-NEXT:     } 1 ([[ALR:%.+]]): {
  // CHECK-NEXT:       [[RP:%.+]] = hir.pick [[ALR]] : !hir.any [0 (): {
  // CHECK-NEXT:         [[UV2:%.+]] = uir.unreachable_value
  // CHECK-NEXT:         hir.yield [[UV2]] : !hir.any
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         [[S:%.+]] = hir.add %x, %x
  // CHECK-NEXT:         hir.yield [[S]] : !hir.any
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield [[RP]] : !hir.any
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[RES]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[RD:%.+]] = hir.read_var [[ACC]]
  // CHECK-NEXT:   uir.phase_return [[RD]]
  // CHECK-NEXT: }
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

//===----------------------------------------------------------------------===//
// The producer sits after two stacked nested exits. The value threads up
// through both reach picks: each level's NOT_REACHED yields an
// `uir.unreachable_value` and each REACHED forwards the value from the level
// below, so the reach picks nest one valued pick per gated segment.

// CHECK-LABEL: uir.func @LaterResultTwoStackedExits
uir.func @LaterResultTwoStackedExits(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      %r = uir.if %c : %t {
        uir.break
      } else {
        %d1 = hir.geq %n, %i : %t
        uir.if %d1 { uir.break }
        %d2 = hir.geq %i, %n : %t
        uir.if %d2 { uir.break }
        %v = uir.expr pin 1 : %t {
          %s = hir.add %x, %x : %t
          uir.yield %s : %t
        }
        uir.yield %v : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %r : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // Two nested valued reach picks forward the producer up to the arm's yield.
  // CHECK:      hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:   [[RES:%.+]] = hir.pick [[CFR]] : !hir.any [0 (): {
  // CHECK-NEXT:     [[UV0:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     hir.yield [[UV0]] : !hir.any
  // CHECK-NEXT:   } 1 ([[ALR0:%.+]]): {
  // CHECK-NEXT:     [[RP0:%.+]] = hir.pick [[ALR0]] : !hir.any [0 (): {
  // CHECK-NEXT:       [[UV1:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       hir.yield [[UV1]] : !hir.any
  // CHECK-NEXT:     } 1 ([[ALR1:%.+]]): {
  // CHECK-NEXT:       [[RP1:%.+]] = hir.pick [[ALR1]] : !hir.any [0 (): {
  // CHECK-NEXT:         [[UV2:%.+]] = uir.unreachable_value
  // CHECK-NEXT:         hir.yield [[UV2]] : !hir.any
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         [[S:%.+]] = hir.add %x, %x
  // CHECK-NEXT:         hir.yield [[S]] : !hir.any
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield [[RP1]] : !hir.any
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield [[RP0]] : !hir.any
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     hir.write_var %{{.+}}, [[RES]]
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

//===----------------------------------------------------------------------===//
// The producer sits between two nested exits. It threads through only the
// reach pick that gates its segment; the exit after it rides an unvalued
// reach pick, since no demanded value lives beyond it.

// CHECK-LABEL: uir.func @LaterResultValueBetweenExits
uir.func @LaterResultValueBetweenExits(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      %r = uir.if %c : %t {
        uir.break
      } else {
        %d1 = hir.geq %n, %i : %t
        uir.if %d1 { uir.break }
        %v = uir.expr pin 1 : %t {
          %s = hir.add %x, %x : %t
          uir.yield %s : %t
        }
        %d2 = hir.geq %i, %n : %t
        uir.if %d2 { uir.break }
        uir.yield %v : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %r : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The first reach pick is valued and carries the producer; the second, after
  // it, is unvalued.
  // CHECK:      [[RES:%.+]] = hir.pick [[CFR:%.+]] : !hir.any [0 (): {
  // CHECK-NEXT:     uir.unreachable_value
  // CHECK:      } 1 ([[ALR0:%.+]]): {
  // CHECK-NEXT:     [[RP0:%.+]] = hir.pick [[ALR0]] : !hir.any [0 (): {
  // CHECK-NEXT:       [[UV1:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       hir.yield [[UV1]] : !hir.any
  // CHECK-NEXT:     } 1 ([[ALR1:%.+]]): {
  // CHECK-NEXT:       [[S:%.+]] = hir.add %x, %x
  // CHECK-NEXT:       hir.pick [[ALR1]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield [[S]] : !hir.any
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield [[RP0]] : !hir.any
  // CHECK-NEXT:   }]
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

//===----------------------------------------------------------------------===//
// Rejoining content with an empty trailing segment: the exit-carrying branch is
// the loop's last op, so the `continued` alternative holds only the nested
// arm-pick and no following segment content.

// CHECK-LABEL: uir.func @RejoiningArmTrailingEmpty
uir.func @RejoiningArmTrailingEmpty(%n: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:     [[CF:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK-NEXT:       hir.choice 0()
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[I:%.+]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // The back-edge reach link carries no payload (empty trailing segment).
  // CHECK:          [[R:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack([[CF]], [[R]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      } else {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SI:%.+]]): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, [[SI]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  uir.return
}

//===----------------------------------------------------------------------===//
// Two sequential exit-carrying divergences, each with rejoining content. Two
// chain links and two arm-choices compose: the full-activation record nests
// `continued_0(ac0, continued_1(ac1))`, and the replicate replays ac0's
// content, the middle segment, then ac1's content in program order.

// CHECK-LABEL: uir.func @TwoDivergencesRejoin
uir.func @TwoDivergencesRejoin(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:     [[CFA:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CA1:%.+]] = hir.choice 1([[I:%.+]])
  // CHECK-NEXT:       uir.yield [[CA1]]
  // CHECK-NEXT:     }
  // The second divergence's exit record carries div0's cf result and a
  // REACHED div0 link wrapping div1's exit-built cf and NOT_REACHED link.
  // CHECK:     [[CFB:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       [[XB:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[XNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[XR:%.+]] = hir.choice 1([[XB]], [[XNR]])
  // CHECK-NEXT:       hir.opaque_pack([[CFA]], [[XR]])
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CB1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CB1]]
  // CHECK-NEXT:     }
  // The full record nests div1's slots inside div0's REACHED link.
  // CHECK:          [[R1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     [[R0:%.+]] = hir.choice 1([[CFB]], [[R1]])
  // CHECK-NEXT:     hir.opaque_pack([[CFA]], [[R0]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %d1 = hir.geq %i, %n : %t
      uir.if %d1 {
        uir.break
      } else {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.yield
      }
      uir.expr pin 1 {
        hir.write_var %acc, %n : %t
        uir.yield
      }
      %d2 = hir.geq %n, %i : %t
      uir.if %d2 {
        uir.break
      } else {
        uir.expr pin 1 {
          hir.write_var %acc, %one : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // div0's cf pick (arm content), then div0's reach pick gating the middle
  // segment, div1's cf pick, and div1's reach pick, nested in program order.
  // CHECK: hir.replicate ([[CF0:%.+]], [[L0:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CF0]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SM0:%.+]]): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, [[SM0]]
  // CHECK:          }]
  // CHECK-NEXT:     hir.pick [[L0]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[CF1:%.+]], [[L1:%.+]]): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, %n
  // CHECK-NEXT:       hir.pick [[CF1]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.write_var
  // CHECK:            }]
  // CHECK-NEXT:       hir.pick [[L1]] [0 (): {
  // CHECK:            }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Nested exit inside an exit-carrying arm: the `if done` continue-arm is not
// straight-line -- it carries content, a nested exit-carrying `if d2`, and its
// own `break`. The arm decomposes recursively: its cf payload nests the arm's
// first-segment samples and the nested divergence's reach link, and the
// receiving cf alternative holds a nested reach pick. Both nested paths run no
// further content here, so both sub-alternatives are empty.

// CHECK-LABEL: uir.func @NestedExitArm
uir.func @NestedExitArm(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // The nested `if d2` exit: the arm's cf nests a NOT_REACHED arm link.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EE:%.+]] = hir.choice 0([[I]], [[SNR]])
  // CHECK-NEXT:         [[NR0:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[EE]], [[NR0]])
  // CHECK-NEXT:         hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:         uir.break
  // CHECK-NEXT:       }
  // The outer `break`: the arm's cf nests a REACHED arm link (empty payload).
  // CHECK-NEXT:       [[SR:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[EC:%.+]] = hir.choice 0([[I]], [[SR]])
  // CHECK-NEXT:       [[NR1:%.+]] = hir.choice 0()
  // CHECK-NEXT:       hir.opaque_pack([[EC]], [[NR1]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // The body back-edge: a REACHED link carrying the trailing segment sample.
  // CHECK:          [[BR:%.+]] = hir.choice 1([[I]])
  // CHECK-NEXT:     hir.opaque_pack(%{{.+}}, [[BR]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:     uir.continue
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        %d2 = hir.geq %i, %n : %t
        uir.if %d2 {
          uir.break
        }
        uir.break
      }
      %next = hir.add %i, %one : %t
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[BARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[BARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[DARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[DARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Nested exit inside a continue-arm carrying content on both nested paths. The
// exit arm runs content B, then a nested exit-carrying `if d`, then content C
// before its `continue`. Distinct arm-interior samples (%e for B, %f for C)
// cross via the cf nesting: %e rides the arm's cf payload (visible on both
// nested paths), %f rides the arm's REACHED link (only the fall-through).

// CHECK-LABEL: uir.func @NestedExitArmContent
uir.func @NestedExitArmContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       [[E:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:       [[D:%.+]] = hir.geq %n, [[I]]
  // The nested break: cf(%e, NOT_REACHED()), loop link NOT_REACHED.
  // CHECK-NEXT:       uir.if [[D]] {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EE:%.+]] = hir.choice 0([[E]], [[SNR]])
  // CHECK-NEXT:         [[NR0:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[I]], [[EE]], [[NR0]])
  // CHECK-NEXT:         hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:         uir.break
  // CHECK-NEXT:       }
  // The arm continue: cf(%e, REACHED(%f)), loop link NOT_REACHED — the arm
  // ends the activation, so the trailing segment does not run.
  // CHECK-NEXT:       [[F:%.+]] = hir.add [[E]], %{{.+}}
  // CHECK-NEXT:       [[SR:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       [[EC:%.+]] = hir.choice 0([[E]], [[SR]])
  // CHECK-NEXT:       [[NR1:%.+]] = hir.choice 0()
  // CHECK-NEXT:       hir.opaque_pack([[I]], [[EC]], [[NR1]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:       uir.continue [[I]]
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // The back edge: a REACHED link with no payload — the trailing segment's
  // sample deduplicates against the record's shared prefix field.
  // CHECK:          [[BR:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack([[I]], %{{.+}}, [[BR]])
  // CHECK-NEXT:     hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:     uir.continue
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %c = hir.geq %i, %n : %t
      uir.if %c {
        %e = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %e : %t
          uir.yield
        }
        %d = hir.geq %n, %i : %t
        uir.if %d {
          uir.break
        }
        %f = hir.add %e, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %f : %t
          uir.yield
        }
        uir.continue %i : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[SI:%.+]], [[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.write_var [[ACC]], [[SI]]
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[EARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[EARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SI]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.yield
  // CHECK-NEXT:   }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Recursive rejoining content: a nested exit-carrying `if d` inside the
// continue-arm of `if c` has a rejoining arm holding later-phase content X.
// The nested divergence gets its own cf choice inside the arm's plan: it
// rides the arm's cf payload next to the arm's reach link, and the receiving
// arm alternative dispatches it with a nested cf pick before the nested reach
// pick gating the following segment C.

// CHECK-LABEL: uir.func @NestedRejoinInExitArm
uir.func @NestedRejoinInExitArm(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       [[E:%.+]] = hir.add
  // CHECK-NEXT:       hir.geq
  // CHECK-NEXT:       hir.choice_type
  // The nested `if d` is rebuilt with its own cf result; its break site
  // builds the whole record bottom-up.
  // CHECK-NEXT:       [[NCF:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:         [[XC:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[XNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.choice 0([[E]], [[XC]], [[XNR]])
  // CHECK:              uir.break
  // CHECK-NEXT:       } else {
  // CHECK:              [[NCF1:%.+]] = hir.choice 1(%{{.+}})
  // CHECK-NEXT:         uir.yield [[NCF1]]
  // CHECK-NEXT:       }
  // The arm continue: the arm's cf carries B's sample, the nested cf result,
  // and the arm's REACHED link with segment %f.
  // CHECK-NEXT:       [[F:%.+]] = hir.add
  // CHECK-NEXT:       [[SR:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       hir.choice 0([[E]], [[NCF]], [[SR]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      uir.if %c {
        %e = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %e : %t
          uir.yield
        }
        %d = hir.geq %n, %i : %t
        uir.if %d {
          uir.break
        } else {
          %g = hir.add %e, %one : %t
          uir.expr pin 1 {
            hir.write_var %acc, %g : %t
            uir.yield
          }
          uir.yield
        }
        %f = hir.add %e, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %f : %t
          uir.yield
        }
        uir.continue %i : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The arm alternative replays B, dispatches X via the nested cf pick, then
  // gates C behind the arm's reach pick, in program order.
  // CHECK: hir.replicate
  // CHECK:     hir.pick %{{.+}} [0 ([[EARG:%.+]], [[NCFR:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[EARG]]
  // CHECK-NEXT:       hir.pick [[NCFR]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[GARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[GARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A rejoining arm that hides a nested exit but runs no fall-through content.
// The arm is a block like the loop body: its nested exit builds its record
// with the arm's cf tag wrapping a NOT_REACHED arm link, and its content-free
// fall-through yields the same tag with a REACHED arm link. The cf stays
// binary (one tag per arm) and the reach link gates the back edge.

// CHECK-LABEL: uir.func @RejoiningNestedExit
uir.func @RejoiningNestedExit(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // The then arm exits (cf tag 0) after pushing content %i.
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0([[I]])
  // CHECK-NEXT:       [[NR0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       hir.opaque_pack([[CF0]], [[NR0]])
  // CHECK:            uir.break
  // The else arm's nested exit wraps a NOT_REACHED arm link in cf tag 1.
  // CHECK:          } else {
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[CF1X:%.+]] = hir.choice 1([[SNR]])
  // CHECK-NEXT:         [[NR1:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[CF1X]], [[NR1]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // The content-free fall-through yields cf tag 1 with a REACHED arm link.
  // CHECK-NEXT:       [[SRR:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[SRR]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // CHECK:          [[R:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack(%{{.+}}, [[R]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.break
      } else {
        %d2 = hir.geq %i, %n : %t
        uir.if %d2 { uir.break }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[BARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, [[BARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Both arms of an exit-carrying `if` rejoin, and one hides a nested exit.
// Both arms carry fall-through content, so the cf choice has one tag per arm
// (then=0, else=1) carrying that arm's samples. The else arm's nested exit
// builds its record with cf tag 1 wrapping a NOT_REACHED arm link, while the
// fall-through yields tag 1 with %f in the REACHED arm link. The receiving cf
// pick replays %e or the else arm's nested structure, then the reach pick
// runs the trailing segment %i.

// CHECK-LABEL: uir.func @ExArmsRejoinNestedExit
uir.func @ExArmsRejoinNestedExit(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          [[CF:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // Then rejoins with content %e as cf tag 0 (no nested divergence, so the
  // payload is just the arm's samples).
  // CHECK-NEXT:       [[E:%.+]] = hir.add
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0([[E]])
  // CHECK-NEXT:       uir.yield [[CF0]]
  // CHECK-NEXT:     } else {
  // The else arm's nested exit wraps a NOT_REACHED arm link in cf tag 1; its
  // fall-through yields tag 1 with %f in the REACHED arm link.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[CF1X:%.+]] = hir.choice 1([[SNR]])
  // CHECK-NEXT:         [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[CF1X]], [[NR]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[F:%.+]] = hir.add
  // CHECK-NEXT:       [[SR:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[SR]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // The back-edge record carries the cf result and the trailing segment %i in
  // the REACHED link.
  // CHECK:          [[R:%.+]] = hir.choice 1([[I:%.+]])
  // CHECK-NEXT:     hir.opaque_pack([[CF]], [[R]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      uir.if %c {
        %e = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %e : %t
          uir.yield
        }
        uir.yield
      } else {
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %f = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %f : %t
          uir.yield
        }
        uir.yield
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The cf pick replays %e or the else arm's nested structure; the reach pick
  // then runs the trailing segment %i.
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[EARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[EARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[IARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Both arms rejoin and both hide a nested exit. Each cf tag nests its arm's
// reach link: NOT_REACHED at the arm's nested break, REACHED with the arm's
// content sample (%p or %q) at its yield. The receiving cf pick replays
// whichever arm ran, and the loop's reach pick gates the segment %i.

// CHECK-LABEL: uir.func @BothArmsRejoinNestedExit
uir.func @BothArmsRejoinNestedExit(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          [[CF:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // The then arm: its nested exit wraps a NOT_REACHED arm link in cf tag 0;
  // the fall-through yields tag 0 with %p in the REACHED arm link.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SP:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EP:%.+]] = hir.choice 0([[SP]])
  // CHECK-NEXT:         [[NRP:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[EP]], [[NRP]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[P:%.+]] = hir.add
  // CHECK-NEXT:       [[RP:%.+]] = hir.choice 1([[P]])
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0([[RP]])
  // CHECK-NEXT:       uir.yield [[CF0]]
  // The else arm: the same shape under cf tag 1 with %q.
  // CHECK-NEXT:     } else {
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SQ:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EQ:%.+]] = hir.choice 1([[SQ]])
  // CHECK-NEXT:         [[NRQ:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[EQ]], [[NRQ]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[Q:%.+]] = hir.add
  // CHECK-NEXT:       [[RQ:%.+]] = hir.choice 1([[Q]])
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[RQ]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // CHECK:          [[R:%.+]] = hir.choice 1([[I:%.+]])
  // CHECK-NEXT:     hir.opaque_pack([[CF]], [[R]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      uir.if %c {
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %p = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %p : %t
          uir.yield
        }
        uir.yield
      } else {
        %e = hir.geq %i, %n : %t
        uir.if %e { uir.break }
        %q = hir.add %n, %i : %t
        uir.expr pin 1 {
          hir.write_var %acc, %q : %t
          uir.yield
        }
        uir.yield
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[SUB0:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUB0]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[PARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC:%.+]], [[PARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SUB1:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUB1]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[QARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[QARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[IARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A rejoining arm whose every path exits (its yield is unreachable): the
// nested `if d { break } else { continue }` leaves the arm's yield dead, so
// the arm provides no cf result (its yield is emptied) and the cf is provided
// by the else fall-through alone. Content before the nested exit (%y) rides
// the arm's cf payload — built identically at both exit sites, since a
// content-free nested divergence records only NOT_REACHED regardless of which
// exit fired — so %y is replayed once before the arm's reach pick.

// CHECK-LABEL: uir.func @RejoinArmAllExits
uir.func @RejoinArmAllExits(%n: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // The break record: cf tag 0 carries %y and a NOT_REACHED arm link.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EB:%.+]] = hir.choice 0([[Y:%.+]], [[SB]])
  // CHECK-NEXT:         [[NRB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[I:%.+]], [[EB]], [[NRB]])
  // CHECK:              uir.break
  // The continue record: identical — which exit fired is not recorded, since
  // the content-free nested divergence gates nothing.
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         [[SC:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EC:%.+]] = hir.choice 0([[Y]], [[SC]])
  // CHECK-NEXT:         [[NRC:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[I]], [[EC]], [[NRC]])
  // CHECK:              uir.continue
  // CHECK-NEXT:       }
  // The arm's dead yield provides unreachable_value; the else fall-through
  // provides the cf.
  // CHECK-NEXT:       [[DEAD:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       uir.yield [[DEAD]]
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // CHECK:          [[R:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack([[I]], %{{.+}}, [[R]])
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %c = hir.geq %i, %n : %t
      uir.if %c {
        %y = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %y : %t
          uir.yield
        }
        %d = hir.geq %n, %i : %t
        uir.if %d {
          uir.break
        } else {
          uir.continue %i : %t
        }
        uir.yield
      }
      uir.expr pin 1 {
        hir.write_var %acc, %i : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[IREC:%.+]], [[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // %x runs unconditionally, before the cf pick.
  // CHECK-NEXT:     hir.write_var [[ACC:%.+]], [[IREC]]
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[YARG:%.+]], [[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[YARG]]
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IREC]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  uir.return
}

//===----------------------------------------------------------------------===//
// Both arms carry content and exit: no path falls through the divergence, so
// the rebuilt `if` keeps no cf result (an `if` whose regions all exit early
// produces no results) — every record's cf is built at its own exit site, and
// the loop body terminal is unreachable, so no back-edge record is pushed.

// CHECK-LABEL: uir.func @BothArmsBreakContent
uir.func @BothArmsBreakContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} {
  // CHECK-NEXT:       [[CF0:%.+]] = hir.choice 0([[I:%.+]])
  // CHECK-NEXT:       [[NR0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       hir.opaque_pack([[CF0]], [[NR0]])
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[M:%.+]] = hir.add
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[M]])
  // CHECK-NEXT:       [[NR1:%.+]] = hir.choice 0()
  // CHECK-NEXT:       hir.opaque_pack([[CF1]], [[NR1]])
  // CHECK:            uir.break
  // CHECK-NEXT:     }
  // CHECK-NEXT:     uir.unreachable
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.break
      } else {
        %m = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %m : %t
          uir.yield
        }
        uir.break
      }
      uir.unreachable
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[SI:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[SI]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SM:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SM]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Shared arm prefix: later-phase content before a nested exit inside a
// rejoining arm runs on both the exit path and the fall-through. Its replay
// lives once in the cf pick's arm alternative, ahead of the arm's reach pick,
// so no content is cloned across record variants: the cf is built with a
// NOT_REACHED arm link at the nested break and with a REACHED link at the arm
// yield, and both carry the shared prefix sample %w.

// CHECK-LABEL: uir.func @RejoinSharedPrefix
uir.func @RejoinSharedPrefix(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[W:%.+]] = hir.add
  // The nested break: cf tag 1 carries %w and a NOT_REACHED arm link.
  // CHECK-NEXT:       hir.geq
  // CHECK-NEXT:       uir.if %{{.+}} {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[XC:%.+]] = hir.choice 1([[W]], [[SNR]])
  // CHECK-NEXT:         [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[XC]], [[NR]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // The fall-through: the same tag with %f in the REACHED arm link.
  // CHECK-NEXT:       [[F:%.+]] = hir.add
  // CHECK-NEXT:       [[SR:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[W]], [[SR]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      uir.if %c {
        uir.break
      } else {
        %w = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %w : %t
          uir.yield
        }
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %f = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %f : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The shared prefix %w replays once, before the arm's reach pick.
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[WARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[WARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Shared arm prefix with a post-branch sample: the trailing segment samples an
// effectful phase -1 value (%g) that exists only on continuing activations. It
// rides the loop-level reach link's REACHED payload; exit records carry the
// empty NOT_REACHED tag, and the receiving reach pick binds %g exactly on the
// paths that replay its consumer.

// CHECK-LABEL: uir.func @SharedPrefixPostBranchSample
uir.func @SharedPrefixPostBranchSample(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // The nested break: NOT_REACHED loop link, no %g field.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SNR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[XC:%.+]] = hir.choice 0([[I:%.+]], [[SNR]])
  // CHECK-NEXT:         [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[XC]], [[NR]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK:          } else {
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  // The back edge samples %g into the REACHED loop link.
  // CHECK-NEXT:     [[G:%.+]] = hir.read_var
  // CHECK-NEXT:     hir.add
  // CHECK-NEXT:     [[R:%.+]] = hir.choice 1([[G]])
  // CHECK-NEXT:     hir.opaque_pack(%{{.+}}, [[R]])
  uir.expr pin -1 {
    %v = hir.var : %t
    uir.loop (%i = %zero : %t) {
      hir.write_var %v, %i : %t
      %c = hir.geq %i, %n : %t
      uir.if %c {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %m = hir.add %i, %one : %t
        uir.expr pin 1 {
          hir.write_var %acc, %m : %t
          uir.yield
        }
        uir.yield
      }
      %g = hir.read_var %v : %t
      uir.expr pin 1 {
        hir.write_var %acc, %g : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 ([[SI:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[SI]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[SM:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[SM]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A rejoining arm whose fall-through (after a nested exit) carries another
// exit-carrying branch with its own rejoining content: the nested rejoining
// divergence gets its own cf choice inside the arm's REACHED link, and the
// receiving structure recurses — a cf pick and reach pick per level.

// CHECK-LABEL: uir.func @RejoinNestedFallthrough
uir.func @RejoinNestedFallthrough(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          uir.if %{{.+}} : %{{.+}} {
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // The first nested exit ends the arm's chain before the inner divergence.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[ANR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[XC:%.+]] = hir.choice 1([[ANR]])
  // CHECK-NEXT:         [[NR:%.+]] = hir.choice 0()
  // CHECK-NEXT:         hir.opaque_pack([[XC]], [[NR]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // The inner rejoining divergence is rebuilt with its own cf result, which
  // rides the arm's REACHED link.
  // CHECK:            [[ICF:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK:              uir.break
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         [[H:%.+]] = hir.add
  // CHECK-NEXT:         [[ICF1:%.+]] = hir.choice 1([[H]])
  // CHECK-NEXT:         uir.yield [[ICF1]]
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[IR:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[AR:%.+]] = hir.choice 1([[ICF]], [[IR]])
  // CHECK-NEXT:       [[CF1:%.+]] = hir.choice 1([[AR]])
  // CHECK-NEXT:       uir.yield [[CF1]]
  // CHECK-NEXT:     }
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      uir.if %c {
        uir.break
      } else {
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %e = hir.geq %i, %n : %t
        uir.if %e {
          uir.break
        } else {
          %h = hir.add %i, %one : %t
          uir.expr pin 1 {
            hir.write_var %acc, %h : %t
            uir.yield
          }
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[CFR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[ALINK:%.+]]): {
  // CHECK-NEXT:       hir.pick [[ALINK]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[ICFR:%.+]], [[ILR:%.+]]): {
  // CHECK-NEXT:         hir.pick [[ICFR]] [0 (): {
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         } 1 ([[SH:%.+]]): {
  // CHECK-NEXT:           hir.write_var %{{.+}}, [[SH]]
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         }]
  // CHECK-NEXT:         hir.pick [[ILR]] [0 (): {
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         } 1 (): {
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         }]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// An exit-carrying divergence whose rejoining arm yields a demanded
// later-phase result the arm produces no later content for: the value comes
// from outside the arm (a function argument), so the receiving cf pick's
// rejoining alternative yields it directly, and the exit alternative yields
// an `uir.unreachable_value` that only gated-off paths could consume.

// CHECK-LABEL: uir.func @DivergenceLaterResultForwarded
uir.func @DivergenceLaterResultForwarded(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      %r = uir.if %done : %t {
        uir.break
      } else {
        uir.yield %x : %t
      }
      uir.expr pin 1 {
        hir.write_var %acc, %r : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate ([[CFR:%.+]], [[LR:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     [[RES:%.+]] = hir.pick [[CFR]] : !hir.any [0 (): {
  // CHECK-NEXT:       [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       hir.yield [[UV]] : !hir.any
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.yield %x : !hir.any
  // CHECK-NEXT:     }]
  // CHECK-NEXT:     hir.pick [[LR]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, [[RES]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

// A rejoining arm that does not deliver the branch's kept result yields
// `uir.unreachable_value` in that slot. The cf choice lands in the branch's
// last result and the unreachable value rides the kept slot exactly like a real
// value, so the divergence sorts through the ordinary valued cf pick; the
// unreachable value is consumed only on the never-taken fall-through path.
// CHECK-LABEL: uir.func @ShortYieldKeptResult
uir.func @ShortYieldKeptResult(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: uir.phase -1 {
  // CHECK:          [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT:     [[IF:%.+]]:2 = uir.if %{{.+}} : %{{.+}}, [[CT]] {
  // The then arm delivers the kept result and cf tag 0.
  // CHECK-NEXT:       [[TCF:%.+]] = hir.choice 0([[I:%.+]])
  // CHECK-NEXT:       uir.yield [[I]], [[TCF]] : %{{.+}}, [[CT]]
  // CHECK-NEXT:     } else {
  // The else arm records its nested break, then rejoins yielding an unreachable
  // value for the kept slot and cf tag 1. The authored `uir.unreachable_value`
  // is a `never` producer left in place; never-forwarding rewrites the yield's
  // use of it to a fresh stand-in at this phase, exactly as it leaves a never
  // loop in place and forwards its result.
  // CHECK:            uir.unreachable_value
  // CHECK-NEXT:       [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       [[EIN:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[ECF:%.+]] = hir.choice 1([[EIN]])
  // CHECK-NEXT:       uir.yield [[UV]], [[ECF]] : %{{.+}}, [[CT]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     hir.add [[IF]]#0
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %c = hir.geq %i, %n : %t
      %r = uir.if %c : %t {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        uir.yield %i : %t
      } else {
        %d = hir.geq %n, %i : %t
        uir.if %d { uir.break }
        %uv = uir.unreachable_value
        uir.yield %uv : %t
      }
      %next = hir.add %r, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The phase-0 replicate reconstructs the then arm's later write via the cf
  // pick; the else alternative replays only its nested break record.
  // CHECK: uir.phase 0 {
  // CHECK:          hir.replicate ([[CFR:%.+]], %{{.+}}) in %{{.+}} {
  // CHECK-NEXT:       hir.pick [[CFR]] [0 ([[YARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var %{{.+}}, [[YARG]]
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

//===----------------------------------------------------------------------===//
// Never-value forwarding: a control-flow result no path provides gets the
// bottom phase `never`. Every real-phase use of it is rewritten to a fresh
// `uir.unreachable_value` at the consuming phase, and the `never` result is
// left unused on its producer. `never` is non-viral, so the consumer keeps its
// own real phase.

// A never `uir.if` result (neither arm provides it) consumed by a same-phase
// op: the unreachable value and the consumer wrap into the consumer's phase,
// and the never-producing if wraps into its own phase with an unused result.
// CHECK-LABEL: uir.func @NeverConsumedSamePhase
uir.func @NeverConsumedSamePhase(%c: 0, %x: 0) -> (out: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: [[IF:%.+]] = uir.if %c : [[T]] {
  // CHECK-NEXT:   [[A:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[A]] : [[T]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[B:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[B]] : [[T]]
  // CHECK-NEXT: }
  %r = uir.if %c : %t {
    %uv0 = uir.unreachable_value
    uir.yield %uv0 : %t
  } else {
    %uv1 = uir.unreachable_value
    uir.yield %uv1 : %t
  }
  // CHECK-NEXT: [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT: [[U:%.+]] = hir.add %x, [[UV]] : [[T]]
  %u = hir.add %x, %r : %t
  // CHECK-NEXT: uir.phase_return [[U]] : [[T]]
  // CHECK-NEXT: }
  uir.return %u : %t
}

// The same never `uir.if` result consumed one phase later: the unreachable
// value materializes in the consumer's phase (1), while the never-producing
// if wraps into phase 0 with a contentless phase return.
// CHECK-LABEL: uir.func @NeverConsumedLater
uir.func @NeverConsumedLater(%c: 0, %x: 1) -> (out: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: uir.if %c : [[T]] {
  // CHECK-NEXT:   [[A:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[A]] : [[T]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[B:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[B]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase_return{{$}}
  // CHECK-NEXT: }
  %r = uir.if %c : %t {
    %uv0 = uir.unreachable_value
    uir.yield %uv0 : %t
  } else {
    %uv1 = uir.unreachable_value
    uir.yield %uv1 : %t
  }
  // CHECK: uir.phase 1 {
  // CHECK-NEXT: [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT: [[U:%.+]] = hir.add %x, [[UV]] : [[T]]
  // CHECK-NEXT: uir.phase_return [[U]] : [[T]]
  // CHECK-NEXT: }
  %u = hir.add %x, %r : %t
  uir.return %u : %t
}

// A never value consumed by the function return at a const (phase -1) result
// slot: the unreachable value materializes at the result's declared phase, not
// the return op's own phase (always 0). This exercises the return's
// per-result-slot use attribution.
// CHECK-LABEL: uir.func @NeverReturnedConst
uir.func @NeverReturnedConst(%c: 0) -> (out: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase -1 {
  // CHECK-NEXT: [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT: uir.phase_return [[UV]] : [[T]]
  // CHECK-NEXT: }
  %r = uir.if %c : %t {
    %uv0 = uir.unreachable_value
    uir.yield %uv0 : %t
  } else {
    %uv1 = uir.unreachable_value
    uir.yield %uv1 : %t
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: uir.if %c : [[T]] {
  // CHECK-NEXT:   [[A:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[A]] : [[T]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[B:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[B]] : [[T]]
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase_return{{$}}
  // CHECK-NEXT: }
  uir.return %r : %t
}

// A never `uir.loop` result (no breaks) consumed by a const op: forwarding
// replaces the const-phase use, so the loop wraps into phase 0 unused while
// the consumer stays at phase -1. (Formerly a hard-error input.)
// CHECK-LABEL: uir.func @NeverLoopResult
uir.func @NeverLoopResult(%a: -1, %x: 0) -> (c: -1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase -1 {
  // CHECK-NEXT: [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT: [[S:%.+]] = hir.add %a, [[UV]] : [[T]]
  // CHECK-NEXT: uir.phase_return [[S]] : [[T]]
  // CHECK-NEXT: }
  %r = uir.loop : %t {
    uir.continue
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: uir.loop : [[T]] {
  // CHECK-NEXT:   uir.continue
  // CHECK-NEXT: }
  // CHECK-NEXT: uir.phase_return{{$}}
  // CHECK-NEXT: }
  %s = hir.add %a, %r : %t
  uir.return %s : %t
}

// A dead pure region-free op consuming a never result with no real operand
// stays float (undemanded), so never-forwarding leaves it alone. The sorter's
// float-sink rule places it in the never producer's phase after the producer,
// keeping the never operand, unused — no crossing, no error.
// CHECK-LABEL: uir.func @DeadFloatNeverConsumer
uir.func @DeadFloatNeverConsumer(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: [[L:%.+]] = uir.loop : [[T]] {
  // CHECK-NEXT:   uir.continue
  // CHECK-NEXT: }
  // CHECK-NEXT: hir.add [[L]], [[L]] : [[T]]
  // CHECK-NEXT: uir.phase_return{{$}}
  // CHECK-NEXT: }
  %r = uir.loop : %t {
    uir.continue
  }
  %d = hir.add %r, %r : %t
  uir.return
}

// A never loop nested in a floating frame inside an earlier-phase branch: the
// frame's later-phase interior (the dead multiply) derives a real phase from
// its live sibling operand, so its never operand is forwarded to an
// unreachable value that lands in the receiving pick. The never loop wraps into
// the phase-0 branch arm, unused. (Formerly a hard-error input.)
// CHECK-LABEL: uir.func @FloatingFrameNeverInterior
uir.func @FloatingFrameNeverInterior(%c: 0, %a: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK: uir.if %c
  // CHECK: uir.loop
  // CHECK: uir.phase 1 {
  // CHECK: hir.pick
  // CHECK: [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT: hir.mul [[UV]], %a : {{.*}}
  %r = uir.if %c : %t {
    %fl = uir.expr : %t {
      %s = uir.loop : %t {
        uir.continue
      }
      uir.yield %s : %t
    }
    %dead = hir.mul %fl, %a : %t
    %live = hir.add %a, %a : %t
    uir.yield %live : %t
  } else {
    %w = hir.mul %a, %a : %t
    uir.yield %w : %t
  }
  uir.return %r : %t
}
