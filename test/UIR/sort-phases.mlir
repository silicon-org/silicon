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
// Rejoining-arm content: an exit-carrying branch whose rejoining arm holds
// later-phase content is rebuilt so the arm yields an arm-choice of that
// content's samples. The choice rides the divergence's `continued` link, and a
// nested pick in the receiving `continued` alternative replays the arm content
// before the following segment. The rejoining arm samples an arm-interior
// phase -1 value (%m), which the arm-choice captures inside the arm.

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
  // The rebuilt if yields the arm-choice; the exit arm breaks after pushing its
  // own exit record.
  // CHECK-NEXT:     [[AC:%.+]] = uir.if [[DONE]] : %{{.+}} {
  // CHECK-NEXT:       [[EX:%.+]] = hir.choice 0([[I]])
  // CHECK-NEXT:       [[RE:%.+]] = hir.opaque_pack([[EX]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RE]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[M:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:       [[AC0:%.+]] = hir.choice 0([[M]])
  // CHECK-NEXT:       uir.yield [[AC0]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // The continued link carries the segment sample and the arm-choice.
  // CHECK-NEXT:     [[CO:%.+]] = hir.choice 1([[I]], [[AC]])
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[CO]])
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
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[SC:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[SC]]
  // CHECK-NEXT:       hir.yield
  // The continued alternative dispatches the arm-choice (rejoining content)
  // before the following segment content.
  // CHECK-NEXT:     } 1 ([[SE:%.+]], [[ACARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[ACARG]] [0 ([[SM:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[SM]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
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
// so the arm-pick that replays that content is made valued: it yields the
// result, and the following segment consumes the pick's result. The exit arm
// (break) provides no result; on the break path the value never materializes,
// and the consumer sits only in the `continued` alternative.

// CHECK-LABEL: uir.func @DivergenceLaterResult
uir.func @DivergenceLaterResult(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // The rebuilt if drops the later result and yields only the arm-choice; the
  // exit arm pushes its own exit record and breaks.
  // CHECK: uir.phase -1 {
  // CHECK-NEXT:   [[LIST:%.+]] = hir.opaque_list_create
  // CHECK-NEXT:   uir.loop ([[I:%.+]] = %{{.+}} : %{{.+}}) {
  // CHECK-NEXT:     [[DONE:%.+]] = hir.geq [[I]], %n
  // CHECK-NEXT:     hir.choice_type
  // CHECK-NEXT:     [[AC:%.+]] = uir.if [[DONE]] : %{{.+}} {
  // CHECK-NEXT:       [[EX:%.+]] = hir.choice 0()
  // CHECK-NEXT:       [[RE:%.+]] = hir.opaque_pack([[EX]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]], [[RE]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[AC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:       uir.yield [[AC0]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     [[NEXT:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:     [[CO:%.+]] = hir.choice 1([[AC]])
  // CHECK-NEXT:     [[RC:%.+]] = hir.opaque_pack([[CO]])
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
  // The continued alternative's arm-pick is valued: it reconstructs the later
  // result by yielding the rejoining arm's content value, which the following
  // segment then writes.
  // CHECK-NEXT: uir.phase 0 {
  // CHECK-NEXT:   [[ACC:%.+]] = hir.var
  // CHECK-NEXT:   hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[ACARG:%.+]]): {
  // CHECK-NEXT:       [[RES:%.+]] = hir.pick [[ACARG]] : !hir.any [0 (): {
  // CHECK-NEXT:         [[S:%.+]] = hir.add %x, %x
  // CHECK-NEXT:         hir.yield [[S]] : !hir.any
  // CHECK-NEXT:       }]
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
  // CHECK:     [[AC:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[AC0:%.+]] = hir.choice 0([[I:%.+]])
  // CHECK-NEXT:       uir.yield [[AC0]]
  // CHECK-NEXT:     }
  // The continued link carries only the arm-choice (empty trailing segment).
  // CHECK:          [[CO:%.+]] = hir.choice 1([[AC]])
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
  // CHECK: hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[ACARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[ACARG]] [0 ([[SM:%.+]]): {
  // CHECK-NEXT:         hir.write_var %{{.+}}, [[SM]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
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
  // CHECK:     [[AC0:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       [[C0:%.+]] = hir.choice 0([[I:%.+]])
  // CHECK-NEXT:       uir.yield [[C0]]
  // CHECK-NEXT:     }
  // The second divergence's exit record wraps div0's continued link (ac0).
  // CHECK:     [[AC1:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK-NEXT:       hir.choice 1([[AC0]], %{{.+}})
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK-NEXT:       uir.yield
  // CHECK-NEXT:     }
  // The full record nests both continued links and both arm-choices.
  // CHECK:          [[TAIL:%.+]] = hir.choice 1([[AC1]])
  // CHECK-NEXT:     hir.choice 1([[AC0]], [[TAIL]])
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
  // CHECK: hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK:     hir.pick [[REC]] [0 (): {
  // CHECK:     } 1 ([[A0:%.+]], [[CHAIN:%.+]]): {
  // div0 rejoining content, then the middle segment, then div1's pick.
  // CHECK-NEXT:       hir.pick [[A0]] [0 ([[SM0:%.+]]): {
  // CHECK-NEXT:         hir.write_var %{{.+}}, [[SM0]]
  // CHECK:            }]
  // CHECK-NEXT:       hir.write_var %{{.+}}, %n
  // CHECK-NEXT:       hir.pick [[CHAIN]] [0 (): {
  // CHECK:            } 1 ([[A1:%.+]]): {
  // CHECK-NEXT:         hir.pick [[A1]] [0 (): {
  // CHECK-NEXT:           hir.write_var
  // CHECK:              }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Nested exit inside an exit-carrying arm: the `if done` continue-arm is not
// straight-line -- it carries content, a nested exit-carrying `if d2`, and its
// own `break`. The exit arm decomposes recursively: its `exited` chain link
// nests a sub-chain over the nested divergence, and the receiving `exited`
// alternative holds a nested pick over that sub-chain. Both nested paths run no
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
  // CHECK:          uir.if %{{.+}} {
  // The nested `if d2` exit: exited(div0) wraps exited(sub_div0) (empty).
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SE:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EE:%.+]] = hir.choice 0([[I]], [[SE]])
  // CHECK-NEXT:         hir.opaque_pack([[EE]])
  // CHECK-NEXT:         hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:         uir.break
  // CHECK-NEXT:       }
  // The outer `break`: exited(div0) wraps continued(sub_div0) (empty payload).
  // CHECK-NEXT:       [[SC:%.+]] = hir.choice 1()
  // CHECK-NEXT:       [[EC:%.+]] = hir.choice 0([[I]], [[SC]])
  // CHECK-NEXT:       hir.opaque_pack([[EC]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:       uir.break
  // CHECK-NEXT:     }
  // The body back-edge: continued(div0) carrying the trailing segment sample.
  // CHECK:          [[BC:%.+]] = hir.choice 1([[I]])
  // CHECK-NEXT:     hir.opaque_pack([[BC]])
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
  // CHECK-NEXT:   hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[BARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[BARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
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
// cross via the nesting: %e rides the `exited(div0)` payload (visible on both
// nested paths), %f rides the nested `continued` link (only the fall-through).

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
  // CHECK:          uir.if %{{.+}} {
  // CHECK-NEXT:       [[E:%.+]] = hir.add [[I]], %{{.+}}
  // CHECK-NEXT:       [[D:%.+]] = hir.geq %n, [[I]]
  // The nested break: exited(div0)(%e, exited(sub_div0)()).
  // CHECK-NEXT:       uir.if [[D]] {
  // CHECK-NEXT:         [[SE:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EE:%.+]] = hir.choice 0([[E]], [[SE]])
  // CHECK-NEXT:         hir.opaque_pack([[I]], [[EE]])
  // CHECK-NEXT:         hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:         uir.break
  // CHECK-NEXT:       }
  // The arm continue: exited(div0)(%e, continued(sub_div0)(%f)).
  // CHECK-NEXT:       [[F:%.+]] = hir.add [[E]], %{{.+}}
  // CHECK-NEXT:       [[SC:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       [[EC:%.+]] = hir.choice 0([[E]], [[SC]])
  // CHECK-NEXT:       hir.opaque_pack([[I]], [[EC]])
  // CHECK-NEXT:       hir.opaque_list_push [[LIST]]
  // CHECK-NEXT:       uir.continue [[I]]
  // CHECK-NEXT:     }
  // CHECK:          [[BC:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack([[I]], [[BC]])
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
  // CHECK-NEXT:   hir.replicate ([[SI:%.+]], [[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.write_var [[ACC]], [[SI]]
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[EARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[EARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
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
// continue-arm of `if c` has a rejoining arm holding later-phase content X. The
// nested divergence is rebuilt to yield an arm-choice; the choice rides the
// nested `continued` link, and the receiving nested `continued` alternative
// replays X (via an arm-pick) before the following segment C. Case 7a composes
// under case 7b-i with no extra machinery.

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
  // The nested `if d` is rebuilt with an arm-choice result.
  // CHECK:          [[AC:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // CHECK-NEXT:       hir.choice 0()
  // CHECK:            uir.break
  // CHECK-NEXT:     } else {
  // CHECK:            [[ACHOICE:%.+]] = hir.choice 0(%{{.+}})
  // CHECK-NEXT:       uir.yield [[ACHOICE]]
  // CHECK-NEXT:     }
  // The arm continue: continued(sub_div0) carries segment %f and the arm-choice.
  // CHECK:          [[SC:%.+]] = hir.choice 1(%{{.+}}, [[AC]])
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
  // The nested continued alternative replays X before segment C, in order.
  // CHECK: hir.replicate
  // CHECK:     hir.pick %{{.+}} [0 ([[EARG:%.+]], [[SUB:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC:%.+]], [[EARG]]
  // CHECK-NEXT:       hir.pick [[SUB]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]], [[ACARG:%.+]]): {
  // CHECK-NEXT:         hir.pick [[ACARG]] [0 ([[GARG:%.+]]): {
  // CHECK-NEXT:           hir.write_var [[ACC]], [[GARG]]
  // CHECK-NEXT:           hir.yield
  // CHECK-NEXT:         }]
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A rejoining arm that hides a nested exit but runs no fall-through content. The
// arm is a block like the loop body: its nested exit builds its own record under
// the arm's exit tag, and its content-free fall-through takes the `continued`
// tag with no arm-choice. The `if` gains a second exit tag (then=0, else=1),
// and the receiving pick a third alternative (continued=2).

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
  // The then arm exits (tag 0) after pushing content %i.
  // CHECK:          uir.if %{{.+}} {
  // CHECK-NEXT:       [[EX0:%.+]] = hir.choice 0([[I]])
  // CHECK-NEXT:       hir.opaque_pack([[EX0]])
  // CHECK:            uir.break
  // The else arm rejoins but hides a nested exit (tag 1) with an empty sub-chain.
  // CHECK:          } else {
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SUB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EX1:%.+]] = hir.choice 1([[SUB]])
  // CHECK-NEXT:         hir.opaque_pack([[EX1]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       uir.yield
  // CHECK-NEXT:     }
  // The content-free fall-through takes the continued tag (2).
  // CHECK:          [[CO:%.+]] = hir.choice 2()
  // CHECK-NEXT:     hir.opaque_pack([[CO]])
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
  // CHECK: hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[BARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var %{{.+}}, [[BARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 2 (): {
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Both arms of an exit-carrying `if` rejoin, and one hides a nested exit. Both
// arms carry fall-through content, so the arm-choice has one tag per rejoining
// arm (then=0, else=1). The else arm's nested exit builds an exit record (exit
// tag 0) while its fall-through content %f rides the arm-choice, not the exit
// record. The receiving continued alternative dispatches the two-tag arm-choice
// (replaying %e or %f), then runs the trailing segment %i in program order.

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
  // CHECK:          [[AC:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // Then rejoins with content %e as arm-choice tag 0.
  // CHECK-NEXT:       [[E:%.+]] = hir.add
  // CHECK-NEXT:       [[AC0:%.+]] = hir.choice 0([[E]])
  // CHECK-NEXT:       uir.yield [[AC0]]
  // CHECK-NEXT:     } else {
  // Else hides a nested exit (exit tag 0, empty payload) and rejoins with %f as
  // arm-choice tag 1.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SUB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EX:%.+]] = hir.choice 0([[SUB]])
  // CHECK-NEXT:         hir.opaque_pack([[EX]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[F:%.+]] = hir.add
  // CHECK-NEXT:       [[AC1:%.+]] = hir.choice 1([[F]])
  // CHECK-NEXT:       uir.yield [[AC1]]
  // CHECK-NEXT:     }
  // The continued record carries the trailing segment %i and the arm-choice.
  // CHECK:          [[CO:%.+]] = hir.choice 1([[I:%.+]], [[AC]])
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
  // CHECK: hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // The continued alternative replays the two-tag arm-choice, then segment %i.
  // CHECK-NEXT:     } 1 ([[IARG:%.+]], [[ACARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[ACARG]] [0 ([[EARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC:%.+]], [[EARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[FARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[FARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Both arms rejoin and both hide a nested exit. Each arm is exit-capable (two
// exit tags, then=0 and else=1) and rejoins with content (two arm-choice tags).
// The two exit records both end in an empty nested sub-chain, and the continued
// record replays whichever arm's content ran (%p or %q) before the segment %i.

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
  // CHECK:          [[AC:%.+]] = uir.if %{{.+}} : %{{.+}} {
  // The then arm: nested exit under exit tag 0, rejoin content %p as arm tag 0.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SP:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EP:%.+]] = hir.choice 0([[SP]])
  // CHECK-NEXT:         hir.opaque_pack([[EP]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[P:%.+]] = hir.add
  // CHECK-NEXT:       [[AC0:%.+]] = hir.choice 0([[P]])
  // CHECK-NEXT:       uir.yield [[AC0]]
  // The else arm: nested exit under exit tag 1, rejoin content %q as arm tag 1.
  // CHECK-NEXT:     } else {
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SQ:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EQ:%.+]] = hir.choice 1([[SQ]])
  // CHECK-NEXT:         hir.opaque_pack([[EQ]])
  // CHECK:              uir.break
  // CHECK-NEXT:       }
  // CHECK-NEXT:       [[Q:%.+]] = hir.add
  // CHECK-NEXT:       [[AC1:%.+]] = hir.choice 1([[Q]])
  // CHECK-NEXT:       uir.yield [[AC1]]
  // CHECK-NEXT:     }
  // CHECK:          [[CO:%.+]] = hir.choice 2([[I:%.+]], [[AC]])
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
  // CHECK: hir.replicate ([[REC:%.+]]) in %{{.+}} {
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[SUB0:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUB0]] [0 (): {
  // CHECK:            }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 ([[SUB1:%.+]]): {
  // CHECK-NEXT:       hir.pick [[SUB1]] [0 (): {
  // CHECK:            }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 2 ([[IARG:%.+]], [[ACARG:%.+]]): {
  // CHECK-NEXT:       hir.pick [[ACARG]] [0 ([[PARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC:%.+]], [[PARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 ([[QARG:%.+]]): {
  // CHECK-NEXT:         hir.write_var [[ACC]], [[QARG]]
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IARG]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A rejoining arm whose every path exits (its yield is unreachable): the nested
// `if d { break } else { continue }` leaves the arm's yield dead. The arm is a
// pure exit-capable arm — its content %y rides the exit records, and it makes no
// arm-choice contribution. Content before the nested exit (%y) is shared across
// the break and continue records, but both sit under the arm's single exit tag,
// so %y is replayed once before the nested pick.

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
  // CHECK:          uir.if %{{.+}} {
  // The break record: exited(then) carries %y then the nested break tag 0.
  // CHECK:            uir.if %{{.+}} {
  // CHECK-NEXT:         [[SB:%.+]] = hir.choice 0()
  // CHECK-NEXT:         [[EB:%.+]] = hir.choice 0([[Y:%.+]], [[SB]])
  // CHECK-NEXT:         hir.opaque_pack([[I:%.+]], [[EB]])
  // CHECK:              uir.break
  // The continue record: exited(then) carries %y then the nested continue tag 1.
  // CHECK-NEXT:       } else {
  // CHECK-NEXT:         [[SC:%.+]] = hir.choice 1()
  // CHECK-NEXT:         [[EC:%.+]] = hir.choice 0([[Y]], [[SC]])
  // CHECK-NEXT:         hir.opaque_pack([[I]], [[EC]])
  // CHECK:              uir.continue
  // CHECK-NEXT:       }
  // CHECK-NEXT:       uir.yield
  // CHECK-NEXT:     }
  // The c-false fall-through takes the continued tag (1) with no arm-choice.
  // CHECK:          [[CO:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.opaque_pack([[I]], [[CO]])
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
  // CHECK: hir.replicate ([[IREC:%.+]], [[REC:%.+]]) in %{{.+}} {
  // %x runs unconditionally, before the divergence pick.
  // CHECK-NEXT:     hir.write_var [[ACC:%.+]], [[IREC]]
  // CHECK-NEXT:     hir.pick [[REC]] [0 ([[YARG:%.+]], [[SUBARG:%.+]]): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[YARG]]
  // CHECK-NEXT:       hir.pick [[SUBARG]] [0 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 1 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       } 2 (): {
  // CHECK-NEXT:         hir.yield
  // CHECK-NEXT:       }]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     } 1 (): {
  // CHECK-NEXT:       hir.write_var [[ACC]], [[IREC]]
  // CHECK-NEXT:       hir.yield
  // CHECK-NEXT:     }]
  uir.return
}

//===----------------------------------------------------------------------===//
// Never-value forwarding: a control-flow result no path provides gets the
// bottom phase `never`. Every real-phase use of it is rewritten to a fresh
// `uir.unreachable_value` at the consuming phase, and the `never` result is
// left unused on its producer. `never` is non-viral, so the consumer keeps its
// own real phase.

// A never `uir.if` result (both arms short-yield) consumed by a same-phase op:
// the unreachable value and the consumer wrap into the consumer's phase, and
// the never-producing if wraps into its own phase with an unused result.
// CHECK-LABEL: uir.func @NeverConsumedSamePhase
uir.func @NeverConsumedSamePhase(%c: 0, %x: 0) -> (out: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: [[IF:%.+]] = uir.if %c : [[T]] {
  // CHECK-NEXT:   uir.yield
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   uir.yield
  // CHECK-NEXT: }
  %r = uir.if %c : %t {
    uir.yield
  } else {
    uir.yield
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
  // CHECK: uir.phase_return{{$}}
  // CHECK-NEXT: }
  %r = uir.if %c : %t {
    uir.yield
  } else {
    uir.yield
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
    uir.yield
  } else {
    uir.yield
  }
  // CHECK: uir.phase 0 {
  // CHECK-NEXT: uir.if %c : [[T]] {
  // CHECK: uir.phase_return{{$}}
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
