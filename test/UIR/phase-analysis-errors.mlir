// RUN: silicon-opt --test-phase-analysis2 %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Pure op operand available too late: arg at phase 0 used in const block.
// Two block arg errors (one per operand of hir.add), plus the pure op error.

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureOpTooLate(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    // hir.type_of shifts +1, so pushes latest = -1+1 = 0 to %a. Satisfies.
    // But the type_of result is at p(%a)-1 = -1. The unary op hir.type_of
    // itself is at -1.
    %ta = hir.type_of %a
    // expected-remark @+2 {{required by operand at phase -1}}
    // expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
    %sum = hir.add %a, %ta : %t
    // expected-remark @+1 {{required by yield operand}}
    uir.yield %sum : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Return from inside a const block.

uir.func @ReturnFromConstBlock(%val: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.expr pin -1 {
    // expected-error @+1 {{return from a phase-shifted block is not allowed}}
    uir.return %val -> (%t)
  }
  uir.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Break from inside a const block.

uir.func @BreakFromConstBlock(%cond: 0, %val: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    uir.expr pin -1 {
      // expected-error @+1 {{break from a phase-shifted block is not allowed}}
      uir.break %val : %t
    }
    uir.yield
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Continue from inside a const block.

uir.func @ContinueFromConstBlock(%cond: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  uir.loop {
    uir.expr pin -1 {
      // expected-error @+1 {{continue from a phase-shifted block is not allowed}}
      uir.continue
    }
    uir.yield
  }
  uir.return -> ()
}

// -----

//===----------------------------------------------------------------------===//
// Block arg yielded directly in const block.

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @BlockArgInConstBlock(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    // expected-remark @+1 {{required by yield operand}}
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Block arg used as signature type operand (phase 0, needs -1).

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @BlockArgInSignature(%a: 0) -> () {
  // expected-remark @+1 {{required by signature type of arg 0}}
  uir.signature (%a) -> ()
} {
  uir.return -> ()
}

// -----

//===----------------------------------------------------------------------===//
// Call feasibility error.

uir.func @CallFeasibilityTarget(%n: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %n -> (%t2)
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallFeasibilityError(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  // expected-remark @+1 {{required by call argument 0 at phase -1}}
  %r = uir.call @CallFeasibilityTarget(%a) : (%ta) -> (%ta) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r -> (%ta)
}

// -----

//===----------------------------------------------------------------------===//
// If condition at wrong phase.

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @IfCondTooLate(%cond: 0, %a: -1) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0)
} {
  %t = hir.int_type
  // expected-remark @+1 {{required by if condition at phase -1}}
  %0 = uir.if %cond : %t {
    uir.yield %a : %t
  } else {
    uir.yield %a : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Body-phase arg cannot satisfy const return (result: -1).

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ConstReturnBodyArg(%x: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // expected-remark @+1 {{required by return value at phase -1}}
  uir.return %x -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Pure op with body-phase operand cannot satisfy const return. Two error
// chains: %x→add (operand too late) and add→return (result too late).

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ConstReturnPureOp(%x: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // expected-remark @+2 {{required by operand at phase -1}}
  // expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
  %sum = hir.add %x, %c1 : %t
  // expected-remark @+1 {{required by return value at phase -1}}
  uir.return %sum -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Call result at phase 0 cannot satisfy const return.

uir.func @ConstReturnCallHelper(%a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %a -> (%t)
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ConstReturnCall(%x: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @+1 {{required by call argument 0 at phase -1}}
  %r = uir.call @ConstReturnCallHelper(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r -> (%tr)
}

// -----

//===----------------------------------------------------------------------===//
// Dyn value pinned at block phase: phase 1 cannot satisfy pin at phase 0.

// expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @DynPinAtBlockPhase(%a: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // expected-remark @+2 {{required by operand at phase 0}}
  // expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
  %sum = hir.add %a, %c1 : %t
  // expected-remark @+1 {{required by pin at phase 0}}
  %x = uir.pin %sum, 0 : !hir.any
  uir.return %x -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Dyn arg returned via non-dyn result: phase 1 cannot satisfy phase 0.

// expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @DynReturnMismatch(%a: 1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // expected-remark @+1 {{required by return value at phase 0}}
  uir.return %a -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Dyn condition on block-anchored uir.if: condition at phase 1 cannot satisfy
// the if at block phase 0. Needs uir.expr wrapping to float.

// expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @DynCondIf(%flag: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  %c0 = hir.constant_int 0 : %t
  // expected-remark @+1 {{required by if condition at phase 0}}
  %r = uir.if %flag : %t {
    uir.yield %c1 : %t
  } else {
    uir.yield %c0 : %t
  }
  uir.return %r -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Stacked const return mismatch: const arg (-1) cannot satisfy const const
// return (-2).

// expected-error @+1 {{value at phase -1 cannot satisfy requirement for phase -2}}
uir.func @StackedConstReturnMismatch(%x: -1) -> (result: -2) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // expected-remark @+1 {{required by return value at phase -2}}
  uir.return %x -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Nested const block: const arg (-1) used in const { const { ... } } which
// demands phase -2. Two error chains: arg→add operand, add→yield.

// expected-error @+1 {{value at phase -1 cannot satisfy requirement for phase -2}}
uir.func @NestedConstBlockError(%a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %inner = uir.expr pin -1 : %t {
      %c1 = hir.constant_int 1 : %t
      // expected-remark @+2 {{required by operand at phase -2}}
      // expected-error @+1 {{value at phase -1 cannot satisfy requirement for phase -2}}
      %sum = hir.add %a, %c1 : %t
      // expected-remark @+1 {{required by yield operand}}
      uir.yield %sum : %t
    }
    uir.yield %inner : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Dyn block with side-effecting call: result at phase 1 (call anchored at dyn
// block phase) returned via phase-0 result.

uir.func @DynBlockCallHelper() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %c42 = hir.constant_int 42 : %t
  uir.return %c42 -> (%t)
}

uir.func @DynBlockPhaseMismatch() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  // expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
  %0 = uir.expr pin 1 : %t {
    %tr = hir.int_type
    %v = uir.call @DynBlockCallHelper() : () -> (%tr) () -> !hir.any [] -> [0]
    uir.yield %v : %t
  }
  // expected-remark @+1 {{required by return value at phase 0}}
  uir.return %0 -> (%t)
}
