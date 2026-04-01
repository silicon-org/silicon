// RUN: silicon-opt --test-phase-analysis2 %s -split-input-file -verify-diagnostics

//===----------------------------------------------------------------------===//
// Pure op operand available too late: arg at phase 0 used in const block.
// The add is pure and would float, but the call anchors its result at phase -1,
// forcing the constraint. The add needs %a at phase -1, which is too late.

uir.func @PureOpTooLateHelper(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureOpTooLate(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ta = hir.type_of %a
    // expected-remark @+1 {{required by operand at phase -1}}
    %sum = hir.add %a, %ta : %t
    // Call anchored at -1 forces %sum to be available at -1.
    %tr = hir.int_type
    // expected-remark @+1 {{required by call argument 0 at phase -1}}
    %v = uir.call @PureOpTooLateHelper(%sum) : (%tr) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %v : %t
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
// Return from inside a dyn block (symmetric to @ReturnFromConstBlock).

uir.func @ReturnFromDynBlock(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.expr pin 1 {
    // expected-error @+1 {{return from a phase-shifted block is not allowed}}
    uir.return %x -> (%t)
  }
  uir.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Break from inside a dyn block (symmetric to @BreakFromConstBlock).

uir.func @BreakFromDynBlock(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    uir.expr pin 1 {
      // expected-error @+1 {{break from a phase-shifted block is not allowed}}
      uir.break %x : %t
    }
    uir.yield
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Continue from inside a dyn block (symmetric to @ContinueFromConstBlock).

uir.func @ContinueFromDynBlock(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  uir.loop {
    uir.expr pin 1 {
      // expected-error @+1 {{continue from a phase-shifted block is not allowed}}
      uir.continue
    }
    uir.yield
  }
  uir.return -> ()
}

// -----

//===----------------------------------------------------------------------===//
// Break inside nested dyn blocks: outer dyn at +1, loop at +1, inner dyn at
// +2. Break targets loop at +1, but block phase is +2.

uir.func @BreakFromNestedDynBlock(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    %1 = uir.loop : %t {
      uir.expr pin 1 {
        // expected-error @+1 {{break from a phase-shifted block is not allowed}}
        uir.break %x : %t
      }
      uir.yield
    }
    uir.yield %1 : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Block arg passed to call in const block: %a at phase 0 cannot satisfy the
// call's arg requirement at phase -1.

uir.func @IdentityHelper(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @BlockArgInConstBlock(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @+1 {{required by call argument 0 at phase -1}}
    %v = uir.call @IdentityHelper(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %v : %t
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
  // expected-remark @+1 {{required by operand at phase -1}}
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

uir.func @ConstReturnCall(%x: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-error @+1 {{call result 0 at phase 0 cannot satisfy requirement for phase -1}}
  %r = uir.call @ConstReturnCallHelper(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  // expected-remark @+1 {{required by return value at phase -1}}
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
  // expected-remark @+1 {{required by operand at phase 0}}
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
// Nested const block: const arg (-1) passed to a call inside
// const { const { ... } } which anchors at phase -2. The call demands its arg
// at phase -2, but %a is only at -1.

uir.func @NestedIdentityHelper(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// expected-error @+1 {{value at phase -1 cannot satisfy requirement for phase -2}}
uir.func @NestedConstBlockError(%a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %inner = uir.expr pin -1 : %t {
      %ta = hir.int_type
      %tr = hir.int_type
      // Call anchored at -2. Arg needs phase -2, but %a is at -1.
      // expected-remark @+1 {{required by call argument 0 at phase -2}}
      %v = uir.call @NestedIdentityHelper(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
      uir.yield %v : %t
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
  %0 = uir.expr pin 1 : %t {
    %tr = hir.int_type
    // expected-error @+1 {{value at phase 1 cannot satisfy requirement for phase 0}}
    %v = uir.call @DynBlockCallHelper() : () -> (%tr) () -> !hir.any [] -> [0]
    // expected-remark @+1 {{required by yield operand}}
    uir.yield %v : %t
  }
  // expected-remark @+1 {{required by return value at phase 0}}
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Phase boundary off-by-one: pure op earliest > latest.

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureBoundaryError(%a: 0) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %lit = hir.constant_int 1 : %t
  // expected-remark @+1 {{required by operand at phase -1}}
  %sum = hir.add %a, %lit : %t
  // expected-remark @+1 {{required by return value at phase -1}}
  uir.return %sum -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Phase boundary off-by-one: call arg one phase too late.

uir.func @BoundaryCallHelper(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x -> (%t2)
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallBoundaryError(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @+1 {{required by call argument 0 at phase -1}}
  %r = uir.call @BoundaryCallHelper(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r -> (%tr)
}

// -----

//===----------------------------------------------------------------------===//
// Phase boundary off-by-one: dyn result demands call one phase too early.

uir.func @DynResultHelper(%x: 0) -> (result: 1) {
  %t = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t) -> (%t2)
} {
  %t3 = hir.int_type
  uir.return %x -> (%t3)
}

uir.func @DynResultBoundaryError(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-error @+1 {{call result 0 at phase 1 cannot satisfy requirement for phase 0}}
  %r = uir.call @DynResultHelper(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [1]
  // expected-remark @+1 {{required by return value at phase 0}}
  uir.return %r -> (%tr)
}

// -----

//===----------------------------------------------------------------------===//
// Short-circuit || inside const block with non-const RHS. Const block at -1.
// If at -1. Else branch needs b at -1. b at 0. Error.

uir.func @IdentityHelper2(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @OrErrorInConstBlock(%a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %true_val = hir.constant_bool <true>
  %0 = uir.expr pin -1 : %t {
    %or = uir.if %a : %t {
      uir.yield %true_val : %t
    } else {
      // expected-remark @below {{required by yield operand}}
      uir.yield %b : %t
    }
    // Anchor the if result at -1 via a call.
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %v = uir.call @IdentityHelper2(%or) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %v : %t
  }
  uir.return %0 -> (%t)
}

// -----

//===----------------------------------------------------------------------===//
// Pin prevents floating: a+42 would float to -1, but pin fixes it at phase 0.
// Passing to a const arg at -1 fails because the pin anchors the result.

uir.func @NeedsConstForPin(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x -> (%t)
}

uir.func @PinPreventsFloat(%a: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %fortytwo = hir.constant_int 42 : %t
  %sum = hir.add %a, %fortytwo : %t
  // Pin at phase 0 (let binding). Without the pin, %sum would float to -1.
  // expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
  %x = uir.pin %sum, 0 : !hir.any
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %r = uir.call @NeedsConstForPin(%x) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r -> (%tr)
}