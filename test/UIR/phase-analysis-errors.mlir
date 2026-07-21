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
  uir.return %x : %t
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
  uir.return %0 : %t
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
    uir.return %val : %t
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
    uir.continue
  }
  uir.return %0 : %t
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
    uir.continue
  }
  uir.return
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
    uir.return %x : %t
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
    uir.continue
  }
  uir.return %0 : %t
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
    uir.continue
  }
  uir.return
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
      uir.continue
    }
    uir.yield %1 : %t
  }
  uir.return %0 : %t
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
  uir.return %x : %t
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
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Block arg used as signature type operand (phase 0, needs -1).

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @BlockArgInSignature(%a: 0) -> () {
  // expected-remark @+1 {{required by signature type of arg 0}}
  uir.signature (%a) -> ()
} {
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// Call feasibility error.

uir.func @CallFeasibilityTarget(%n: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %n : %t2
}

// expected-error @+1 {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallFeasibilityError(%a: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %ta = hir.int_type
  // expected-remark @+1 {{required by call argument 0 at phase -1}}
  %r = uir.call @CallFeasibilityTarget(%a) : (%ta) -> (%ta) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r : %ta
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
  uir.return %x : %t
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
  uir.return %sum : %t
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
  uir.return %a : %t
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
  uir.return %r : %tr
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
  uir.return %x : %t
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
  uir.return %a : %t
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
  uir.return %r : %t
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
  uir.return %x : %t
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
  uir.return %x : %t
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
  uir.return %0 : %t
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
  uir.return %c42 : %t
}

uir.func @DynBlockPhaseMismatch() -> (result: 0) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    %tr = hir.int_type
    // expected-error @+1 {{call result 0 at phase 1 cannot satisfy requirement for phase 0}}
    %v = uir.call @DynBlockCallHelper() : () -> (%tr) () -> !hir.any [] -> [0]
    // expected-remark @+1 {{required by yield operand}}
    uir.yield %v : %t
  }
  // expected-remark @+1 {{required by return value at phase 0}}
  uir.return %0 : %t
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
  uir.return %sum : %t
}

// -----

//===----------------------------------------------------------------------===//
// Phase boundary off-by-one: call arg one phase too late.

uir.func @BoundaryCallHelper(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
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
  uir.return %r : %tr
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
  uir.return %x : %t3
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
  uir.return %r : %tr
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
  uir.return %x : %t
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
  uir.return %0 : %t
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
  uir.return %x : %t
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
  uir.return %r : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Balanced nesting off-by-one: 3 const + 2 dyn = net -1.

uir.func @ReturnAlmostCancels(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.expr pin -1 {
    uir.expr pin 1 {
      uir.expr pin -1 {
        uir.expr pin 1 {
          uir.expr pin -1 {
            // expected-error @below {{return from a phase-shifted block is not allowed}}
            uir.return %x : %t
          }
          uir.unreachable
        }
        uir.unreachable
      }
      uir.unreachable
    }
    uir.unreachable
  }
  uir.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// break with unbalanced nesting around non-zero loop.

uir.func @BreakUnbalancedNonzeroLoop(%x: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %1 = uir.loop : %t {
      uir.expr pin 1 {
        // expected-error @below {{break from a phase-shifted block is not allowed}}
        uir.break %x : %t
      }
      uir.continue
    }
    uir.yield %1 : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Balanced nesting: values DON'T cancel. const { dyn { a } } with a call
// in the const block to pin the dyn result at -1.

uir.func @BalancedIdentity(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @BalancedConstDynValueError(%a: 0) -> (result: 0) {
  %t0 = hir.int_type
  uir.signature (%t0) -> (%t0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %1 = uir.expr pin 1 : %t {
      // expected-remark @below {{required by yield operand}}
      uir.yield %a : %t
    }
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %r = uir.call @BalancedIdentity(%1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %r : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Symmetric dyn side: dyn { const { dyn { a } } } with call pinning.

uir.func @BalancedIdentity2(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @BalancedDynConstDynValueError(%a: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    %1 = uir.expr pin -1 : %t {
      %2 = uir.expr pin 1 : %t {
        // expected-remark @below {{required by yield operand}}
        uir.yield %a : %t
      }
      %ta = hir.int_type
      %tr = hir.int_type
      // expected-remark @below {{required by call argument 0 at phase 0}}
      %r = uir.call @BalancedIdentity2(%2) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
      uir.yield %r : %t
    }
    uir.yield %1 : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Pure op bottleneck: identity call pins at block phase -1.

uir.func @BottleneckIdentity(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureBottleneckSingle(%a: -1, %b: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  uir.signature (%0, %1, %2) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ab = hir.add %a, %b : %t
    // expected-remark @below {{required by operand at phase -1}}
    %abc = hir.add %ab, %c : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %r = uir.call @BottleneckIdentity(%abc) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %r : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Two bottlenecks: both b and c at phase 0.

uir.func @BottleneckIdentity2(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureBottleneckTwo(%a: -1, %b: 0, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  uir.signature (%0, %1, %2) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    // expected-remark @below {{required by operand at phase -1}}
    %ab = hir.add %a, %b : %t
    // expected-remark @below {{required by operand at phase -1}}
    %abc = hir.add %ab, %c : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %r = uir.call @BottleneckIdentity2(%abc) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %r : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Nested pure op bottleneck through mul(add(a,b), add(a,c)).

uir.func @BottleneckIdentity3(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @PureBottleneckNested(%a: -1, %b: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  uir.signature (%0, %1, %2) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ab = hir.add %a, %b : %t
    // expected-remark @below {{required by operand at phase -1}}
    %ac = hir.add %a, %c : %t
    // expected-remark @below {{required by operand at phase -1}}
    %r = hir.mul %ab, %ac : %t
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %call = uir.call @BottleneckIdentity3(%r) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %call : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Signature type consuming: N at 0, type latest = -1. Error.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @SigReturnTypeError(%N: 0) -> (result: 0) {
  %t0 = hir.int_type
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by signature type of result 0}}
  uir.signature (%t0) -> (%ut)
} {
  %t = hir.int_type
  %fortytwo = hir.constant_int 42 : %t
  uir.return %fortytwo : %t
}

// -----

//===----------------------------------------------------------------------===//
// Const return but N only at -1. Type latest = -2. Error.

// expected-error @below {{value at phase -1 cannot satisfy requirement for phase -2}}
uir.func @SigConstReturnTypeError(%N: -1) -> (result: -1) {
  %t0 = hir.int_type
  // expected-remark @below {{required by operand at phase -2}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by signature type of result 0}}
  uir.signature (%t0) -> (%ut)
} {
  %t = hir.int_type
  %fortytwo = hir.constant_int 42 : %t
  %r = uir.expr pin -1 : %t {
    uir.yield %fortytwo : %t
  }
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Dyn call inside const block: floating expr lets call float to -2 (for dyn
// result at -1). Arg at -2. a at -1 > -2. ERROR.

uir.func @DynCallDeferred(%x: 0) -> (result: 1) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  %0 = uir.expr pin 1 : %t2 { uir.yield %x : %t2 }
  uir.return %0 : %t2
}

uir.func @DynCallPin(%x: 0) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase -1 cannot satisfy requirement for phase -2}}
uir.func @DynCallInConstError(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    %ta = hir.int_type
    %tr = hir.int_type
    %r = uir.expr : %t {
      // expected-remark @below {{required by call argument 0 at phase -2}}
      // expected-remark @below {{required by call result 0 at phase -1}}
      %inner = uir.call @DynCallDeferred(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [1]
      // expected-remark @below {{required by yield operand}}
      uir.yield %inner : %t
    }
    %ta2 = hir.int_type
    %tr2 = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %pinned = uir.call @DynCallPin(%r) : (%ta2) -> (%tr2) (!hir.any) -> !hir.any [0] -> [0]
    uir.yield %pinned : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Multi-consumer error: let-pinned call result at 0 used as const arg at -1.

uir.func @MultiConsumerComputeErr(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  %r = hir.add %x, %c1 : %t
  uir.return %r : %t
}

uir.func @MultiConsumerNeedsConstErr(%x: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @MultiConsumerError(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %ta = hir.int_type
  %tr = hir.int_type
  %call = uir.call @MultiConsumerComputeErr(%a) : (%ta) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  // expected-error @below {{pinned value at phase 0 cannot satisfy requirement for phase -1}}
  %r = uir.pin %call, 0 : !hir.any
  %ta2 = hir.int_type
  %tr2 = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %nc = uir.call @MultiConsumerNeedsConstErr(%r) : (%ta2) -> (%tr2) (!hir.any) -> !hir.any [-1] -> [0]
  %sum = hir.add %nc, %r : %t
  uir.return %sum : %t
}

// -----

//===----------------------------------------------------------------------===//
// return in balanced nesting inside outer const block. Net -1 from body. ERROR.

uir.func @ReturnBalancedInConst(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.expr pin -1 {
    uir.expr pin 1 {
      uir.expr pin -1 {
        // expected-error @below {{return from a phase-shifted block is not allowed}}
        uir.return %x : %t
      }
      uir.unreachable
    }
    uir.unreachable
  }
  uir.unreachable
}

// -----

//===----------------------------------------------------------------------===//
// Multiple break sites: one OK, one in const block ERROR.

uir.func @MultiBreakOneError(%flag: 0, %a: 0, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    uir.if %flag {
      uir.break %a : %t
    } else {
      uir.yield
    }
    uir.expr pin -1 {
      // expected-error @below {{break from a phase-shifted block is not allowed}}
      uir.break %b : %t
    }
    uir.continue
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Continue and break mixed: break OK, continue in const block ERROR.

uir.func @ContinueBreakMixed(%flag: 0, %a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> (%t0)
} {
  %t = hir.int_type
  %0 = uir.loop : %t {
    uir.if %flag {
      uir.break %a : %t
    } else {
      uir.yield
    }
    uir.expr pin -1 {
      // expected-error @below {{continue from a phase-shifted block is not allowed}}
      uir.continue
    }
    uir.continue
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// CSE-shared pure op error: b+1 at 0 can't satisfy const arg at -1.

uir.func @SharedNeedsConstErr(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @SharedOpError(%a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  // expected-remark @below {{required by operand at phase -1}}
  %bp1 = hir.add %b, %c1 : %t
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %r1 = uir.call @SharedNeedsConstErr(%bp1) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  %sum = hir.add %bp1, %bp1 : %t
  %r2 = hir.add %r1, %sum : %t
  uir.return %r2 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Call with const arg inside if whose result feeds into an outer const arg.
// Inner call at block phase 0, arg offset -1: arg at -1. b at 0 > -1. ERROR.

uir.func @FloatedIfOuter(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

uir.func @FloatedIfInner(%y: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %y : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallInFloatedIf(%a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %bt = hir.bool_type
  %c0 = hir.constant_int 0 : %t
  %cond = hir.gt %a, %c0 : %bt
  %ifr = uir.if %cond : %t {
    %ta = hir.int_type
    %tr = hir.int_type
    // expected-remark @below {{required by call argument 0 at phase -1}}
    %r = uir.call @FloatedIfInner(%b) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
    uir.yield %r : %t
  } else {
    uir.yield %a : %t
  }
  %ta2 = hir.int_type
  %tr2 = hir.int_type
  %r2 = uir.call @FloatedIfOuter(%ifr) : (%ta2) -> (%tr2) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r2 : %tr2
}

// -----

//===----------------------------------------------------------------------===//
// Loop version: inner call with const arg inside loop whose result feeds
// into outer const arg.

uir.func @FloatedLoopOuter2(%x: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %x : %t2
}

uir.func @FloatedLoopInner(%y: -1) -> (result: 0) {
  %t = hir.int_type
  uir.signature (%t) -> (%t)
} {
  %t2 = hir.int_type
  uir.return %y : %t2
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @LoopCallCascade(%flag: -1, %a: -1, %b: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  %loopr = uir.loop : %t {
    uir.if %flag {
      %ta = hir.int_type
      %tr = hir.int_type
      // expected-remark @below {{required by call argument 0 at phase -1}}
      %r = uir.call @FloatedLoopInner(%b) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
      uir.break %r : %t
    } else {
      uir.unreachable
    }
    uir.continue
  }
  %ta2 = hir.int_type
  %tr2 = hir.int_type
  %r2 = uir.call @FloatedLoopOuter2(%loopr) : (%ta2) -> (%tr2) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r2 : %tr2
}

// -----

//===----------------------------------------------------------------------===//
// Error cascade through nested floating calls: const return demands -1,
// outer call floats to -1, inner call floats to -1, arg b at 0 > -1.

uir.func @CascadeAdd(%x: 0, %y: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %r = hir.add %x, %y : %t
  uir.return %r : %t
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @NestedCallCascade(%a: -1, %b: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %ta1 = hir.int_type
  %ta2 = hir.int_type
  %tr1 = hir.int_type
  %fortytwo = hir.constant_int 42 : %t
  // Floating expr lets inner call float to -1.
  %inner = uir.expr : %t {
    // expected-remark @below {{required by call argument 0 at phase -1}}
    // expected-remark @below {{required by call result 0 at phase -1}}
    %r = uir.call @CascadeAdd(%b, %fortytwo) : (%ta1, %ta2) -> (%tr1) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
    // expected-remark @below {{required by yield operand}}
    uir.yield %r : %t
  }
  %ta3 = hir.int_type
  %ta4 = hir.int_type
  %tr2 = hir.int_type
  // Floating expr lets outer call float to -1.
  %outer = uir.expr : %t {
    // expected-remark @below {{required by call argument 1 at phase -1}}
    // expected-remark @below {{required by call result 0 at phase -1}}
    %r = uir.call @CascadeAdd(%a, %inner) : (%ta3, %ta4) -> (%tr2) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
    // expected-remark @below {{required by yield operand}}
    uir.yield %r : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %outer : %t
}

// -----

//===----------------------------------------------------------------------===//
// Zero-use error: non-const arg b in side-effect inside const block.

func.func private @side_effect_err(!hir.any)

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ZeroUseError(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin -1 : %t {
    // expected-remark @below {{required by operand at phase -1}}
    func.call @side_effect_err(%b) : (!hir.any) -> ()
    %c0 = hir.constant_int 0 : %t
    uir.yield %c0 : %t
  }
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Loop as const return: break value at 0 but const return demands -1.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @LoopConstReturnError(%x: 0) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %c1 = hir.constant_int 1 : %t
  %0 = uir.loop : %t {
    // expected-remark @below {{required by operand at phase -1}}
    %sum = hir.add %x, %c1 : %t
    // expected-remark @below {{required by yield operand}}
    uir.break %sum : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Dyn break value error: deferred() result at +1, return demands 0.

uir.func @DeferredForLoop() -> (result: 1) {
  %0 = hir.int_type
  uir.signature () -> (%0)
} {
  %t = hir.int_type
  %0 = uir.expr pin 1 : %t {
    %c42 = hir.constant_int 42 : %t
    uir.yield %c42 : %t
  }
  uir.return %0 : %t
}

uir.func @LoopDynBreakError() -> (result: 0) {
  %0 = hir.int_type
  uir.signature () -> (%0)
} {
  %t = hir.int_type
  %ta = hir.int_type
  %0 = uir.loop : %t {
    %tr = hir.int_type
    // expected-error @below {{call result 0 at phase 1 cannot satisfy requirement for phase 0}}
    %call = uir.call @DeferredForLoop() : () -> (%tr) () -> !hir.any [] -> [1]
    // expected-remark @below {{required by yield operand}}
    uir.break %call : %t
  }
  // expected-remark @below {{required by return value at phase 0}}
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// if as const return: one branch has body-phase value. The if at phase 0
// selects between distinct values for a result demanded at phase -1.

uir.func @IfConstReturnError(%x: -1, %y: 0) -> (result: -1) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%0)
} {
  %t = hir.int_type
  %bt = hir.bool_type
  %c0 = hir.constant_int 0 : %t
  %cmp = hir.gt %x, %c0 : %bt
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %0 = uir.if %cmp : %t {
    // expected-note @below {{value provided here}}
    uir.yield %x : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %y : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Result Phase Floor Tests
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// if at phase 0, both branches yield distinct const values. The if selects
// between them, so the result can't be at -1.

uir.func @IfDistinctYieldFloorError(%sel: -1, %a: -1, %b: -1) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %r = uir.if %sel : %t {
    // expected-note @below {{value provided here}}
    uir.yield %a : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %b : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Both arms of a phase-0 if yield `uir.unreachable_value` poison. Provider-ness
// is a control-flow property, so both reachable arms provide the slot: it is a
// two-provider selection at phase 0, not a `never` result, and cannot deliver
// the const (phase -1) function result. The floor error fires exactly as for
// concrete values.

uir.func @PoisonArmsConstFloorError(%c: 0) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %r = uir.if %c : %t {
    %uv0 = uir.unreachable_value
    // expected-note @below {{value provided here}}
    uir.yield %uv0 : %t
  } else {
    %uv1 = uir.unreachable_value
    // expected-note @below {{value provided here}}
    uir.yield %uv1 : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Nested if: both branches yield distinct const values at phase -1. Since the
// if is at phase 0 (function body), this is an error.

uir.func @NestedIfFloorError(%a: -1, %b: -1, %c: -1) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t0)
} {
  %t = hir.int_type
  %bt = hir.bool_type
  %c0 = hir.constant_int 0 : %t
  %cmp = hir.gt %a, %c0 : %bt
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %0 = uir.if %cmp : %t {
    // expected-note @below {{value provided here}}
    uir.yield %a : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %b : %t
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %0 : %t
}

// -----

//===----------------------------------------------------------------------===//
// Loop at phase 0, two breaks with distinct const values. Floor violated.

uir.func @LoopDistinctBreakFloorError(%a: -1, %b: -1, %flag: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %r = uir.loop : %t {
    uir.if %flag {
      // expected-note @below {{value provided here}}
      uir.break %a : %t
    } else {
      // expected-note @below {{value provided here}}
      uir.break %b : %t
    }
    uir.continue
  }
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Function returns: two returns provide distinct values for a const result.
// The function body is at phase 0, so the result can't be at -1.

// expected-error @below {{result 0 at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
uir.func @FuncDistinctReturnFloorError(%a: -1, %b: -1, %flag: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  uir.if %flag {
    // expected-note @below {{return value provided here}}
    uir.return %a : %t
  }
  // expected-note @below {{return value provided here}}
  uir.return %b : %t
}

// -----

//===----------------------------------------------------------------------===//
// Function returns: two returns provide the *same* SSA value for a const
// result. A result provided by more than one return is a selection at the
// body phase regardless of the values' identity, so this errors like the
// distinct-value case.

// expected-error @below {{result 0 at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
uir.func @FuncSameReturnFloorError(%x: -1, %flag: 0) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  uir.if %flag {
    // expected-note @below {{return value provided here}}
    uir.return %x : %t
  }
  // expected-note @below {{return value provided here}}
  uir.return %x : %t
}

// -----

//===----------------------------------------------------------------------===//
// if inside a const block at phase -1, yields distinct values for a result
// demanded at phase -2 (from a doubly-const call arg). The if is at phase -1,
// so the result can't be below -1.

uir.func @NeedsDoubleConstArg(%x: -2) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @IfInConstFloorError(%a: -2, %b: -2, %flag: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  %sel = uir.expr pin -1 : %t {
    // expected-error @below {{result at phase -2 is selected by control flow at phase -1; result phase must be >= -1}}
    %r = uir.if %flag : %t {
      // expected-note @below {{value provided here}}
      uir.yield %a : %t
    } else {
      // expected-note @below {{value provided here}}
      uir.yield %b : %t
    }
    // expected-remark @below {{required by yield operand}}
    uir.yield %r : %t
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -2}}
  %result = uir.call @NeedsDoubleConstArg(%sel) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-2] -> [0]
  uir.return %result : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Short-circuit && with both const args, result fed as const arg to a call.
// The if at phase 0 selects between distinct values (%b and %false_val) but
// the call demands the result at phase -1. Floor violated.

uir.func @NeedsConstBoolFloor(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @ShortCircuitFloorError(%a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %false_val = hir.constant_bool <false>
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %and = uir.if %a : %t {
    // expected-note @below {{value provided here}}
    uir.yield %b : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %false_val : %t
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %r = uir.call @NeedsConstBoolFloor(%and) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r : %tr
}
// -----

//===----------------------------------------------------------------------===//
// References get no special treatment from the floor rule (§11): an if at
// phase 0 selecting between two const references cannot deliver the selected
// reference at -1, exactly like any other value.

uir.func @NeedsConstRef(%r: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c = hir.constant_int 1 : %t
  uir.return %c : %t
}

uir.func @SelectRefFloorError(%sel: 0, %x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %v0 = uir.expr pin -1 : %t {
    %v = hir.var : %t
    hir.write_var %v, %x : %t
    uir.yield %v : %t
  }
  %v1 = uir.expr pin -1 : %t {
    %v = hir.var : %t
    hir.write_var %v, %x : %t
    uir.yield %v : %t
  }
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %sel_ref = uir.if %sel : %t {
    // expected-note @below {{value provided here}}
    uir.yield %v0 : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %v1 : %t
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %r = uir.call @NeedsConstRef(%sel_ref) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Ordinary causality on the reference operand: a write cannot consume a
// reference before the phase that creates it. (Home-phase legality is an
// evaluation-time check, not a PhaseAnalysis error; see sampling-model.md §11.)

uir.func @WriteBeforeRefExists(%x: -1) -> () {
  %t0 = hir.int_type
  uir.signature (%t0) -> ()
} {
  %t = hir.int_type
  // expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
  %v = hir.var : %t
  uir.expr pin -1 {
    // expected-remark @below {{required by operand at phase -1}}
    hir.write_var %v, %x : %t
    uir.yield
  }
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// Same causality rule for reads: the reference does not exist yet at -1.

uir.func @ReadBeforeRefExists(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  // expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
  %v = hir.var : %t
  hir.write_var %v, %x : %t
  %r = uir.expr pin -1 : %t {
    // expected-remark @below {{required by operand at phase -1}}
    %r0 = hir.read_var %v : %t
    uir.yield %r0 : %t
  }
  uir.return %x : %t
}

// -----

//===----------------------------------------------------------------------===//
// If nested in a floating expr with its result demanded at -1: the demand
// resolves the floating expr at -1, which anchors the if and checks its
// condition. Previously region ops without a resolved phase were silently
// skipped, leaving the condition unchecked.

uir.func @FloatingIfCondHelper(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @FloatingIfCondTooLate(%sel: 0, %a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  %r = uir.expr : %t {
    // expected-remark @below {{required by if condition at phase -1}}
    %s = uir.if %sel : %t {
      uir.yield %a : %t
    } else {
      uir.yield %b : %t
    }
    // expected-remark @below {{required by yield operand}}
    uir.yield %s : %t
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %c = uir.call @FloatingIfCondHelper(%r) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %c : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Result phase floor hit by a second, tighter demand: the pin records the
// floored result phase 0, then the const call arg demands -1 and re-enters
// the floor check.

uir.func @FloorSecondDemandHelper(%x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @FloorErrorOnSecondDemand(%sel: 0, %a: -1, %b: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %r = uir.if %sel : %t {
    // expected-note @below {{value provided here}}
    uir.yield %a : %t
  } else {
    // expected-note @below {{value provided here}}
    uir.yield %b : %t
  }
  %p = uir.pin %r, 0 : !hir.any
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %c = uir.call @FloorSecondDemandHelper(%r) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %c : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Type operand diagnostics: each op kind that carries type operands reports
// its own demand context when the type value arrives too late. All cases use
// a computed type (hir.uint_type %N) so the type cannot float.

// If result type: demand at blockPhase - 1 = -1, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @IfResultTypeTooLate(%N: 0, %sel: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by if result type at phase -1}}
  %r = uir.if %sel : %ut {
    uir.yield %sel : %ut
  } else {
    uir.yield %sel : %ut
  }
  uir.return %r : %ut
}

// -----

// Loop result type: demand at blockPhase - 1 = -1, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @LoopResultTypeTooLate(%N: 0, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by loop result type at phase -1}}
  %r = uir.loop : %ut {
    uir.break %x : %ut
  }
  uir.return %r : %ut
}

// -----

// Pinned expr result type: expr at -1, demand at -2, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -2}}
uir.func @ExprResultTypeTooLate(%N: 0, %x: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // expected-remark @below {{required by operand at phase -2}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by expr result type at phase -2}}
  %r = uir.expr pin -1 : %ut {
    uir.yield %x : %ut
  }
  uir.return %r : %ut
}

// -----

// Loop init value: the iteration argument lives at the loop's phase 0, the
// init value arrives at phase 1.

uir.func @LoopInitDynHelper() -> (result: 1) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %r = uir.expr pin 1 : %t {
    %c = hir.constant_int 1 : %t
    uir.yield %c : %t
  }
  uir.return %r : %t
}

uir.func @LoopInitTooLate(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %tr = hir.int_type
  // expected-error @below {{call result 0 at phase 1 cannot satisfy requirement for phase 0}}
  %d = uir.call @LoopInitDynHelper() : () -> (%tr) () -> !hir.any [] -> [1]
  // expected-remark @below {{required by loop init value at phase 0}}
  %r = uir.loop (%it = %d : %t) : %t {
    uir.break %it : %t
  }
  uir.return %r : %t
}

// -----

// Loop init type: demand at blockPhase - 1 = -1, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @LoopInitTypeTooLate(%N: 0, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by loop init type at phase -1}}
  %r = uir.loop (%it = %x : %ut) : %t {
    uir.break %it : %t
  }
  uir.return %r : %t
}

// -----

// Continue carried value: flows into the next iteration's argument at the
// loop's phase 0, but arrives at phase 1.

uir.func @ContinueDynHelper() -> (result: 1) {
  %t = hir.int_type
  uir.signature () -> (%t)
} {
  %t = hir.int_type
  %r = uir.expr pin 1 : %t {
    %c = hir.constant_int 1 : %t
    uir.yield %c : %t
  }
  uir.return %r : %t
}

uir.func @ContinueValueTooLate(%x: 0, %flag: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %tr = hir.int_type
  // expected-error @below {{call result 0 at phase 1 cannot satisfy requirement for phase 0}}
  %d = uir.call @ContinueDynHelper() : () -> (%tr) () -> !hir.any [] -> [1]
  %r = uir.loop (%it = %x : %t) : %t {
    uir.if %flag {
      uir.break %it : %t
    }
    // expected-remark @below {{required by continue value at phase 0}}
    uir.continue %d : %t
  }
  uir.return %r : %t
}

// -----

// Continue type operand: demand at blockPhase - 1 = -1, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ContinueTypeTooLate(%N: 0, %x: 0, %flag: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  %r = uir.loop (%it = %x : %t) : %t {
    uir.if %flag {
      uir.break %it : %t
    }
    // expected-remark @below {{required by continue type operand}}
    uir.continue %it : %ut
  }
  uir.return %r : %t
}

// -----

// Call typeOfArgs: arg offset 0, type demand at -1, %N at 0.

uir.func @CallArgTypeTarget(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallArgTypeTooLate(%N: 0, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  %tr = hir.int_type
  // expected-remark @below {{required by type of call argument 0}}
  %r = uir.call @CallArgTypeTarget(%x) : (%ut) -> (%tr) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %tr
}

// -----

// Call typeOfResults: result offset 0, type demand at -1, %N at 0.

uir.func @CallResultTypeTarget(%x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  uir.return %x : %t
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @CallResultTypeTooLate(%N: 0, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %ta = hir.int_type
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by type of call result 0}}
  %r = uir.call @CallResultTypeTarget(%x) : (%ta) -> (%ut) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %ut
}

// -----

// Return type operand: result at phase 0, type demand at -1, %N at 0.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @ReturnTypeTooLate(%N: 0, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by return type operand}}
  uir.return %x : %ut
}

// -----

// Yield type operand: the if result is demanded at 0, so the yield's type
// operand is demanded at -1; %N at 0. The if's own result type uses a
// floating int_type so only the yield's type operand fails.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @YieldTypeTooLate(%N: 0, %sel: 0, %a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  %r = uir.if %sel : %t {
    // expected-remark @below {{required by yield type operand}}
    uir.yield %a : %ut
  } else {
    uir.yield %a : %t
  }
  // expected-remark @below {{required by return value at phase 0}}
  uir.return %r : %t
}

// -----

// Side-effecting op demand context: the var reference is demanded at -1 by a
// const call arg, anchoring the floating expr at -1; processing the block
// fails on the write consuming the phase-0 value, and the remark chain
// includes the side-effecting op that carried the demand.

uir.func @VarRefConstTarget(%r: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c = hir.constant_int 1 : %t
  uir.return %c : %t
}

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @SideEffectDemandContext(%a: 0) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %te = hir.int_type
  %v = uir.expr : %te {
    // expected-remark @below {{required by side-effecting op at phase -1}}
    %v0 = hir.var : %t
    // expected-remark @below {{required by operand at phase -1}}
    hir.write_var %v0, %a : %t
    // expected-remark @below {{required by yield operand}}
    uir.yield %v0 : %te
  }
  %ta = hir.int_type
  %tr = hir.int_type
  // expected-remark @below {{required by call argument 0 at phase -1}}
  %r = uir.call @VarRefConstTarget(%v) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
  uir.return %r : %tr
}

// -----

//===----------------------------------------------------------------------===//
// Variable declared with a computed type that arrives too late: the varType
// operand of hir.var follows the -1 rule like any other type operand.

// expected-error @below {{value at phase 0 cannot satisfy requirement for phase -1}}
uir.func @VarTypeTooLate(%N: 0, %x: 0) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0, %t1) -> ()
} {
  // expected-remark @below {{required by operand at phase -1}}
  %ut = hir.uint_type %N
  // expected-remark @below {{required by operand at phase -1}}
  %v = hir.var : %ut
  hir.write_var %v, %x : %ut
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// Causality on a call's earlier half under later-phase CF: the call anchors at
// the if's phase 1, its const arg (offset -1) demands phase 0, but is fed the
// phase-1 dyn value. The D-rule places the earlier half at phase 0 regardless
// of the surrounding CF, so the argument arrives too late.

uir.func @EarlierHalfErrTarget(%n: -1) -> (result: 0) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  uir.signature (%t0) -> (%t1)
} {
  %t = hir.int_type
  %c = hir.constant_int 1 : %t
  uir.return %c : %t
}

// expected-error @below {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @CallEarlierHalfArgTooLate(%d: 1, %cond: 1) -> (result: 1) {
  %t0 = hir.int_type
  %t1 = hir.bool_type
  %t2 = hir.int_type
  uir.signature (%t0, %t1) -> (%t2)
} {
  %t = hir.int_type
  %r = uir.expr pin 1 : %t {
    %s = uir.if %cond : %t {
      %ta = hir.int_type
      %tr = hir.int_type
      // expected-remark @below {{required by call argument 0 at phase 0}}
      %v = uir.call @EarlierHalfErrTarget(%d) : (%ta) -> (%tr) (!hir.any) -> !hir.any [-1] -> [0]
      uir.yield %v : %t
    } else {
      uir.yield %d : %t
    }
    uir.yield %s : %t
  }
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Causality on a var write under later-phase CF: the write sits in a const
// block under phase-1 CF and anchors at phase 0 (D-rule), but consumes the
// phase-1 dyn value. A phase-0 op can never consume a phase-1 value, no matter
// what CF surrounds it.

// expected-error @below {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @DRuleWriteConsumesLaterValue(%x: 0, %d: 1, %cond: 1) -> () {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.bool_type
  uir.signature (%t0, %t1, %t2) -> ()
} {
  %t = hir.int_type
  %v = hir.var : %t
  hir.write_var %v, %x : %t
  uir.expr pin 1 {
    uir.if %cond {
      uir.expr pin -1 {
        // expected-remark @below {{required by operand at phase 0}}
        hir.write_var %v, %d : %t
        uir.yield
      }
      uir.yield
    }
    uir.yield
  }
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// A demand on a floating expr's result also demands its result-type operand
// one phase earlier, just like a pinned expr or any other typed value. Here
// the expr result is returned at phase 0, so its result type `%ty` is demanded
// at phase -1; `%ty` is `type_of` a phase-1 arg, which pushes the arg to phase
// 0 and conflicts with its declared phase 1. Without the demand-path type
// bound this would silently deliver the result at phase 0 above a phase-1
// type.

// expected-error @below {{value at phase 1 cannot satisfy requirement for phase 0}}
uir.func @LiveConduitTypeBound(%a: 1, %x: 0) -> (result: 0) {
  %t0 = hir.int_type
  uir.signature (%t0, %t0) -> (%t0)
} {
  %t = hir.int_type
  %ty = hir.type_of %a
  // expected-remark @below {{required by expr result type at phase -1}}
  %r = uir.expr : %ty {
    %sum = hir.add %x, %x : %t
    uir.yield %sum : %t
  }
  // expected-remark @below {{required by return value at phase 0}}
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// A phase-shifted exit inside a dead frame is a user error exactly as in a
// live one: the continue sits inside a `pin 1` sub-block, one phase above its
// target loop, so it cannot reach the loop's iteration argument. The derived
// frame flows through the same exit-phase check that placed frames obey.

uir.func @ShiftedExitInDeadFrame(%x: 0) -> () {
  %t0 = hir.int_type
  uir.signature (%t0) -> ()
} {
  %t = hir.int_type
  %dead = uir.expr : %t {
    %s = uir.loop (%i = %x : %t) : %t {
      %p = uir.expr pin 1 : %t {
        // expected-error @below {{continue from a phase-shifted block is not allowed}}
        uir.continue %i : %t
      }
      uir.break %p : %t
    }
    uir.yield %s : %t
  }
  uir.return
}

// -----

// Break variant of the same check: a `pin 1` sub-block between a dead
// frame's loop and its break shifts the exit one phase above its target.

uir.func @ShiftedBreakInDeadFrame(%x: 0) -> () {
  %t0 = hir.int_type
  uir.signature (%t0) -> ()
} {
  %t = hir.int_type
  %dead = uir.expr : %t {
    %s = uir.loop (%i = %x : %t) : %t {
      %p = uir.expr pin 1 : %t {
        // expected-error @below {{break from a phase-shifted block is not allowed}}
        uir.break %i : %t
      }
      uir.continue %p : %t
    }
    uir.yield %s : %t
  }
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// Pick result floor: two alternatives yield distinct const values, so the
// pick at phase 0 selects between them and the result can't be at -1 —
// exactly the if-result floor rule.

uir.func @PickDistinctYieldFloorError(%sel: -1, %a: -1, %b: -1) -> (result: -1) {
  %t0 = hir.int_type
  %t1 = hir.int_type
  %t2 = hir.int_type
  %t3 = hir.int_type
  uir.signature (%t0, %t1, %t2) -> (%t3)
} {
  %t = hir.int_type
  %ch = hir.choice 0(%sel)
  // expected-error @below {{result at phase -1 is selected by control flow at phase 0; result phase must be >= 0}}
  %r = hir.pick %ch : !hir.any [0 (): {
    // expected-note @below {{value provided here}}
    hir.yield %a : !hir.any
  } 1 (): {
    // expected-note @below {{value provided here}}
    hir.yield %b : !hir.any
  }]
  // expected-remark @below {{required by return value at phase -1}}
  uir.return %r : %t
}

// -----

//===----------------------------------------------------------------------===//
// Signature from inside a phase-shifted block.

uir.func @SignatureFromConstBlock(%x: 0) -> () {
  %0 = hir.int_type
  uir.expr pin -1 {
    // expected-error @+1 {{signature from a phase-shifted block is not allowed}}
    uir.signature (%0) -> ()
  }
  uir.unreachable
} {
  uir.return
}
