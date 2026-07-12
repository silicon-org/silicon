// RUN: silicon-opt --sort-phases --split-input-file --verify-diagnostics %s

// Phase analysis does not model hir.replicate regions; their contents lack
// phase entries and sorting must abort loudly instead of mis-sorting them.
uir.func @Unmodeled(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  %t = hir.int_type
  %list = hir.opaque_list_create
  hir.replicate (%r) in %list {
    // expected-error @below {{is not covered by phase analysis; cannot sort ops the analysis does not model}}
    %a = hir.add %x, %x : %t
    hir.yield
  }
  uir.return
}

// -----

// An expr terminated by a loop exit cannot be dissolved.
uir.func @BreakCarrier(%n: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.loop : %t {
    // expected-error @below {{cannot be dissolved: body is terminated by uir.break instead of a yield}}
    uir.expr {
      uir.break %x : %t
    }
    uir.unreachable
  }
  uir.return %x : %t
}

// -----

// An early return nested in structured control flow needs function-level
// exit chains.
uir.func @EarlyExit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.if %c {
    // expected-error @below {{is an early function exit; function-level exit chains are not implemented in phase sorting yet}}
    uir.return %x : %t
  }
  uir.return %x : %t
}

// -----

// A return-terminated expr carrier is an early exit too; dissolution leaves
// it intact and the exit rejection reports the return inside it.
uir.func @ReturnCarrier(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.expr {
    // expected-error @below {{is an early function exit; function-level exit chains are not implemented in phase sorting yet}}
    uir.return %x : %t
  }
  uir.unreachable
}

// -----

// A uir.call whose callee resolves to something other than a uir.func or
// uir.split_func cannot be decomposed: its phase entries are undefined.
hir.func @NotUnified() -> () {
  hir.signature () -> ()
} {
  hir.return -> ()
}

uir.func @CallsNonFunc(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // expected-error @below {{callee @NotUnified is neither a 'uir.func' nor a 'uir.split_func'}}
  uir.call @NotUnified() : () -> () () -> () [] -> []
  uir.return
}

// -----

// The sorted-phase ops are sorting's output vocabulary; feeding them back
// into sorting means the pipeline is out of order.
uir.func @PhaseInInput(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // expected-error @below {{is phase-sorting output vocabulary and cannot appear in phase-sorting input}}
  uir.phase 0 {
    uir.phase_return
  }
  uir.return
}

// -----

uir.func @PhaseCallCallee(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  uir.return
}

uir.func @PhaseCallInInput(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  %t = hir.int_type
  // expected-error @below {{is phase-sorting output vocabulary and cannot appear in phase-sorting input}}
  uir.phase_call @PhaseCallCallee[0] (%x) : (%t) -> ()
  uir.return
}

// -----

// A never-produced control-flow result (a loop without breaks) diverges
// from its op's phase: the value never materializes and every consumer is
// unreachable. Unreachable-consumer handling is a later sorting case.
uir.func @NeverResult(%a: -1, %x: 0) -> (c: -1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase never instead of its own phase 0; result forwarding (identity-provided or never-produced control-flow results) is not implemented in phase sorting yet}}
  %r = uir.loop : %t {
    uir.continue
  }
  %s = hir.add %a, %r : %t
  uir.return %s : %t
}

// -----

// A control-flow result provided by the same SSA value on all paths (the
// identity exemption) is delivered at that value's phase, which can precede
// the CF op. Forwarding the provider instead of crossing is a later sorting
// case.
uir.func @IdentityBackward(%a: -1, %c: 0) -> (r: -1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase -1 instead of its own phase 0; result forwarding (identity-provided or never-produced control-flow results) is not implemented in phase sorting yet}}
  %r = uir.if %c : %t {
    uir.yield %a : %t
  } else {
    uir.yield %a : %t
  }
  %s = hir.mul %r, %r : %t
  uir.return %s : %t
}

// -----

// The divergence also goes forward: an identity-provided result of a
// transparent-yield region delivers a later-phase value out of an
// earlier-phase op, so the op's interior references values that do not
// exist at its phase.
uir.func @IdentityForward(%c: -1, %x: 0) -> (r: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.expr pin -1 : %t {
    // expected-error @below {{delivers result #0 at phase 0 instead of its own phase -1; result forwarding (identity-provided or never-produced control-flow results) is not implemented in phase sorting yet}}
    %i = uir.if %c : %t {
      uir.yield %x : %t
    } else {
      uir.yield %x : %t
    }
    uir.yield %i : %t
  }
  %s = hir.add %r, %x : %t
  uir.return %s : %t
}

// -----

// An if with a later-phase result whose arms do not all yield: the else arm
// exits early instead. Branch extraction classifies the result by asking
// whether the yielding arms select distinct values, which must tolerate the
// non-yield arm rather than assume every arm yields. With only one provider
// the result is not choice-ified, so the if is left for the nested-content
// rejection.
uir.func @ExitArmLaterResult(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = uir.if %c : %t {
    // expected-error @below {{at phase 1 is nested inside phase 0 content; hoisting and extraction across phase boundaries are not implemented in phase sorting yet}}
    %a = hir.add %y, %y : %t
    uir.yield %a : %t
  } else {
    uir.return %y : %t
  }
  uir.return %v : %t
}

// -----

// Floating (dead) content nested inside earlier-phase control flow can
// consume later-phase values without any demand anchoring it; it cannot
// sink out of the region it is glued into, so the backward use is rejected.
uir.func @FloatBackwardInBranch(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %m = uir.expr pin 1 : %t {
    %v = hir.var : %t
    hir.write_var %v, %x : %t
    // expected-note @below {{value defined here}}
    %rd = hir.read_var %v : %t
    uir.yield %rd : %t
  }
  uir.if %c {
    %y = uir.expr : %t {
      // expected-error @below {{uses a value from phase 1 inside phase 0 content; backward crossings are not implemented in phase sorting yet}}
      %dead = hir.add %m, %m : %t
      uir.yield %dead : %t
    }
    uir.yield
  }
  uir.return
}

// -----

// Content at a different phase nested under a loop needs loop-record
// extraction, which is a later sorting case. Branch extraction handles the
// same shape under a uir.if, but not under a loop.
uir.func @LaterContentInLoop(%n: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %v = hir.var : %t
  uir.loop {
    uir.expr pin 1 {
      // expected-error @below {{at phase 1 is nested inside phase 0 content; hoisting and extraction across phase boundaries are not implemented in phase sorting yet}}
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.continue
  }
  uir.return
}
