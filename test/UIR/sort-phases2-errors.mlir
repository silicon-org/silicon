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

// Call decomposition into uir.phase_call entries is a later sorting case.
uir.func @Callee(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @Caller(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{decomposition into 'uir.phase_call' entries is not implemented in phase sorting yet}}
  %r = uir.call @Callee(%x) : (%t) -> (%t) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %t
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

// A value computed at one phase and consumed at a later one must cross as a
// phase-op result, which is a later sorting case.
uir.func @Crossing(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = uir.expr pin -1 : %t {
    // expected-note @below {{value defined here}}
    %m = hir.mul %a, %a : %t
    uir.yield %m : %t
  }
  // expected-error @below {{uses a value defined in phase -1; cross-phase value crossings are not implemented in phase sorting yet}}
  %r = hir.add %sq, %x : %t
  uir.return %r : %t
}

// -----

// The signature terminator consuming a value computed inside a phase is a
// crossing as well (it will consume a phase-op result).
uir.func @SigCrossing(%x: 0) -> (result: 1) {
  %0 = hir.int_type
  %v = hir.var : %0
  // expected-note @below {{value defined here}}
  %r = hir.read_var %v : %0
  // expected-error @below {{uses a value defined in phase 0; cross-phase value crossings are not implemented in phase sorting yet}}
  uir.signature (%0) -> (%r)
} {
  %t = hir.int_type
  uir.return %x : %t
}

// -----

// A floating op whose operands moved into different phase regions cannot be
// placed without a crossing.
uir.func @DeadFloatAcrossPhases(%a: -1, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %r1 = uir.expr pin -1 : %t {
    %v1 = hir.var : %t
    hir.write_var %v1, %a : %t
    // expected-note @below {{value defined here}}
    %rd1 = hir.read_var %v1 : %t
    uir.yield %rd1 : %t
  }
  %v2 = hir.var : %t
  hir.write_var %v2, %x : %t
  %r2 = hir.read_var %v2 : %t
  %dead = uir.expr : %t {
    // expected-error @below {{uses a value defined in phase -1; cross-phase value crossings are not implemented in phase sorting yet}}
    %m = hir.add %r1, %r2 : %t
    uir.yield %m : %t
  }
  uir.return
}

// -----

// Content at a different phase nested under control flow needs hoisting or
// extraction, which are later sorting cases.
uir.func @LaterContentInBranch(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %v = hir.var : %t
  uir.if %c {
    uir.expr pin 1 {
      // expected-error @below {{at phase 1 is nested inside phase 0 content; hoisting and extraction across phase boundaries are not implemented in phase sorting yet}}
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.yield
  }
  uir.return
}
