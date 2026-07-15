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
  // expected-error @below {{delivers result #0 at phase never instead of its own phase 0; result forwarding (single-provider or never-produced control-flow results, and receivers of crossings more than one phase wide) is not implemented in phase sorting yet}}
  %r = uir.loop : %t {
    uir.continue
  }
  %s = hir.add %a, %r : %t
  uir.return %s : %t
}

// -----

// A result provided by a single yield while the other arm exits is delivered
// at the provider's phase, which can precede the CF op. Forwarding the
// provider instead of crossing is a later sorting case.
uir.func @SingleProviderBackward(%a: -1, %c: 0) -> (r: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %l = uir.loop : %t {
    // expected-error @below {{delivers result #0 at phase -1 instead of its own phase 0; result forwarding (single-provider or never-produced control-flow results, and receivers of crossings more than one phase wide) is not implemented in phase sorting yet}}
    %r = uir.if %c : %t {
      uir.break %a : %t
    } else {
      uir.yield %a : %t
    }
    %s = hir.mul %r, %r : %t
    uir.break %s : %t
  }
  uir.return %l : %t
}

// -----

// An if with a later-phase result whose arms do not all yield: the else arm
// exits early instead. Branch extraction classifies the result by counting
// its providing yields, which must tolerate the non-yield arm rather than
// assume every arm yields. With only one provider the result is not
// choice-ified, so the if is left for the nested-content rejection.
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

// A break value shared with other later-phase effects that stay behind cannot
// be sliced out: the entangled content (here a var written and read for the
// break value) would be duplicated into the receiving pick. Rejected.
uir.func @SharedBreakValueSlice(%n: 0, %x: 0) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.loop : %t {
    %done = hir.geq %x, %n : %t
    uir.if %done {
      %v = uir.expr pin 1 : %t {
        // expected-error @below {{is shared between a break value and other later-phase content; not implemented in phase sorting yet}}
        %vv = hir.var : %t
        hir.write_var %vv, %x : %t
        %rr = hir.read_var %vv : %t
        uir.yield %rr : %t
      }
      uir.break %v : %t
    }
    uir.continue
  }
  uir.return %r : %t
}

// -----

// Two break sites carrying the same value slice: from the first site the slice
// is used by the second break, i.e. by content outside the first site, so it is
// rejected as a shared break-value slice.
uir.func @SharedAcrossBreaks(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      // expected-error @below {{is shared between a break value and other later-phase content; not implemented in phase sorting yet}}
      %s = hir.add %x, %k : %t
      %c1 = hir.geq %k, %n : %t
      uir.if %c1 {
        uir.break %s : %t
      }
      %c2 = hir.geq %k, %one : %t
      uir.if %c2 {
        uir.break %s : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  uir.return %y : %t
}

// -----

// A break that does not carry the loop's later-phase result while another break
// does: break arity is not a verifier invariant, so the missing value is caught
// here.
uir.func @BreakMissingResult(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %c1 = hir.geq %k, %n : %t
      uir.if %c1 {
        %s = hir.add %x, %k : %t
        uir.break %s : %t
      }
      %c2 = hir.geq %k, %one : %t
      uir.if %c2 {
        // expected-error @below {{does not provide the loop's result 0}}
        uir.break
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  uir.return %y : %t
}

// -----

// A multi-result loop whose later result is choice-ified but one break omits a
// trailing kept (phase-p) result: every result the break must supply is
// positionally indexed, so the missing kept value is caught, not indexed out of
// range.
uir.func @BreakMissingKeptResult(%n: -1, %x: 0) -> (a: 0, b: -1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0, %0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %y:2 = uir.expr pin -1 : %t, %t {
    %r:2 = uir.loop (%k = %zero : %t) : %t, %t {
      %c1 = hir.geq %k, %n : %t
      uir.if %c1 {
        %s = hir.add %x, %k : %t
        uir.break %s, %k : %t, %t
      }
      %c2 = hir.geq %k, %one : %t
      uir.if %c2 {
        %m = hir.add %x, %one : %t
        // expected-error @below {{does not provide the loop's result 1}}
        uir.break %m : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r#0, %r#1 : %t, %t
  }
  uir.return %y#0, %y#1 : %t, %t
}

// -----

// A return exiting the function under a const loop with later-phase content is
// a function-level exit chain, not loop-record extraction.
uir.func @ReturnUnderLoop(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = hir.var : %t
  uir.loop {
    uir.expr pin 1 {
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.if %c {
      // expected-error @below {{exits the function under a loop with later-phase content; function-level exit chains are not implemented in phase sorting yet}}
      uir.return %x : %t
    }
    uir.continue
  }
  uir.unreachable
}

// -----

// An exit-carrying divergence whose branch has a later-phase result would need
// break-value reconstruction from the divergence; not implemented.
uir.func @DivergenceLaterResult(%n: -1, %x: 0) -> (result: 0) {
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
      // expected-error @below {{carries a loop exit and has later-phase results; exit-carrying branches with demanded later-phase results are not implemented in phase sorting yet}}
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
  %rr = hir.read_var %acc : %t
  uir.return %rr : %t
}

// -----

// An exit-carrying arm that itself diverges (later-phase content plus a nested
// exit) needs nested exit-path choices; not implemented.
uir.func @NestedDivergence(%n: -1, %x: 0) -> (result: 0) {
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
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %acc, %i : %t
          uir.yield
        }
        %d2 = hir.geq %i, %n : %t
        // expected-error @below {{diverges inside an exit-carrying arm; nested divergence is not implemented in phase sorting yet}}
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
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

// -----

// An exit-carrying branch whose rejoining arm carries later-phase content
// would need an arm-local sub-choice; not implemented.
uir.func @RejoiningArmContent(%n: -1, %x: 0) -> (result: 0) {
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
      // expected-error @below {{has later-phase content on a rejoining arm of an exit-carrying branch; not implemented in phase sorting yet}}
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
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

// -----

// An exit-carrying branch whose rejoining arm hides a nested exit needs an
// arm-local sub-choice; not implemented.
uir.func @RejoiningNestedExit(%n: -1, %x: 0) -> (result: 0) {
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
      // expected-error @below {{has a rejoining arm containing a nested exit; nested divergence is not implemented in phase sorting yet}}
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
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

// -----

// A loop body with later-phase content whose activation ends in a return is a
// function-level exit chain, not loop-record extraction.
uir.func @LoopBodyReturn(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = hir.var : %t
  uir.loop {
    uir.expr pin 1 {
      hir.write_var %v, %x : %t
      uir.yield
    }
    // expected-error @below {{terminates a loop body with later-phase content; function-level exit chains are not implemented in phase sorting yet}}
    uir.return %x : %t
  }
  uir.unreachable
}

// -----

// A dead never-breaking loop inside a dead frame sorts like its live twin:
// the frame defaults to its block phase and the loop delivers a `never`
// result. Consuming that result needs unreachable-consumer handling, which
// the sorter rejects — same as a live loop that never breaks.
uir.func @FloatingFrameRematerialized(%c: 0, %a: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.if %c : %t {
    %fl = uir.expr : %t {
      // expected-error @below {{delivers result #0 at phase never instead of its own phase 0; result forwarding (single-provider or never-produced control-flow results, and receivers of crossings more than one phase wide) is not implemented in phase sorting yet}}
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
