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

// A result provided by a single yield while the other arm exits is delivered
// at the provider's phase, which can precede the CF op. Forwarding the
// provider instead of crossing is a later sorting case.
uir.func @SingleProviderBackward(%a: -1, %c: 0) -> (r: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %l = uir.loop : %t {
    // expected-error @below {{delivers result #0 at phase -1 instead of its own phase 0; result forwarding (single-provider control-flow results, and receivers of crossings more than one phase wide) is not implemented in phase sorting yet}}
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

// The other arm becomes a non-provider by yielding a value that never
// materializes (a breakless loop's result), not by exiting: the result is
// still single-provider on the surviving arm. The analysis and the extraction
// now agree on the provider count (both exclude the never arm), so this lands
// as the ordinary single-provider forwarding rejection rather than a spurious
// mismatch.
uir.func @NeverArmForwardResult(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase 1 instead of its own phase 0; result forwarding (single-provider control-flow results, and receivers of crossings more than one phase wide) is not implemented in phase sorting yet}}
  %v = uir.if %c : %t {
    uir.yield %y : %t
  } else {
    %loop = uir.loop : %t { uir.continue }
    uir.yield %loop : %t
  }
  uir.return %v : %t
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

