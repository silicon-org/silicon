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

// A single-provider loop result whose provider is defined inside the loop at a
// float phase (a constant): it does not forward (not external, iteration-
// invariant content) and does not choice-ify (a float phase is not a later real
// phase), so it survives to the delivery rejection awaiting float sinking.
uir.func @LoopBreakFloatConst(%c: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase float instead of its own phase 0; delivering this cross-phase result is not implemented in phase sorting yet (a multi-provider selection from loop or break-value extraction whose receiving pick crosses more than one phase, or a single-provider float value that neither forwards nor choice-ifies)}}
  %l = uir.loop : %t {
    %five = hir.constant_int 5 : %t
    uir.break %five : %t
  }
  uir.return %l : %t
}

// -----

// A genuine multi-provider wide break value: a phase-0 loop with two break
// sites delivering distinct phase-2 values. Break-value extraction choice-ifies
// them into a receiving `hir.pick` after the loop, but that pick still delivers
// its result two phases wide. Branch extraction's re-extraction fixpoint runs
// before break-value extraction, so this pick is never fed back through it and
// the wide delivery is rejected instead of mis-sorted.
uir.func @WideMultiBreak(%c: 0, %x: 2, %y: 2) -> (r: 2) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase 2 instead of its own phase 1; delivering this cross-phase result is not implemented in phase sorting yet (a multi-provider selection from loop or break-value extraction whose receiving pick crosses more than one phase, or a single-provider float value that neither forwards nor choice-ifies)}}
  %l = uir.loop : %t {
    uir.if %c {
      uir.break %x : %t
    } else {
      uir.break %y : %t
    }
    uir.continue
  }
  uir.return %l : %t
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

// Hoisting, forwarding, and extraction all run on the body only, so an
// earlier-phase op nested inside a later-phase frame in the signature region is
// not hoisted out and surfaces as a nested-content rejection instead. Here a
// phase -1 `hir.uint_type` sits inside a phase-0 `uir.if` frame that computes
// the argument type conditionally.
uir.func @SignatureFrameHoist(%N: -1, %c: 0) -> (result: 0) {
  %int = hir.int_type
  %ty = uir.if %c : %int {
    // expected-error @below {{at phase -1 is nested inside later phase 0 content; hoisting it out is not implemented in phase sorting yet}}
    %w = hir.uint_type %N
    uir.yield %int : %int
  } else {
    uir.yield %int : %int
  }
  uir.signature (%int, %int) -> (%int)
} {
  %t = hir.int_type
  %r = hir.add %c, %c : %t
  uir.return %r : %t
}
