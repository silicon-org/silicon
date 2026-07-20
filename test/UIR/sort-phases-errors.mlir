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

// A multi-provider result demanded more than one phase past its if: both arms
// yield an external phase-2 value, so branch extraction choice-ifies it into a
// pick at phase 1, but the reconstructed value materializes at phase 2. The
// pick is a receiver of a crossing more than one phase wide and would itself
// need re-extraction at the phase-1-to-2 boundary.
uir.func @WideReceiver(%c: 0, %x: 2, %y: 2) -> (r: 2) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{delivers result #0 at phase 2 instead of its own phase 1; delivering this cross-phase result is not implemented in phase sorting yet (a multi-provider selection whose receiving pick would cross more than one phase, or a single-provider float value that neither forwards nor choice-ifies)}}
  %v = uir.if %c : %t {
    uir.yield %x : %t
  } else {
    uir.yield %y : %t
  }
  uir.return %v : %t
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
  // expected-error @below {{delivers result #0 at phase float instead of its own phase 0; delivering this cross-phase result is not implemented in phase sorting yet (a multi-provider selection whose receiving pick would cross more than one phase, or a single-provider float value that neither forwards nor choice-ifies)}}
  %l = uir.loop : %t {
    %five = hir.constant_int 5 : %t
    uir.break %five : %t
  }
  uir.return %l : %t
}

// -----

// A single-provider if result whose provider is later-phase content defined
// inside the arm (not external): it does not forward. The interior later-phase
// op is caught by the nested-content rejection before the if's own delivery
// check (the post-order walk reaches the interior op first).
uir.func @SingleProviderInternalIf(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = uir.if %c : %t {
    // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
    %w = hir.add %y, %y : %t
    uir.yield %w : %t
  } else {
    %loop = uir.loop : %t { uir.continue }
    uir.yield %loop : %t
  }
  uir.return %v : %t
}

// -----

// An if with a later-phase result whose arms do not all yield: the else arm
// is a dead path terminated by `uir.unreachable`. Branch extraction classifies
// the result by counting its providing yields, which must tolerate the
// non-yielding arm rather than assume every arm yields. With only one provider
// the result is not choice-ified, so the if is left for the nested-content
// rejection.
uir.func @UnreachableArmLaterResult(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = uir.if %c : %t {
    // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
    %a = hir.add %y, %y : %t
    uir.yield %a : %t
  } else {
    uir.unreachable
  }
  uir.return %v : %t
}

// -----

// An if arm terminated by `uir.unreachable` is a dead path with no
// continuation yield for branch extraction to rebuild, so the if is left
// alone and its later-phase content is reported by the nested-content
// rejection (this used to crash extraction on the terminator cast).
uir.func @UnreachableArmIf(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %va = hir.var : %t
  uir.if %c {
    uir.expr pin 1 {
      // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
      hir.write_var %va, %x : %t
      uir.yield
    }
    uir.yield
  } else {
    uir.unreachable
  }
  uir.return
}

// -----

// The same dead-path rule for a pick: an alternative terminated by
// `uir.unreachable` has no continuation yield, so a pick with later-phase
// content in another alternative is left for the nested-content rejection.
uir.func @UnreachableAltPick(%c: 0, %x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %va = hir.var : %t
  %ch = hir.choice 0 ()
  hir.pick %ch [
    0 (): {
      uir.expr pin 1 {
        // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
        hir.write_var %va, %x : %t
        uir.yield
      }
      hir.yield
    }
    1 (): {
      uir.unreachable
    }
  ]
  uir.return
}

// -----

// A single-provider pick result whose provider is later-phase content defined
// inside the alternative (not external): like the if analogue, it does not
// forward, and with one provider it is not a choice-ification obligation
// either, so the interior later-phase op is caught by the nested-content
// rejection.
uir.func @SingleProviderInternalPick(%c: 0, %y: 1) -> (result: 1) {
  %0 = hir.int_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  %t = hir.int_type
  %ch = hir.choice 0 ()
  %v = hir.pick %ch : !hir.any [
    0 (): {
      // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
      %w = hir.add %y, %y : %t
      hir.yield %w : !hir.any
    }
    1 (): {
      uir.unreachable
    }
  ]
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

// A breakless loop in the then arm makes it provably diverge — its yield is
// unreachable, so it provides no result. The if is thus single-provider on the
// else arm, whose provider %w is defined inside the op at a later phase.
// Single-provider interior content is not choice-ified, so the later-phase %w
// hits the nested-content rejection. This is the narrowed residual: an interior
// single provider now arises only when every other arm provably diverges.
uir.func @DivergingArmSingleProvider(%c: 0, %a: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.if %c : %t {
    %loop = uir.loop : %t { uir.continue }
    uir.yield %loop : %t
  } else {
    // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
    %w = hir.mul %a, %a : %t
    uir.yield %w : %t
  }
  uir.return %r : %t
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
