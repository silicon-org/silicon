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
  // expected-error @below {{delivers result #0 at phase float instead of its own phase 0; delivering this cross-phase result is not implemented in phase sorting yet (a single-provider float value awaiting float sinking)}}
  %l = uir.loop : %t {
    %five = hir.constant_int 5 : %t
    uir.break %five : %t
  }
  uir.return %l : %t
}

// -----

// Later-phase content in a provably diverging branch arm (a breakless loop
// before the yield) has no reaching arm to extract into: branch extraction
// leaves the branch untouched, and the content surfaces as a nested-content
// rejection once the extraction fixpoint settles.
uir.func @DivergingArmLaterContent(%c: 0, %x: 1) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  %t = hir.int_type
  %v = hir.var : %t
  uir.if %c {
    uir.loop {
      uir.continue
    }
    uir.expr pin 1 {
      // expected-error @below {{at phase 1 is nested inside earlier phase 0 content; extracting this later-phase content is not implemented in phase sorting yet}}
      hir.write_var %v, %x : %t
      uir.yield
    }
    uir.yield
  }
  uir.return
}

