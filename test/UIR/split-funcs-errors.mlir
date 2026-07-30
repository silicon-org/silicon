// RUN: silicon-opt --split-input-file --verify-diagnostics --split-funcs %s

// A body phase later than the declared span has no executor: user error.
uir.func @past_span(%a: 0) -> (result: 0) {
  %s = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.mul %a, %a : %int
    uir.phase_return %r : %int
  }
  // expected-error @below {{phase 1 runs past the function's last declared phase 0, which has no executor}}
  uir.phase 1 {
    %int = hir.int_type
    %x = hir.add %a, %a : %int
    uir.phase_return
  }
  uir.unreachable
}

// -----

// The call-implied variant: a caller whose decomposed `uir.phase_call` to a
// late callee phase lands in a phase op past the caller's own span. The callee
// @Late legally spans [0, 1]; the caller @EarlyCaller declares span [0].
uir.func @Late(%a: 0) -> (r: 1) {
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase 1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s0) -> (%s1)
} {
  %p = uir.phase 0 {
    %int = hir.int_type
    %sq = hir.mul %a, %a : %int
    uir.phase_return yield %sq
  }
  uir.phase 1 {
    %int = hir.int_type
    uir.phase_return %p : %int
  }
  uir.unreachable
}

uir.func @EarlyCaller(%x: 0) -> () {
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s0) -> ()
} {
  %ctx = uir.phase 0 {
    %t = hir.int_type
    %c = uir.phase_call @Late[0] (%x) : (%t) -> () context
    uir.phase_return yield %c
  }
  // expected-error @below {{phase 1 runs past the function's last declared phase 0, which has no executor}}
  uir.phase 1 {
    %t = hir.int_type
    %r = uir.phase_call @Late[1] () context %ctx : () -> (%t)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// A floating op left at the region top level is not fully threaded form.
uir.func @float_at_top(%a: 0) -> (result: 0) {
  %s = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  // expected-error @below {{is not a 'uir.phase' op or the region terminator}}
  %int = hir.int_type
  uir.phase 0 {
    %r = hir.mul %a, %a : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}

// -----

// A phase whose exits are nested inside structured control flow (terminated by
// uir.unreachable rather than a single top-level uir.phase_return) is not
// straight-line phase content the splitter supports.
uir.func @cf_in_phase(%a: 0, %c: 0) -> (result: 0) {
  %s = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s, %s) -> (%s)
} {
  // expected-error @below {{phase body is not terminated by 'uir.phase_return'}}
  uir.phase 0 {
    %int = hir.int_type
    uir.if %c {
      uir.phase_return %a : %int
    } else {
      uir.phase_return %a : %int
    }
    uir.unreachable
  }
  uir.unreachable
}
