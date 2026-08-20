// RUN: silicon-opt --split-input-file --verify-diagnostics --split-funcs %s

// A body phase later than the declared span has no executor: user error.
uir.split_func @past_span(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @past_span.impl []

uir.func @past_span.impl(%a: 0) -> (result: 0) {
  %s = uir.phase -1 {
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
uir.split_func @Late(%a: 0) -> (r: 1) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @Late.impl []

uir.func @Late.impl(%a: 0) -> (r: 1) {
  %s0 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s1 = uir.phase 0 {
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

uir.split_func @EarlyCaller(%x: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @EarlyCaller.impl []

uir.func @EarlyCaller.impl(%x: 0) -> () {
  %s0 = uir.phase -1 {
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
uir.split_func @float_at_top(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @float_at_top.impl []

uir.func @float_at_top.impl(%a: 0) -> (result: 0) {
  %s = uir.phase -1 {
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
uir.split_func @cf_in_phase(%a: 0, %c: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @cf_in_phase.impl []

uir.func @cf_in_phase.impl(%a: 0, %c: 0) -> (result: 0) {
  %s = uir.phase -1 {
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

// -----

// A `uir.func` without a `uir.split_func` holding its declaration was never
// sorted: there is nothing for callers to bind to, and no flat signature.
// expected-error @below {{is not the implementation of a 'uir.split_func'}}
uir.func @unsnapshotted(%a: 0) -> (result: 0) {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %a : %int
  }
  uir.unreachable
}

// -----

// A phase call must name the callee's late-binding interface, whose phase list
// resolves the entry; the implementation function has no phase list.
uir.split_func @named_impl(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @named_impl.impl []

uir.func @named_impl.impl(%a: 0) -> (result: 0) {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %a : %int
  }
  uir.unreachable
}

uir.split_func @impl_caller(%x: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @impl_caller.impl []

uir.func @impl_caller.impl(%x: 0) -> () {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> ()
} {
  uir.phase 0 {
    %t = hir.int_type
    // expected-error @below {{callee @named_impl.impl is not a 'uir.split_func'}}
    %r = uir.phase_call @named_impl.impl[0] (%x) : (%t) -> (%t)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// A callee in the declared state has no implementation to call into: its phase
// list is empty, so no entry resolves.
uir.split_func @declared_only(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} []

uir.split_func @declared_caller(%x: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @declared_caller.impl []

uir.func @declared_caller.impl(%x: 0) -> () {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> ()
} {
  uir.phase 0 {
    %t = hir.int_type
    // expected-error @below {{callee @declared_only has no function for phase 0; its implementation is not available}}
    %r = uir.phase_call @declared_only[0] (%x) : (%t) -> (%t)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// An implementation belongs to exactly one declaration: two claims would split
// the same function twice, into two sets of per-phase functions.
uir.split_func @first_claim(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @shared_impl []

// expected-error @below {{shares its implementation @shared_impl with another 'uir.split_func'}}
uir.split_func @second_claim(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @shared_impl []

uir.func @shared_impl(%a: 0) -> (result: 0) {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %a : %int
  }
  uir.unreachable
}

// -----

// The threaded form holds one `uir.phase` op per phase: a declared span phase
// without one has no function for callers to thread their entry through.
uir.split_func @span_gap(%a: 0) -> (result: 1) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @span_gap.impl []

// expected-error @below {{has no 'uir.phase' op for phase 1 in either region, but holds phases -1 through 1}}
uir.func @span_gap.impl(%a: 0) -> (result: 1) {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> (%s)
} {
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %a : %int
  }
  uir.unreachable
}

// -----

// The same rule inside the phase range: a gap in the middle breaks the context
// chain that hands each phase's crossings to the next. The phase must be
// missing from both regions — the signature's own phase ops fill the chain just
// as well as the body's, since the two are merged phase by phase.
uir.split_func @interior_gap(%a: -3, %x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @interior_gap.impl []

// expected-error @below {{has no 'uir.phase' op for phase -2 in either region, but holds phases -4 through 0}}
uir.func @interior_gap.impl(%a: -3, %x: 0) -> (result: 0) {
  %sm4 = uir.phase -4 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%sm4, %sm1) -> (%sm1)
} {
  uir.phase -3 {
    %int = hir.int_type
    %sq = hir.mul %a, %a : %int
    uir.phase_return
  }
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %x : %int
  }
  uir.unreachable
}

// -----

// A body with no phase op at all has nothing to split: the declared span phase
// 0 is missing even though the signature reaches phase -1.
uir.split_func @no_phases(%a: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @no_phases.impl []

// expected-error @below {{has no 'uir.phase' op for phase 0 in either region, but holds phases -1 through 0}}
uir.func @no_phases.impl(%a: 0) -> () {
  %s = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.signature (%s) -> ()
} {
  uir.unreachable
}

// -----

// A `uir.signature` operand that is not a result of the phase P-1 op cannot
// describe a phase-P interface: the type must reach the per-phase function
// through its incoming context, and a value produced in the phase itself never
// does. Here the phase 0 argument's type is computed by the phase 0 signature
// op instead of the phase -1 one.
uir.split_func @sig_type_same_phase(%a: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @sig_type_same_phase.impl []

uir.func @sig_type_same_phase.impl(%a: 0) -> () {
  uir.phase -1 {
    uir.phase_return
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  // expected-error @below {{types a phase 0 argument with a value that is not a result of the phase -1 signature op}}
  uir.signature (%s0) -> ()
} {
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}

// -----

// The same rule for a declared result.
uir.split_func @sig_result_same_phase(%a: 0) -> (r: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @sig_result_same_phase.impl []

uir.func @sig_result_same_phase.impl(%a: 0) -> (r: 0) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  %s0 = uir.phase 0 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  // expected-error @below {{types a phase 0 result with a value that is not a result of the phase -1 signature op}}
  uir.signature (%sm1) -> (%s0)
} {
  uir.phase 0 {
    %int = hir.int_type
    uir.phase_return %a : %int
  }
  uir.unreachable
}
