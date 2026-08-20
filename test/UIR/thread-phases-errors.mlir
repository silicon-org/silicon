// RUN: silicon-opt --split-input-file --verify-diagnostics --thread-phases %s

// A region top-level op that is neither a phase op, the terminator, nor a
// floating pure op (here an impure hir.var) is not sorted-form input.
uir.func @impure_at_top(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{is not a 'uir.phase' op, the region terminator, or a floating op}}
  %v = hir.var : %t
  uir.phase 0 {
    uir.phase_return %a : %t
  }
  uir.unreachable
}

// -----

// A region terminator that is neither uir.unreachable nor uir.signature.
uir.func @bad_terminator(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %r = hir.add %a, %a : %t
  // expected-error @below {{cannot terminate a threaded region}}
  uir.return %r : %t
}

// -----

// Two phase ops for the same phase.
uir.func @duplicate_phase(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.phase 0 {
    uir.phase_return %a : %t
  }
  // expected-error @below {{duplicate 'uir.phase' op for phase 0}}
  uir.phase 0 {
    %r = hir.add %a, %a : %t
    uir.phase_return
  }
  uir.unreachable
}

// -----

// A floating op consuming a phase-op result.
uir.func @float_consumes_result(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{result is consumed by a floating op}}
  %p = uir.phase -1 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return yield %sq
  }
  %bad = hir.add %p, %p : %t
  uir.phase 0 {
    %r = hir.add %x, %bad : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

// -----

// An argument used before its declared phase.
// expected-error @below {{argument declared at phase 0 is used at earlier phase -1}}
uir.func @arg_before_phase(%a: 0) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.phase -1 {
    %r = hir.add %a, %a : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

// -----

// A phase-op result consumed at an earlier phase (phase ops out of order, so
// the higher phase dominates the lower one).
uir.func @backward_crossing(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{result at phase 2 is consumed at earlier or equal phase 0}}
  %hi = uir.phase 2 {
    %m = hir.mul %a, %a : %t
    uir.phase_return yield %m
  }
  uir.phase 0 {
    %r = hir.add %hi, %a : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

// -----

// A function result declared at a phase with no phase op to carry it.
// expected-error @below {{declares a result at phase 0 but has no 'uir.phase' op there to carry it}}
uir.func @missing_result_phase(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.phase -1 {
    uir.phase_return %a : %t
  }
  uir.unreachable
}

// -----

// A phase return that carries the wrong number of function results.
uir.func @result_count_mismatch(%a: 0) -> (r0: 0, r1: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0, %0)
} {
  %t = hir.int_type
  uir.phase 0 {
    // expected-error @below {{carries 1 function results but phase 0 declares 2}}
    uir.phase_return %a : %t
  }
  uir.unreachable
}

// -----

// A signature type value produced by the same phase as the argument it
// ascribes. A phase-P argument's type is phase P-1 content: the per-phase
// function's signature must be constructible without evaluating anything, so a
// value the phase itself computes can never describe that phase's interface.
uir.func @sig_type_same_phase(%x: 0) -> () {
  // expected-error @below {{result at phase 0 types a phase 0 value, whose type must be phase -1 content}}
  %p = uir.phase 0 {
    %ty = hir.int_type
    uir.phase_return yield %ty
  }
  uir.signature (%p) -> ()
} {
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}

// -----

// The argument variant of the same rule: an argument declared at phase P
// cannot ascribe a phase-P value either.
// expected-error @below {{argument declared at phase 0 types a phase 0 value, whose type must be phase -1 content}}
uir.func @sig_type_same_phase_arg(%T: 0, %x: 0) -> () {
  %t = hir.int_type
  uir.signature (%t, %T) -> ()
} {
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}
