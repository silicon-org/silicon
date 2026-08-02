// RUN: silicon-opt --split-input-file --verify-diagnostics --split-funcs %s

// A body phase later than the declared span has no executor: user error.
uir.split_func @past_span(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @past_span.impl []

uir.func @past_span.impl(%a: 0) -> (result: 0) {
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
uir.split_func @Late(%a: 0) -> (r: 1) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @Late.impl []

uir.func @Late.impl(%a: 0) -> (r: 1) {
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

uir.split_func @EarlyCaller(%x: 0) -> () {
  %int = hir.int_type
  uir.signature (%int) -> ()
} impl @EarlyCaller.impl []

uir.func @EarlyCaller.impl(%x: 0) -> () {
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
uir.split_func @float_at_top(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @float_at_top.impl []

uir.func @float_at_top.impl(%a: 0) -> (result: 0) {
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
uir.split_func @cf_in_phase(%a: 0, %c: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} impl @cf_in_phase.impl []

uir.func @cf_in_phase.impl(%a: 0, %c: 0) -> (result: 0) {
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

// -----

// A `uir.func` without a `uir.split_func` holding its declaration was never
// sorted: there is nothing for callers to bind to, and no flat signature.
// expected-error @below {{is not the implementation of a 'uir.split_func'}}
uir.func @unsnapshotted(%a: 0) -> (result: 0) {
  %s = uir.phase 0 {
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
  %s = uir.phase 0 {
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
  %s = uir.phase 0 {
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
  %s = uir.phase 0 {
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

// A dependent argument type computed by a call crosses a phase boundary in the
// signature. Delivering it to the later phase means relocating the computation
// into the producing phase's body, which re-evaluates it there — only pure,
// region-free computations survive that move, and a phase call does not.
uir.split_func @Width(%N: -1) -> (w: -1) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} impl @Width.impl []

uir.func @Width.impl(%N: -1) -> (w: -1) {
  %sm1 = uir.phase -1 {
    %int = hir.int_type
    uir.phase_return yield %int
  }
  uir.phase 0 {
    uir.phase_return
  }
  uir.signature (%sm1) -> (%sm1)
} {
  uir.phase -1 {
    %int = hir.int_type
    %r = hir.mul %N, %N : %int
    uir.phase_return %r : %int
  }
  uir.phase 0 {
    uir.phase_return
  }
  uir.unreachable
}

uir.split_func @sig_call(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  %w = uir.call @Width(%N) : (%int) -> (%int) [-1] -> [-1]
  %ty = hir.uint_type %w
  uir.signature (%int, %ty) -> (%ty)
} impl @sig_call.impl []

uir.func @sig_call.impl(%N: -1, %x: 0) -> (result: 0) {
  %s0:3 = uir.phase -1 {
    %int = hir.int_type
    // expected-error @below {{'uir.phase_call' op cannot be relocated into the phase -1 body}}
    %w, %c = uir.phase_call @Width[-1] (%N) : (%int) -> (%int) context
    %ty = hir.uint_type %w
    uir.phase_return yield %c, %ty, %int
  }
  %s1 = uir.phase 0 {
    uir.phase_call @Width[0] () context %s0#0 : () -> ()
    uir.phase_return yield %s0#1
  }
  uir.signature (%s0#2, %s1) -> (%s1)
} {
  uir.phase -1 {
    uir.phase_return
  }
  uir.phase 0 {
    %int = hir.int_type
    %r = hir.add %x, %x : %int
    uir.phase_return %r : %int
  }
  uir.unreachable
}
