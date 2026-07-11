// RUN: silicon-opt --split-phases2="sort-only=true" --split-input-file --verify-diagnostics %s

// A call whose callee has not been split cannot be decomposed for sorting.
uir.func @unsplit(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @NoWitness(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{callee @unsplit has no uir.split_func witness}}
  %r = uir.call @unsplit(%x) : (%t) -> (%t) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %t
}

// -----

// A break-value slice entangled with later-phase effects that stay behind.
uir.func @SharedBreakValueSlice(%n: 0, %x: 0) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.loop : %t {
    %done = hir.geq %x, %n : %t
    uir.if %done {
      %v = uir.expr pin 1 : %t {
        // expected-error @below {{is shared between a break value and other later-phase content; not implemented yet}}
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

// Later-phase content on a rejoining arm of an exit-carrying branch needs a
// sub-choice covering the arm-local divergence, not implemented yet.
uir.func @RejoinArmContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      // expected-error @below {{has later-phase content on a rejoining arm of an exit-carrying branch; not implemented yet}}
      uir.if %done {
        uir.break
      } else {
        uir.expr pin 1 {
          hir.write_var %v, %i : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

// -----

// A nested exit inside an exit-carrying arm is a nested divergence, not
// implemented yet.
uir.func @NestedDivergenceInExitArm(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %v, %i : %t
        uir.yield
      }
      %done = hir.geq %i, %n : %t
      %big = hir.geq %i, %one : %t
      uir.if %done {
        // expected-error @below {{diverges inside an exit-carrying arm; nested divergence is not implemented yet}}
        uir.if %big {
          uir.break
        }
        uir.break
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

// -----

// A nested exit on a rejoining arm is a nested divergence too.
uir.func @RejoinArmNestedExit(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %v, %i : %t
        uir.yield
      }
      %done = hir.geq %i, %n : %t
      %big = hir.geq %i, %one : %t
      // expected-error @below {{has a rejoining arm containing a nested exit; nested divergence is not implemented yet}}
      uir.if %done {
        uir.break
      } else {
        uir.if %big {
          uir.break
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

// -----

// An exit-carrying branch with later-phase results would need pick-based
// result reconstruction entangled with the exit chain, not implemented yet.
uir.func @ExitIfLaterResult(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %v, %i : %t
        uir.yield
      }
      %done = hir.geq %i, %n : %t
      // expected-error @below {{carries a loop exit and has later-phase results; not implemented yet}}
      %y = uir.if %done : %t {
        uir.break
      } else {
        %w = uir.expr pin 1 : %t {
          %s = hir.add %x, %i : %t
          uir.yield %s : %t
        }
        uir.yield %w : %t
      }
      uir.expr pin 1 {
        hir.write_var %v, %y : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

// -----

// An early function return under a loop with later-phase content needs
// function-level exit chains, which belong to distribution (M3).
uir.func @ReturnExitUnderContent(%n: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.loop (%i = %zero : %t) {
    uir.expr pin 1 {
      hir.write_var %v, %i : %t
      uir.yield
    }
    %done = hir.geq %i, %n : %t
    uir.if %done {
      // expected-error @below {{exits the function under a loop with later-phase content; function-level exit chains are not implemented yet}}
      uir.return %x : %t
    }
    %next = hir.add %i, %one : %t
    uir.continue %next : %t
  }
  uir.return %x : %t
}

// -----

// A definitely-returning later-phase block followed by more later-phase
// content in the same region: the trailing content is dead and must not
// reach sorting.
uir.func @ContentAfterReturnCarrier(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      // expected-error @below {{definitely returns but is followed by more content in the receiving block}}
      uir.expr pin 1 {
        uir.return %x : %t
      }
      uir.expr pin 1 {
        hir.write_var %v, %x : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  uir.return %x : %t
}

// -----

// An expr terminated by a loop exit cannot be dissolved or carried yet.
uir.func @BreakCarrier(%n: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.loop : %t {
    // expected-error @below {{cannot be dissolved: body is terminated by uir.break}}
    uir.expr {
      uir.break %x : %t
    }
    uir.unreachable
  }
  uir.return %x : %t
}

// -----

// A loop body terminated by a function return, with later-phase content in
// the body, needs function-level exit chains too.
uir.func @ReturnTerminatedBody(%n: 0, %x: 0) -> (result: 0) {
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
    // expected-error @below {{terminates a loop body with later-phase content; function-level exit chains are not implemented yet}}
    uir.return %x : %t
  }
  uir.return %x : %t
}
