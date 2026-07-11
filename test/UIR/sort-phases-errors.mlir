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

// A loop result at a later phase is a break-value choice, not implemented
// yet.
uir.func @BreakValue(%n: 0, %x: 0) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{provides result 0 at phase 1; break-value choices are not implemented yet}}
  %r = uir.loop : %t {
    %done = hir.geq %x, %n : %t
    uir.if %done {
      %v = uir.expr pin 1 : %t {
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

// Later-phase content on a partial activation path (before a loop exit)
// requires partial activation records, not implemented yet.
uir.func @PartialActivation(%n: -1, %x: 0) -> (result: 0) {
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
      // expected-error @below {{loop exit after later-phase content requires partial activation records; not implemented yet}}
      uir.if %done {
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

// Later-phase content inside an exit-carrying arm is a partial activation
// too.
uir.func @ContentInExitArm(%n: -1, %x: 0) -> (result: 0) {
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
      // expected-error @below {{contains later-phase content on a partial activation path; not implemented yet}}
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %v, %i : %t
          uir.yield
        }
        uir.break
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
  %r = hir.read_var %v : %t
  uir.return %r : %t
}
