// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// -----

// Verify that uir.yield rejects mismatched values/typeOfValues.
func.func @yield_mismatch(%a: !hir.any, %ty: !hir.any) {
  // expected-error @below {{failed to verify that typeOfValues must match values size}}
  "uir.yield"(%a, %a, %ty) {operandSegmentSizes = array<i32: 2, 1>} : (!hir.any, !hir.any, !hir.any) -> ()
}

// -----

// Verify that uir.break requires a uir.loop ancestor.
func.func @break_outside_loop(%a: !hir.any, %ty: !hir.any) {
  // expected-error @below {{must be nested inside a 'uir.loop'}}
  uir.break %a : %ty
}

// -----

// Verify that uir.continue requires a uir.loop ancestor.
func.func @continue_outside_loop() {
  // expected-error @below {{must be nested inside a 'uir.loop'}}
  uir.continue
}

// -----

// Verify that uir.return rejects mismatched values/typeOfValues.
func.func @return_mismatch(%a: !hir.any, %ty: !hir.any) {
  // expected-error @below {{failed to verify that typeOfValues must match values size}}
  "uir.return"(%a, %a, %ty) {operandSegmentSizes = array<i32: 2, 1>} : (!hir.any, !hir.any, !hir.any) -> ()
}

// -----

// Verify that uir.if without else cannot have results.
func.func @if_no_else_with_results(%cond: !hir.any, %ty: !hir.any) {
  // expected-error @below {{if without 'else' cannot produce results}}
  %r = "uir.if"(%cond, %ty) ({
    uir.yield
  }, {
  }) : (!hir.any, !hir.any) -> !hir.any
}

// -----

// Verify that uir.if then region must have at least one block.
func.func @if_then_empty(%cond: !hir.any) {
  // expected-error @below {{region #0 ('thenRegion') failed to verify constraint: region with at least 1 blocks}}
  "uir.if"(%cond) ({
  }, {
  }) : (!hir.any) -> ()
}

// -----

// Verify that uir.loop body must have at least one block.
func.func @loop_empty_body() {
  // expected-error @below {{region #0 ('body') failed to verify constraint: region with at least 1 blocks}}
  "uir.loop"() ({
  }) : () -> ()
}

// -----

// Verify that floating uir.expr cannot have non-zero phaseShift.
func.func @expr_floating_with_shift(%ty: !hir.any) {
  // expected-error @below {{floating expression (no 'pin') must have phaseShift = 0}}
  %r = "uir.expr"(%ty) ({
    "uir.yield"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  }) {phaseShift = -1 : si32} : (!hir.any) -> !hir.any
}

// -----

// Verify that uir.signature checks arg count against parent.
uir.func @sig_arg_mismatch(%a: 0) -> () {
  // expected-error @below {{has 0 argument types but parent function has 1 arguments}}
  uir.signature () -> ()
} {
  uir.return
}

// -----

// Verify that uir.func rejects mismatched argNames/argPhases.
// expected-error @below {{argNames has 2 entries but function has 1 arguments}}
"uir.func"() ({
  "uir.signature"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
}, {
  "uir.return"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
}) {sym_name = "bad", argPhases = array<i32: 0>, resultPhases = array<i32>, argNames = ["a", "b"], resultNames = []} : () -> ()

// -----

// Verify that uir.call rejects non-existent callee.
uir.func @call_bad_callee(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{'nonexistent' does not reference a valid function}}
  %r = uir.call @nonexistent(%a) : (%a) -> (%a) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %a
}

// -----

// Verify that uir.continue rejects mismatched values/typeOfValues.
func.func @continue_mismatch(%a: !hir.any) {
  uir.loop (%x = %a : %a) {
    // expected-error @below {{failed to verify that typeOfValues must match values size}}
    "uir.continue"(%a, %a, %a) {operandSegmentSizes = array<i32: 2, 1>} : (!hir.any, !hir.any, !hir.any) -> ()
  }
  func.return
}

// -----

// Verify that uir.loop checks init count against init type operand count.
func.func @loop_init_type_mismatch(%a: !hir.any) {
  // expected-error @below {{init count must match init type operand count}}
  "uir.loop"(%a, %a, %a) ({
  ^bb0(%x: !hir.any, %y: !hir.any):
    uir.continue %x, %y : %a, %a
  }) {operandSegmentSizes = array<i32: 2, 1, 0>} : (!hir.any, !hir.any, !hir.any) -> ()
  func.return
}

// -----

// Verify that uir.loop checks body block argument count against init count.
func.func @loop_block_arg_mismatch(%a: !hir.any) {
  // expected-error @below {{body block argument count must match init count}}
  "uir.loop"(%a, %a, %a, %a) ({
  ^bb0(%x: !hir.any):
    uir.continue %x, %x : %a, %a
  }) {operandSegmentSizes = array<i32: 2, 2, 0>} : (!hir.any, !hir.any, !hir.any, !hir.any) -> ()
  func.return
}

// -----

// Verify that uir.continue checks carried value count against the loop's
// iteration argument count.
func.func @continue_arity_mismatch(%a: !hir.any) {
  uir.loop (%x = %a : %a) {
    // expected-error @below {{carried value count must match enclosing loop's iteration argument count}}
    uir.continue
  }
  func.return
}

// -----

// Verify that a uir.continue nested inside a uir.if is checked against the
// enclosing loop's iteration argument count.
func.func @continue_nested_arity_mismatch(%cond: !hir.any, %a: !hir.any) {
  uir.loop (%x = %a : %a) {
    uir.if %cond {
      // expected-error @below {{carried value count must match enclosing loop's iteration argument count}}
      uir.continue %x, %x : %a, %a
    }
    uir.continue %x : %a
  }
  func.return
}

// -----

// Verify that uir.signature cannot terminate the function body.
uir.func @sig_in_body(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{expected 'uir.return' or 'uir.unreachable' terminator in function body}}
  uir.signature (%a) -> ()
}

// -----

// Verify that uir.phase requires a uir.func parent.
func.func @phase_wrong_parent() {
  // expected-error @below {{expects parent op 'uir.func'}}
  uir.phase 0 {
    uir.phase_return
  }
  func.return
}

// -----

// Verify that uir.phase rejects terminators other than uir.phase_return and
// uir.unreachable.
uir.func @phase_bad_terminator(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.phase 0 {
    // expected-error @below {{cannot terminate a 'uir.phase' body; use 'uir.phase_return' or 'uir.unreachable'}}
    uir.yield
  }
  uir.unreachable
}

// -----

// Verify that uir.return cannot appear inside a uir.phase, even nested.
uir.func @return_in_phase(%c: 0, %x: 0) -> () {
  uir.signature (%c, %x) -> ()
} {
  uir.phase 0 {
    uir.if %c {
      // expected-error @below {{cannot appear inside a 'uir.phase'}}
      uir.return
    }
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that a uir.phase without any uir.phase_return cannot have results.
uir.func @phase_results_without_return(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{has 1 results but no 'uir.phase_return' to provide them}}
  %p = "uir.phase"() ({
    uir.unreachable
  }) {phase = 0 : si32} : () -> (!hir.any)
  uir.unreachable
}

// -----

// Verify that uir.phase_return requires a uir.phase ancestor.
func.func @phase_return_outside() {
  // expected-error @below {{must be nested inside a 'uir.phase'}}
  uir.phase_return
}

// -----

// Verify that all phase returns of a phase agree on the yield count.
uir.func @phase_yield_mismatch(%c: 0, %x: 0) -> () {
  uir.signature (%c, %x) -> ()
} {
  %p = uir.phase 0 {
    uir.if %c {
      uir.phase_return yield %x
    } else {
      uir.yield
    }
    // expected-error @below {{yields 0 values but enclosing 'uir.phase' has 1 results}}
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects a non-existent callee.
uir.func @phase_call_bad_callee(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{'nonexistent' does not reference a valid function}}
  %r = uir.phase_call @nonexistent[0] (%a) : (%a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call rejects callees that are not unified or split
// functions.
func.func private @not_a_uir_func() {
  func.return
}
uir.func @phase_call_wrong_callee_kind(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{'not_a_uir_func' does not reference a 'uir.func' or 'uir.split_func'}}
  %r = uir.phase_call @not_a_uir_func[0] (%a) : (%a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call rejects phases the callee does not expose.
uir.func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_bad_phase(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{callee @pc_callee has no externally visible phase 1}}
  %r = uir.phase_call @pc_callee[1] (%a) : (%a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call checks the argument count against the callee's
// arguments declared at the called phase.
uir.func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_arg_count(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{has 2 arguments but callee @pc_callee declares 1 arguments at phase 0}}
  %r = uir.phase_call @pc_callee[0] (%a, %a) : (%a, %a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call checks the result count against the callee's
// results declared at the called phase.
uir.func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_result_count(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{has 0 results but callee @pc_callee declares 1 results at phase 0}}
  uir.phase_call @pc_callee[0] (%a) : (%a) -> ()
  uir.return %a : %a
}

// -----

// Verify that non-first phase entries require a context operand.
uir.func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_missing_context(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{requires a context operand: phase 0 is not the callee's first externally visible phase}}
  %r = uir.phase_call @pc_callee2[0] (%a) : (%a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that first phase entries reject a context operand.
uir.func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_spurious_context(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{cannot have a context operand: phase -1 is the callee's first externally visible phase}}
  %ctx = uir.phase_call @pc_callee2[-1] (%a) context %a : (%a) -> () context
  uir.return %a : %a
}

// -----

// Verify that non-last phase entries require a context result.
uir.func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_missing_context_out(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{requires a context result: phase -1 is not the callee's last externally visible phase}}
  uir.phase_call @pc_callee2[-1] (%a) : (%a) -> ()
  uir.return %a : %a
}

// -----

// Verify that last phase entries reject a context result.
uir.func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_spurious_context_out(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{cannot have a context result: phase 0 is the callee's last externally visible phase}}
  %r, %ctx = uir.phase_call @pc_callee2[0] (%a) context %a : (%a) -> (%a) context
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call rejects mismatched arguments/typeOfArgs.
uir.func @phase_call_arg_type_mismatch(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{has 1 arguments but 0 typeOfArgs}}
  "uir.phase_call"(%a) {callee = @pc_callee, calleePhase = 0 : si32, operandSegmentSizes = array<i32: 1, 0, 0, 0>, resultSegmentSizes = array<i32: 0, 0>} : (!hir.any) -> ()
  uir.return
}

// -----

// Verify that uir.phase_call rejects mismatched results/typeOfResults.
uir.func @phase_call_result_type_mismatch(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{has 1 results but 0 typeOfResults}}
  %r = "uir.phase_call"() {callee = @pc_callee, calleePhase = 0 : si32, operandSegmentSizes = array<i32: 0, 0, 0, 0>, resultSegmentSizes = array<i32: 1, 0>} : () -> (!hir.any)
  uir.return
}
