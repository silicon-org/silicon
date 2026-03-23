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
  uir.return -> ()
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
  uir.return %r -> (%a)
}

// -----

// Verify that uir.yield inside loop body must have no values.
uir.func @loop_yield_with_values(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.loop {
    // expected-error @below {{inside loop body must have no values (use 'uir.break' to exit with values)}}
    uir.yield %a : %a
  }
  uir.return -> ()
}

// -----

// Verify that uir.signature cannot appear in function body.
uir.func @sig_in_body(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{cannot appear in function body}}
  uir.signature (%a) -> ()
}
