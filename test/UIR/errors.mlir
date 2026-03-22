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
