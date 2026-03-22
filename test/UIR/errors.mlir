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
