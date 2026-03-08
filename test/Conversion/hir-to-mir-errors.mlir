// RUN: silicon-opt --lower-hir-to-mir --verify-diagnostics --split-input-file %s

// expected-error @below {{return op typeOfArgs count (0) does not match block argument count (1)}}
// expected-note @below {{}}
// expected-error @below {{failed to legalize operation 'hir.func'}}
hir.func @UntypedArg(%a) -> () {
  hir.return : () -> ()
}

// -----

// Verify that hir.unify with different concrete types produces an error during
// HIR-to-MIR lowering.

hir.func @UnifyTypeMismatch() -> (result) {
  %int = hir.int_type
  %unit = hir.unit_type
  // expected-error @below {{hir.unify survived to HIR-to-MIR lowering with different operands}}
  // expected-note @below {{}}
  // expected-error @below {{failed to legalize operation 'hir.unify'}}
  %ty = hir.unify %int, %unit
  %c0 = hir.constant_int 0
  hir.return %c0 : () -> (%ty)
}
