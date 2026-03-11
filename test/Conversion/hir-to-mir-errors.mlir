// RUN: silicon-opt --lower-hir-to-mir --verify-diagnostics --split-input-file %s

hir.func @UntypedArg(%a) -> () {
  // expected-error @below {{'hir.signature' op has 0 argument types but parent function has 1 arguments}}
  hir.signature () -> ()
} {
  hir.return -> ()
}

// -----

// Verify that hir.unify with different concrete types produces an error during
// HIR-to-MIR lowering.

hir.func @UnifyTypeMismatch() -> (result) {
  %int = hir.int_type
  %unit = hir.unit_type
  %ty = hir.unify %int, %unit
  hir.signature () -> (%ty)
} {
  %int = hir.int_type
  %unit = hir.unit_type
  // expected-error @below {{hir.unify survived to HIR-to-MIR lowering with different operands}}
  // expected-note @below {{}}
  // expected-error @below {{failed to legalize operation 'hir.unify'}}
  %ty = hir.unify %int, %unit
  %c0 = hir.constant_int 0 : %ty
  hir.return %c0 -> (%ty)
}

// -----

// Negative uint width: shouldLower passes (ConstantIntOp exists), but the
// conversion pattern rejects the negative value.

hir.func @NegativeUIntWidth() -> () {
  hir.signature () -> ()
} {
  %int = hir.int_type
  %neg = hir.constant_int -1 : %int
  // expected-error @below {{compiler bug: negative uint width -1}}
  // expected-note @below {{}}
  // expected-error @below {{failed to legalize operation 'hir.uint_type'}}
  %ty = hir.uint_type %neg
  hir.return -> ()
}

// -----

// Coerce_type type mismatch: the return op declares the arg type as bool, but
// the coerce_type annotates it as int. After signature conversion the block arg
// becomes !si.bool, conflicting with the !si.int expected by the coerce_type.

hir.func @CoerceTypeMismatch(%a) -> (result) {
  %bool = hir.bool_type
  %int = hir.int_type
  hir.signature (%bool) -> (%int)
} {
  %bool = hir.bool_type
  %int = hir.int_type
  // expected-error @below {{compiler bug: coerce_type type mismatch: input has '!si.bool' but type operand says '!si.int'}}
  // expected-note @below {{}}
  // expected-error @below {{failed to legalize operation 'hir.coerce_type'}}
  %r = hir.coerce_type %a, %int
  hir.return %r -> (%int)
}
