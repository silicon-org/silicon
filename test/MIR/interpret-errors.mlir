// RUN: silicon-opt --allow-unregistered-dialect --interpret %s --split-input-file --verify-diagnostics

// Missing operand value: unrealized_conversion_cast with zero inputs produces
// a result that the interpreter never records a value for. Using that result
// in a subsequent operation triggers the "missing value" error.

mir.func @MissingOperandValue() -> (result: !si.int) {
  %0 = "builtin.unrealized_conversion_cast"() : () -> !si.int
  // expected-error @below {{missing value for operand #0}}
  mir.return %0 : !si.int
}

// -----

// Callee not a mir.func: calling a symbol that resolves to a
// mir.evaluated_func instead of a mir.func.

mir.evaluated_func @not_a_func [#si.int<42> : !si.int]

mir.func @CallNonFunc() -> (result: !si.int) {
  // expected-error @below {{callee @not_a_func is not a mir.func (may not have been lowered yet)}}
  %0 = mir.call @not_a_func() : () -> !si.int
  mir.return %0 : !si.int
}

// -----

// Unsupported operation: an unregistered op the interpreter doesn't handle.

mir.func @UnsupportedOp() -> (result: !si.int) {
  %0 = mir.constant #si.int<1> : !si.int
  // expected-error @below {{operation not supported by interpreter}}
  %1 = "test.unknown_op"(%0) : (!si.int) -> !si.int
  mir.return %1 : !si.int
}
