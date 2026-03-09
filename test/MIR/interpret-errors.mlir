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

// -----

// Division by zero.

mir.func @DivByZero() -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.constant #si.int<0> : !si.int
  // expected-error @below {{division by zero}}
  %2 = mir.div %0, %1 : !si.int
  mir.return %2 : !si.int
}

// -----

// Modulo by zero.

mir.func @ModByZero() -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.constant #si.int<0> : !si.int
  // expected-error @below {{modulo by zero}}
  %2 = mir.mod %0, %1 : !si.int
  mir.return %2 : !si.int
}

// -----

// Shift left amount out of range (>= 64).

mir.func @ShlOutOfRange() -> (result: !si.int) {
  %0 = mir.constant #si.int<1> : !si.int
  %1 = mir.constant #si.int<64> : !si.int
  // expected-error @below {{shift amount 64 is out of range [0, 64)}}
  %2 = mir.shl %0, %1 : !si.int
  mir.return %2 : !si.int
}

// -----

// Shift right with negative amount.

mir.func @ShrNegative() -> (result: !si.int) {
  %0 = mir.constant #si.int<16> : !si.int
  %1 = mir.constant #si.int<-1> : !si.int
  // expected-error @below {{shift amount -1 is out of range [0, 64)}}
  %2 = mir.shr %0, %1 : !si.int
  mir.return %2 : !si.int
}

// -----

// Unknown condition type in cf.cond_br: the interpreter only handles BoolAttr,
// IntAttr, and IntegerAttr conditions. An OpaqueAttr triggers a compiler bug.

mir.func @CondBrUnknownCondType() -> (result: !si.int) {
  %a = mir.constant #si.int<1> : !si.int
  %b = mir.constant #si.int<2> : !si.int
  %opaque = mir.opaque_pack (%a) : (!si.int)
  %cond = "builtin.unrealized_conversion_cast"(%opaque) : (!si.opaque) -> i1
  // expected-error @below {{compiler bug: unsupported condition type}}
  // expected-note @below {{Interpreter::run()}}
  cf.cond_br %cond, ^then, ^else
^then:
  cf.br ^merge(%a : !si.int)
^else:
  cf.br ^merge(%b : !si.int)
^merge(%result: !si.int):
  mir.return %result : !si.int
}
