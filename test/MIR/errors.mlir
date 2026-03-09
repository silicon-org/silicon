// RUN: silicon-opt --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// ReturnOp: wrong number of operands
//===----------------------------------------------------------------------===//

mir.func @too_few_returns() -> (x: !si.int) {
  // expected-error @below {{has 0 operands but enclosing function returns 1 results}}
  mir.return
}

// -----

mir.func @too_many_returns() -> (x: !si.int) {
  %a = mir.constant #si.int<1> : !si.int
  %b = mir.constant #si.int<2> : !si.int
  // expected-error @below {{has 2 operands but enclosing function returns 1 results}}
  mir.return %a, %b : !si.int, !si.int
}

// -----

//===----------------------------------------------------------------------===//
// ReturnOp: wrong operand type
//===----------------------------------------------------------------------===//

mir.func @wrong_return_type() -> (x: !si.bool) {
  %a = mir.constant #si.int<1> : !si.int
  // expected-error @below {{operand #0 has type '!si.int' but enclosing function returns '!si.bool'}}
  mir.return %a : !si.int
}
