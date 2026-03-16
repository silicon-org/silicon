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

// -----

//===----------------------------------------------------------------------===//
// FuncOp: block argument count mismatch
//===----------------------------------------------------------------------===//

// expected-error @below {{'mir.func' op expects 1 arguments but block has 0 arguments}}
"mir.func"() <{argNames = ["x"], function_type = (!si.int) -> (), resultNames = [], sym_name = "arg_count_too_few"}> ({
  "mir.return"() : () -> ()
}) : () -> ()

// -----

// expected-error @below {{'mir.func' op expects 0 arguments but block has 1 arguments}}
"mir.func"() <{argNames = [], function_type = () -> (), resultNames = [], sym_name = "arg_count_too_many"}> ({
^bb0(%arg0: !si.int):
  "mir.return"() : () -> ()
}) : () -> ()

// -----

//===----------------------------------------------------------------------===//
// FuncOp: block argument type mismatch
//===----------------------------------------------------------------------===//

// expected-error @below {{'mir.func' op block argument #0 has type '!si.bool' but function expects '!si.int'}}
"mir.func"() <{argNames = ["x"], function_type = (!si.int) -> (), resultNames = [], sym_name = "arg_type_mismatch"}> ({
^bb0(%arg0: !si.bool):
  "mir.return"() : () -> ()
}) : () -> ()

// -----

//===----------------------------------------------------------------------===//
// FuncOp: argNames count mismatch
//===----------------------------------------------------------------------===//

// expected-error @below {{'mir.func' op argNames has 2 entries but function has 1 arguments}}
"mir.func"() <{argNames = ["x", "y"], function_type = (!si.int) -> (), resultNames = [], sym_name = "argnames_mismatch"}> ({
^bb0(%arg0: !si.int):
  "mir.return"() : () -> ()
}) : () -> ()

// -----

//===----------------------------------------------------------------------===//
// FuncOp: resultNames count mismatch
//===----------------------------------------------------------------------===//

// expected-error @below {{'mir.func' op resultNames has 2 entries but function has 1 results}}
"mir.func"() <{argNames = [], function_type = () -> !si.int, resultNames = ["a", "b"], sym_name = "resultnames_mismatch"}> ({
  %0 = "mir.constant"() <{value = #si.int<1>}> : () -> !si.int
  "mir.return"(%0) : (!si.int) -> ()
}) : () -> ()

// -----

//===----------------------------------------------------------------------===//
// CallOp: wrong argument count
//===----------------------------------------------------------------------===//

mir.func @callee(%x: !si.int) -> (result: !si.int) {
  mir.return %x : !si.int
}

mir.func @call_wrong_arg_count() -> () {
  // expected-error @below {{'mir.call' op has 0 arguments but callee @callee expects 1}}
  mir.call @callee() : () -> !si.int
  mir.return
}

// -----

//===----------------------------------------------------------------------===//
// CallOp: wrong argument type
//===----------------------------------------------------------------------===//

mir.func @callee_int(%x: !si.int) -> (result: !si.int) {
  mir.return %x : !si.int
}

mir.func @call_wrong_arg_type() -> () {
  %b = mir.constant #si.bool<true> : !si.bool
  // expected-error @below {{'mir.call' op argument #0 has type '!si.bool' but callee @callee_int expects '!si.int'}}
  mir.call @callee_int(%b) : (!si.bool) -> !si.int
  mir.return
}

// -----

//===----------------------------------------------------------------------===//
// CallOp: wrong result count
//===----------------------------------------------------------------------===//

mir.func @callee_no_result() -> () {
  mir.return
}

mir.func @call_wrong_result_count() -> () {
  // expected-error @below {{'mir.call' op has 1 results but callee @callee_no_result expects 0}}
  %r = mir.call @callee_no_result() : () -> !si.int
  mir.return
}

// -----

//===----------------------------------------------------------------------===//
// CallOp: wrong result type
//===----------------------------------------------------------------------===//

mir.func @callee_returns_int() -> (result: !si.int) {
  %c = mir.constant #si.int<1> : !si.int
  mir.return %c : !si.int
}

mir.func @call_wrong_result_type() -> () {
  // expected-error @below {{'mir.call' op result #0 has type '!si.bool' but callee @callee_returns_int expects '!si.int'}}
  %r = mir.call @callee_returns_int() : () -> !si.bool
  mir.return
}
