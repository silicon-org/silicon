// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{does not reference a valid `hir.unified_func`}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo(%a: 0) -> () {
  %0 = hir.int_type
  hir.signature (%0) -> ()
} {
  hir.return -> ()
}
// expected-error @below {{has 0 arguments, but @foo expects 1}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} {
  hir.return -> ()
}
// expected-error @below {{has 0 results, but @foo expects 1}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo() -> () {
  hir.signature () -> ()
} {
  // expected-error @below {{cannot appear in the body}}
  hir.signature () -> ()
}

// -----

// expected-error @below {{requires `hir.signature` terminator in the signature}}
hir.unified_func @foo() -> () {
  hir.return -> ()
} {
  hir.return -> ()
}

// -----

hir.unified_func @foo(%a: 0, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  // expected-error @below {{has 1 argument types but parent function has 2 arguments}}
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}

// -----

hir.unified_func @foo(%a: 0) -> (x: 0, y: -1) {
  %0 = hir.int_type
  %1 = hir.int_type
  // expected-error @below {{has 1 result types but parent function has 2 results}}
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{argPhases has 0 entries but call has 1 arguments}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{resultPhases has 0 entries but call has 1 results}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [0] -> []

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{argPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [-1] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{resultPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [0] -> [-1]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{typeOfArgs has 0 entries but call has 1 arguments}}
%2 = hir.unified_call @foo(%arg) : () -> (%int_type) (!hir.any) -> !hir.any [0] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return -> ()
}
%int_type = hir.int_type
%arg = hir.constant_int 42 : %int_type
// expected-error @below {{typeOfResults has 0 entries but call has 1 results}}
%2 = hir.unified_call @foo(%arg) : (%int_type) -> () (!hir.any) -> !hir.any [0] -> [0]

// -----

// expected-error @below {{requires `hir.signature` terminator in the signature}}
hir.split_func @bad_term() -> () {
^bb0:
  cf.br ^bb1
^bb1:
  cf.br ^bb1
} [
  0: @bad_term.0
]

// -----

hir.split_func @bad_sig_args() -> () {
  %0 = hir.int_type
  // expected-error @below {{has 1 argument types but parent function has 0 arguments}}
  hir.signature (%0) -> ()
} [
  0: @bad_sig_args.0
]

// -----

hir.split_func @bad_sig_results() -> () {
  %0 = hir.int_type
  // expected-error @below {{has 1 result types but parent function has 0 results}}
  hir.signature () -> (%0)
} [
  0: @bad_sig_results.0
]

// -----

// expected-error @below {{phaseFuncs must have at least one entry}}
hir.multiphase_func @empty_phases() -> (out) []

// -----

// YieldOp must be inside an ExprOp
func.func @yield_outside() {
  // expected-error @below {{'hir.yield' op expects parent op 'hir.expr'}}
  hir.yield
}

// -----

// YieldOp operand count must match ExprOp result count (too few)
hir.expr 0 : !hir.any, !hir.any {
  // expected-error @below {{'hir.yield' op has 0 operands but parent expr has 2 results}}
  hir.yield
}

// -----

// YieldOp operand count must match ExprOp result count (too many)
hir.expr 0 {
  %0 = hir.int_type
  // expected-error @below {{'hir.yield' op has 1 operands but parent expr has 0 results}}
  hir.yield %0 : !hir.any
}

// -----

// expected-error @below {{block argument must have type !hir.any, got '!si.int'}}
"hir.func"() ({
  "hir.signature"() : () -> ()
}, {
^bb0(%arg0: !si.int):
  "mir.return"() : () -> ()
}) {sym_name = "bad_block_arg_type", argNames = ["arg0"], resultNames = []} : () -> ()

// -----

// expected-error @below {{requires `hir.signature` terminator in the signature}}
hir.func @missing_sig_term() -> () {
^bb0:
  cf.br ^bb1
^bb1:
  cf.br ^bb1
} {
  mir.return
}

// -----

hir.func @sig_in_body() -> () {
  hir.signature () -> ()
} {
  // expected-error @below {{cannot appear in the body}}
  hir.signature () -> ()
}

// -----

hir.func @sig_arg_mismatch() -> () {
  %0 = hir.int_type
  // expected-error @below {{has 1 argument types but parent function has 0 arguments}}
  hir.signature (%0) -> ()
} {
  mir.return
}

// -----

hir.func @sig_result_mismatch() -> () {
  %0 = hir.int_type
  // expected-error @below {{has 1 result types but parent function has 0 results}}
  hir.signature () -> (%0)
} {
  mir.return
}

// -----

// expected-error @below {{signature region has 1 block arguments but function has 0 arguments}}
hir.func @sig_block_arg_mismatch() -> () {
  ^bb0(%x: !hir.any):
  hir.signature () -> ()
} {
  mir.return
}

// -----

hir.func @return_values_mismatch() -> (a, b) {
  %t = hir.int_type
  hir.signature () -> (%t, %t)
} {
  %t = hir.int_type
  %0 = hir.constant_int 42 : %t
  // expected-error @below {{has 1 values but parent function has 2 results}}
  hir.return %0 -> (%t)
}
