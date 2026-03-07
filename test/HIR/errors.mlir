// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{does not reference a valid `hir.unified_func`}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo(%a: 0) -> () {
  %0 = hir.int_type
  hir.unified_signature (%0) -> ()
} {
  hir.unified_return
}
// expected-error @below {{has 0 arguments, but @foo expects 1}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo() -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  hir.unified_return
}
// expected-error @below {{has 0 results, but @foo expects 1}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo() -> () {
  // expected-error @below {{can only appear in the last block}}
  hir.unified_signature () -> ()
^bb1:
  hir.unified_signature () -> ()
} {
  hir.unified_return
}

// -----

// expected-error @below {{requires `hir.unified_return` terminator in the body}}
hir.unified_func @foo() -> () {
  hir.unified_signature () -> ()
} {
  hir.unified_signature () -> ()
}

// -----

// expected-error @below {{requires `hir.unified_signature` terminator in the signature}}
hir.unified_func @foo() -> () {
  hir.unified_return
} {
  hir.unified_return
}

// -----

// expected-error @below {{signature has 1 argument types but function has 2 arguments}}
hir.unified_func @foo(%a: 0, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}

// -----

// expected-error @below {{signature has 1 result types but function has 2 results}}
hir.unified_func @foo(%a: 0) -> (x: 0, y: -1) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{argPhases has 0 entries but call has 1 arguments}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{resultPhases has 0 entries but call has 1 results}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [0] -> []

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{argPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [-1] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{resultPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (%int_type) -> (%int_type) (!hir.any) -> !hir.any [0] -> [-1]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{typeOfArgs has 0 entries but call has 1 arguments}}
%2 = hir.unified_call @foo(%arg) : () -> (%int_type) (!hir.any) -> !hir.any [0] -> [0]

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
%int_type = hir.int_type
// expected-error @below {{typeOfResults has 0 entries but call has 1 results}}
%2 = hir.unified_call @foo(%arg) : (%int_type) -> () (!hir.any) -> !hir.any [0] -> [0]

// -----

// expected-error @below {{requires `hir.signature` terminator in the signature}}
hir.split_func @bad_term() -> () {
^bb0:
  hir.const_br ^bb1
^bb1:
  hir.const_br ^bb1
} [
  0: @bad_term.0
]

// -----

// expected-error @below {{signature has 1 argument types but function has 0 arguments}}
hir.split_func @bad_sig_args() -> () {
  %0 = hir.int_type
  hir.signature (%0) -> ()
} [
  0: @bad_sig_args.0
]

// -----

// expected-error @below {{signature has 1 result types but function has 0 results}}
hir.split_func @bad_sig_results() -> () {
  %0 = hir.int_type
  hir.signature () -> (%0)
} [
  0: @bad_sig_results.0
]

// -----

// expected-error @below {{phaseFuncs must have at least one entry}}
hir.multiphase_func @empty_phases() -> (out) []

// -----

// YieldOp must be inside an ExprOp or IfOp
func.func @yield_outside() {
  // expected-error @below {{'hir.yield' op expects parent op to be one of 'hir.expr, hir.if'}}
  hir.yield
}

// -----

// expected-error @below {{block argument must have type !hir.any, got '!si.int'}}
"hir.func"() ({
^bb0(%arg0: !si.int):
  "mir.return"() : () -> ()
}) {sym_name = "bad_block_arg_type", argNames = ["arg0"], resultNames = []} : () -> ()
