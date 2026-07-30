// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{does not reference a valid `hir.unified_func`}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo(%a: 0) -> () {
  %0 = hir.int_type
  hir.signature (%0) -> ()
} {
  hir.return
}
// expected-error @below {{has 0 arguments, but @foo expects 1}}
hir.unified_call @foo() : () -> () () -> () [] -> []

// -----

hir.unified_func @foo() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} {
  hir.return
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
  hir.return
} {
  hir.return
}

// -----

hir.unified_func @foo(%a: 0, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  // expected-error @below {{has 1 argument types but parent function has 2 arguments}}
  hir.signature (%0) -> (%1)
} {
  hir.return
}

// -----

hir.unified_func @foo(%a: 0) -> (x: 0, y: -1) {
  %0 = hir.int_type
  %1 = hir.int_type
  // expected-error @below {{has 1 result types but parent function has 2 results}}
  hir.signature (%0) -> (%1)
} {
  hir.return
}

// -----

hir.unified_func @foo(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  hir.return
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
  hir.return
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
  hir.return
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
  hir.return
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
  hir.return
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
  hir.return
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
  // expected-error @below {{'hir.yield' op expects parent op to be one of 'hir.expr, hir.replicate, hir.pick'}}
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
  hir.return %0 : %t
}

// -----

// typeOfValues count mismatch with values (too few typeOfValues).
hir.unified_func @return_typeofvalues_too_few() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 42 : %0
  // expected-error @below {{typeOfValues must match values size}}
  "hir.return"(%1) <{operandSegmentSizes = array<i32: 1, 0>}> : (!hir.any) -> ()
}

// -----

// typeOfValues count mismatch with values (too many typeOfValues).
hir.unified_func @return_typeofvalues_too_many() -> () {
  hir.signature () -> ()
} {
  %0 = hir.int_type
  // expected-error @below {{typeOfValues must match values size}}
  "hir.return"(%0) <{operandSegmentSizes = array<i32: 0, 1>}> : (!hir.any) -> ()
}

// -----

// replicate: bodies have no results, so yields must carry no operands.
func.func @replicate_yield_mismatch(%hits: !hir.any, %x: !hir.any) {
  hir.replicate (%a) in %hits {
    // expected-error @below {{has 1 operands but parent replicate has no results}}
    hir.yield %a : !hir.any
  }
  return
}

// -----

// pick: must have at least one alternative.
func.func @pick_no_alternatives(%choice: !hir.any) {
  // expected-error @below {{must have at least one alternative}}
  "hir.pick"(%choice) : (!hir.any) -> ()
  return
}

// -----

// pick: yield operand count must match pick result count.
func.func @pick_yield_mismatch(%choice: !hir.any, %x: !hir.any) {
  %r = hir.pick %choice : !hir.any [
    0 (): {
      // expected-error @below {{has 0 operands but parent pick has 1 results}}
      hir.yield
    }
  ]
  return
}

// -----

// pick: alternative tags must be dense region indices.
func.func @pick_bad_tag(%choice: !hir.any) {
  hir.pick %choice [
    // expected-error @below {{expected tag 0, got 1}}
    1 (): {
      hir.yield
    }
  ]
  return
}

// -----

// eager_func: at least two links are required.
// expected-error @below {{phaseFuncs must have at least two entries}}
hir.eager_func @one_link() -> (out) [
  @one_link.0
]

// -----

// eager_func: the head link must take no arguments.
hir.func @head_args.head(%a) -> (ctx) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @head_args.tail(%ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @head_args.head has 1 arguments, expected 0}}
hir.eager_func @head_args() -> (out) [
  @head_args.head,
  @head_args.tail
]

// -----

// eager_func: the head link must produce exactly one result.
hir.func @head_res.head() -> (c1, c2) {
  %o = hir.opaque_type
  hir.signature () -> (%o, %o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c, %c : %o, %o
}
hir.func @head_res.tail(%ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @head_res.head has 2 results, expected 1}}
hir.eager_func @head_res() -> (out) [
  @head_res.head,
  @head_res.tail
]

// -----

// eager_func: a middle link must take exactly one argument.
hir.func @mid_arity.head() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @mid_arity.mid(%a, %b) -> (ctx) {
  %o = hir.opaque_type
  hir.signature (%o, %o) -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @mid_arity.tail(%ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @mid_arity.mid has 2 arguments, expected 1}}
hir.eager_func @mid_arity() -> (out) [
  @mid_arity.head,
  @mid_arity.mid,
  @mid_arity.tail
]

// -----

// eager_func: the tail link takes the declared args plus one context arg.
hir.func @tail_args.head() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @tail_args.tail(%x) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @tail_args.tail has 1 arguments, expected 2}}
hir.eager_func @tail_args(x) -> (out) [
  @tail_args.head,
  @tail_args.tail
]

// -----

// eager_func: the tail link produces exactly the declared results.
hir.func @tail_res.head() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @tail_res.tail(%x, %ctx) -> () {
  %o = hir.opaque_type
  hir.signature (%o, %o) -> ()
} {
  hir.return
}
// expected-error @below {{link @tail_res.tail has 0 results, expected 1}}
hir.eager_func @tail_res(x) -> (out) [
  @tail_res.head,
  @tail_res.tail
]

// -----

// eager_func: every link must resolve to a symbol.
hir.func @unresolved.tail(%ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @nope does not reference a valid symbol}}
hir.eager_func @unresolved() -> (out) [
  @nope,
  @unresolved.tail
]

// -----

// eager_func: every link must resolve to an `hir.func`.
func.func @not_a_func() {
  return
}
hir.func @wrong_kind.tail(%ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
// expected-error @below {{link @not_a_func must reference an `hir.func`}}
hir.eager_func @wrong_kind() -> (out) [
  @not_a_func,
  @wrong_kind.tail
]

// -----

// hir.call: argument count must match an eager_func callee.
hir.func @call_mismatch.head() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %c = hir.opaque_pack()
  hir.return %c : %o
}
hir.func @call_mismatch.tail(%a, %b, %ctx) -> (out) {
  %o = hir.opaque_type
  hir.signature (%o, %o, %o) -> (%o)
} {
  %o = hir.opaque_type
  %r = hir.opaque_pack()
  hir.return %r : %o
}
hir.eager_func @call_mismatch(a, b) -> (out) [
  @call_mismatch.head,
  @call_mismatch.tail
]
%cm_ty = hir.int_type
// expected-error @below {{has 1 arguments, but callee @call_mismatch expects 2}}
hir.call @call_mismatch(%cm_ty) : (%cm_ty) -> (%cm_ty)
