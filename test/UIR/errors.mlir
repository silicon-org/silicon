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

// -----

// Verify that uir.if without else cannot have results.
func.func @if_no_else_with_results(%cond: !hir.any, %ty: !hir.any) {
  // expected-error @below {{if without 'else' cannot produce results}}
  %r = "uir.if"(%cond, %ty) ({
    uir.yield
  }, {
  }) : (!hir.any, !hir.any) -> !hir.any
}

// -----

// Verify that uir.if then region must have at least one block.
func.func @if_then_empty(%cond: !hir.any) {
  // expected-error @below {{region #0 ('thenRegion') failed to verify constraint: region with at least 1 blocks}}
  "uir.if"(%cond) ({
  }, {
  }) : (!hir.any) -> ()
}

// -----

// Verify that uir.loop body must have at least one block.
func.func @loop_empty_body() {
  // expected-error @below {{region #0 ('body') failed to verify constraint: region with at least 1 blocks}}
  "uir.loop"() ({
  }) : () -> ()
}

// -----

// Verify that floating uir.expr cannot have non-zero phaseShift.
func.func @expr_floating_with_shift(%ty: !hir.any) {
  // expected-error @below {{floating expression (no 'pin') must have phaseShift = 0}}
  %r = "uir.expr"(%ty) ({
    "uir.yield"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
  }) {phaseShift = -1 : si32} : (!hir.any) -> !hir.any
}

// -----

// Verify that uir.signature checks arg count against parent.
uir.func @sig_arg_mismatch(%a: 0) -> () {
  // expected-error @below {{has 0 argument types but parent function has 1 arguments}}
  uir.signature () -> ()
} {
  uir.return
}

// -----

// Verify that uir.func rejects mismatched argNames/argPhases.
// expected-error @below {{argNames has 2 entries but function has 1 arguments}}
"uir.func"() ({
  "uir.signature"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
}, {
  "uir.return"() {operandSegmentSizes = array<i32: 0, 0>} : () -> ()
}) {sym_name = "bad", argPhases = array<i32: 0>, resultPhases = array<i32>, argNames = ["a", "b"], resultNames = []} : () -> ()

// -----

// Verify that uir.call rejects non-existent callee.
uir.func @call_bad_callee(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{'nonexistent' does not reference a valid function}}
  %r = uir.call @nonexistent(%a) : (%a) -> (%a) [0] -> [0]
  uir.return %r : %a
}

// -----

// Verify that uir.call rejects a callee that declares no phases: decomposition
// derives the callee's phase span from its declaration alone, so a callee that
// is neither a uir.func nor a uir.split_func cannot be called.
hir.func @call_not_a_uir_func() -> () {
  hir.signature () -> ()
} {
  hir.return
}

uir.func @call_bad_callee_kind(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{'call_not_a_uir_func' does not reference a 'uir.func' or 'uir.split_func'}}
  uir.call @call_not_a_uir_func() : () -> () [] -> []
  uir.return
}

// -----

// Verify that uir.call rejects mismatched arguments/typeOfArgs.
func.func @call_type_of_args_mismatch(%a: !hir.any) {
  // expected-error @below {{failed to verify that typeOfArgs must match arguments size}}
  "uir.call"(%a) {callee = @nowhere, argPhases = array<i32: 0>, resultPhases = array<i32>, operandSegmentSizes = array<i32: 1, 0, 0>} : (!hir.any) -> ()
  func.return
}

// -----

// Verify that uir.call rejects mismatched results/typeOfResults.
func.func @call_type_of_results_mismatch(%a: !hir.any) {
  // expected-error @below {{failed to verify that typeOfResults must match results size}}
  %r = "uir.call"(%a, %a) {callee = @nowhere, argPhases = array<i32: 0>, resultPhases = array<i32: 0>, operandSegmentSizes = array<i32: 1, 1, 0>} : (!hir.any, !hir.any) -> (!hir.any)
  func.return
}

// -----

// Verify that uir.call rejects argument phases inconsistent with the callee.
uir.func @call_arg_phase_callee(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.return %b : %t
}

uir.func @call_arg_phase_mismatch(%a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{argument phases 0, 0 do not match the phases @call_arg_phase_callee declares for its arguments -1, 0}}
  %r = uir.call @call_arg_phase_callee(%a, %b) : (%t, %t) -> (%t) [0, 0] -> [0]
  uir.return %r : %t
}

// -----

// Verify that uir.call rejects result phases inconsistent with the callee.
uir.func @call_result_phase_callee(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.return %a : %t
}

uir.func @call_result_phase_mismatch(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{result phases 1 do not match the phases @call_result_phase_callee declares for its results 0}}
  %r = uir.call @call_result_phase_callee(%a) : (%t) -> (%t) [0] -> [1]
  uir.return %r : %t
}

// -----

// Verify that uir.continue rejects mismatched values/typeOfValues.
func.func @continue_mismatch(%a: !hir.any) {
  uir.loop (%x = %a : %a) {
    // expected-error @below {{failed to verify that typeOfValues must match values size}}
    "uir.continue"(%a, %a, %a) {operandSegmentSizes = array<i32: 2, 1>} : (!hir.any, !hir.any, !hir.any) -> ()
  }
  func.return
}

// -----

// Verify that uir.loop checks init count against init type operand count.
func.func @loop_init_type_mismatch(%a: !hir.any) {
  // expected-error @below {{init count must match init type operand count}}
  "uir.loop"(%a, %a, %a) ({
  ^bb0(%x: !hir.any, %y: !hir.any):
    uir.continue %x, %y : %a, %a
  }) {operandSegmentSizes = array<i32: 2, 1, 0>} : (!hir.any, !hir.any, !hir.any) -> ()
  func.return
}

// -----

// Verify that uir.loop checks body block argument count against init count.
func.func @loop_block_arg_mismatch(%a: !hir.any) {
  // expected-error @below {{body block argument count must match init count}}
  "uir.loop"(%a, %a, %a, %a) ({
  ^bb0(%x: !hir.any):
    uir.continue %x, %x : %a, %a
  }) {operandSegmentSizes = array<i32: 2, 2, 0>} : (!hir.any, !hir.any, !hir.any, !hir.any) -> ()
  func.return
}

// -----

// Verify that uir.continue checks carried value count against the loop's
// iteration argument count.
func.func @continue_arity_mismatch(%a: !hir.any) {
  uir.loop (%x = %a : %a) {
    // expected-error @below {{carried value count must match enclosing loop's iteration argument count}}
    uir.continue
  }
  func.return
}

// -----

// Verify that a uir.continue nested inside a uir.if is checked against the
// enclosing loop's iteration argument count.
func.func @continue_nested_arity_mismatch(%cond: !hir.any, %a: !hir.any) {
  uir.loop (%x = %a : %a) {
    uir.if %cond {
      // expected-error @below {{carried value count must match enclosing loop's iteration argument count}}
      uir.continue %x, %x : %a, %a
    }
    uir.continue %x : %a
  }
  func.return
}

// -----

// Verify that uir.yield checks its value count against the enclosing if's
// result count.
func.func @yield_arity_mismatch(%cond: !hir.any, %a: !hir.any, %ty: !hir.any) {
  %r = uir.if %cond : %ty {
    // expected-error @below {{provides 0 values but the enclosing 'uir.if' declares 1 results}}
    uir.yield
  } else {
    uir.yield %a : %ty
  }
  func.return
}

// -----

// Verify that uir.yield checks its value count against the enclosing expr's
// result count.
func.func @yield_expr_arity_mismatch(%a: !hir.any, %ty: !hir.any) {
  %r = uir.expr : %ty {
    // expected-error @below {{provides 2 values but the enclosing 'uir.expr' declares 1 results}}
    uir.yield %a, %a : %ty, %ty
  }
  func.return
}

// -----

// Verify that uir.break checks its value count against the target loop's
// result count.
func.func @break_arity_mismatch(%a: !hir.any, %ty: !hir.any) {
  %r = uir.loop : %ty {
    // expected-error @below {{provides 0 values but the target 'uir.loop' declares 1 results}}
    uir.break
  }
  func.return
}

// -----

// Verify that a uir.break nested inside a uir.if is checked against the target
// loop's result count.
func.func @break_nested_arity_mismatch(%cond: !hir.any, %a: !hir.any, %ty: !hir.any) {
  %r = uir.loop : %ty {
    uir.if %cond {
      // expected-error @below {{provides 2 values but the target 'uir.loop' declares 1 results}}
      uir.break %a, %a : %ty, %ty
    }
    uir.break %a : %ty
  }
  func.return
}

// -----

// Verify that uir.signature cannot appear in the function body.
uir.func @sig_in_body(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{must be nested in the signature region of the enclosing function}}
  uir.signature (%a) -> ()
}

// -----

// Verify that uir.signature cannot appear nested under control flow in the
// function body either.
uir.func @sig_nested_in_body(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.if %a {
    // expected-error @below {{must be nested in the signature region of the enclosing function}}
    uir.signature (%a) -> ()
  }
  uir.return
}

// -----

// Verify that uir.signature requires an enclosing function.
func.func @sig_outside_func(%c: !hir.any) {
  uir.if %c {
    // expected-error @below {{must be nested in the signature region of a 'uir.func' or 'uir.split_func'}}
    uir.signature () -> ()
  }
  func.return
}

// -----

// Verify that a uir.signature nested under control flow still checks its
// operand arity against the enclosing function.
uir.func @sig_nested_arity(%a: 0) -> () {
  uir.if %a {
    // expected-error @below {{has 0 argument types but parent function has 1 arguments}}
    uir.signature () -> ()
  }
  uir.signature (%a) -> ()
} {
  uir.return
}

// -----

// Verify that a uir.signature nested under control flow in a split_func
// signature region checks its operand arity against the enclosing function.
uir.split_func @split_sig_nested_arity(%a: 0) -> () {
  uir.if %a {
    // expected-error @below {{has 0 argument types but parent function has 1 arguments}}
    uir.signature () -> ()
  }
  uir.signature (%a) -> ()
} [
  @split_sig_nested_arity.0
]

// -----

// Verify that a uir.split_func rejects the state where an implementation
// reference and a phase list coexist.
// expected-error @below {{has both an implementation reference and a phase list}}
uir.split_func @impl_and_phases(%a: 0) -> () {
  uir.signature (%a) -> ()
} impl @impl_and_phases.impl [
  @impl_and_phases.0
]
uir.func @impl_and_phases.impl(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.return
}

// -----

// Verify that the phase list, which callers resolve their phase calls through,
// names functions that exist.
// expected-error @below {{'no_such_phase_func' does not reference a valid function}}
uir.split_func @dangling_phase_func(%a: 0) -> () {
  uir.signature (%a) -> ()
} [
  @no_such_phase_func
]

// -----

// Verify that the phase list lists every phase of the declared span: a caller
// asks for every span phase, so a short list leaves a legal phase call
// unresolvable.
hir.func private @missing_phase.0() -> () {
  hir.signature () -> ()
} {
  hir.return
}
// expected-error @below {{has 1 phase list entries but its declared span [0, 1] has 2 phases; the phase list maps the span's phases positionally, one entry each}}
uir.split_func @missing_phase(%a: 0) -> (r: 1) {
  uir.signature (%a) -> (%a)
} [
  @missing_phase.0
]

// -----

// Verify that the phase list holds no entry beyond the declared span: callers
// never ask for one, and pre-span phases run eagerly instead of being called.
hir.func private @outside_span.0() -> () {
  hir.signature () -> ()
} {
  hir.return
}
// expected-error @below {{has 3 phase list entries but its declared span [0, 1] has 2 phases; the phase list maps the span's phases positionally, one entry each}}
uir.split_func @outside_span(%a: 0) -> (r: 1) {
  uir.signature (%a) -> (%a)
} [
  @outside_span.0,
  @outside_span.0,
  @outside_span.0
]

// -----

// Verify that a uir.split_func implementation reference must resolve.
// expected-error @below {{'missing_impl' does not reference a valid function}}
uir.split_func @unresolved_impl(%a: 0) -> () {
  uir.signature (%a) -> ()
} impl @missing_impl []

// -----

// Verify that a uir.split_func implementation reference must be a uir.func.
func.func @not_a_func() {
  func.return
}
// expected-error @below {{'not_a_func' does not reference a 'uir.func'}}
uir.split_func @wrong_impl_kind(%a: 0) -> () {
  uir.signature (%a) -> ()
} impl @not_a_func []

// -----

// Verify that a uir.split_func and its implementation agree on the phases
// declared for the arguments.
// expected-error @below {{argument phases -1 do not match the phases @arg_phase_mismatch.impl declares for its arguments 0}}
uir.split_func @arg_phase_mismatch(%a: -1) -> () {
  uir.signature (%a) -> ()
} impl @arg_phase_mismatch.impl []
uir.func @arg_phase_mismatch.impl(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.return
}

// -----

// Verify that a uir.split_func and its implementation agree on the phases
// declared for the results.
// expected-error @below {{result phases 1 do not match the phases @result_phase_mismatch.impl declares for its results 0}}
uir.split_func @result_phase_mismatch(%a: 0) -> (r: 1) {
  uir.signature (%a) -> (%a)
} impl @result_phase_mismatch.impl []
uir.func @result_phase_mismatch.impl(%a: 0) -> (r: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.return %a : %a
}

// -----

// Verify that module-ness stays on the declaration: an implementation that
// kept the marker would lose it silently once splitting retires it.
// expected-error @below {{implementation @module_impl.impl is marked 'isModule'; module-ness belongs to the declaration, not the implementation}}
uir.split_func @module_impl(%a: 0) -> () {
  uir.signature (%a) -> ()
} impl @module_impl.impl []
uir.func @module_impl.impl(%a: 0) -> () attributes {isModule} {
  uir.signature (%a) -> ()
} {
  uir.return
}

// -----

// Verify that uir.phase requires a uir.func parent.
func.func @phase_wrong_parent() {
  // expected-error @below {{expects parent op 'uir.func'}}
  uir.phase 0 {
    uir.phase_return
  }
  func.return
}

// -----

// Verify that uir.phase rejects terminators other than uir.phase_return and
// uir.unreachable.
uir.func @phase_bad_terminator(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.phase 0 {
    // expected-error @below {{cannot terminate a 'uir.phase' body; use 'uir.phase_return' or 'uir.unreachable'}}
    uir.yield
  }
  uir.unreachable
}

// -----

// Verify that uir.return cannot appear inside a uir.phase, even nested.
uir.func @return_in_phase(%c: 0, %x: 0) -> () {
  uir.signature (%c, %x) -> ()
} {
  uir.phase 0 {
    uir.if %c {
      // expected-error @below {{cannot appear inside a 'uir.phase'}}
      uir.return
    }
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that a uir.phase without any uir.phase_return cannot have results.
uir.func @phase_results_without_return(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  // expected-error @below {{has 1 results but no 'uir.phase_return' to provide them}}
  %p = "uir.phase"() ({
    uir.unreachable
  }) {phase = 0 : si32} : () -> (!hir.any)
  uir.unreachable
}

// -----

// Verify that uir.phase_return requires a uir.phase ancestor.
func.func @phase_return_outside() {
  // expected-error @below {{must be nested inside a 'uir.phase'}}
  uir.phase_return
}

// -----

// Verify that all phase returns of a phase agree on the yield count.
uir.func @phase_yield_mismatch(%c: 0, %x: 0) -> () {
  uir.signature (%c, %x) -> ()
} {
  %p = uir.phase 0 {
    uir.if %c {
      uir.phase_return yield %x
    } else {
      uir.yield
    }
    // expected-error @below {{yields 0 values but enclosing 'uir.phase' has 1 results}}
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call requires a uir.phase ancestor: it is
// sorted-output vocabulary that names a specific callee phase and only ever
// legitimately appears inside a phase region.
uir.split_func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} []
uir.func @phase_call_no_phase_ancestor(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  // expected-error @below {{'uir.phase_call' op must be nested inside a 'uir.phase'}}
  %r = uir.phase_call @pc_callee[0] (%a) : (%a) -> (%a)
  uir.return %r : %a
}

// -----

// Verify that uir.phase_call rejects a non-existent callee.
uir.func @phase_call_bad_callee(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{'nonexistent' does not reference a valid function}}
    %r = uir.phase_call @nonexistent[0] (%a) : (%a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects callees that are not functions at all.
func.func private @not_a_uir_func() {
  func.return
}
uir.func @phase_call_wrong_callee_kind(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{'not_a_uir_func' does not reference a 'uir.split_func'}}
    %r = uir.phase_call @not_a_uir_func[0] (%a) : (%a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects a uir.func callee: a phase call names the
// callee's late-binding interface, never the implementation behind it.
uir.func @an_implementation(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}
uir.func @phase_call_implementation_callee(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{'an_implementation' does not reference a 'uir.split_func'}}
    %r = uir.phase_call @an_implementation[0] (%a) : (%a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects phases the callee does not expose.
uir.split_func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} []
uir.func @phase_call_bad_phase(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{callee @pc_callee has no phase 1 in its declared span [0, 0]}}
    %r = uir.phase_call @pc_callee[1] (%a) : (%a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call checks the argument count against the callee's
// arguments declared at the called phase.
uir.split_func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} []
uir.func @phase_call_arg_count(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{has 2 arguments but callee @pc_callee declares 1 arguments at phase 0}}
    %r = uir.phase_call @pc_callee[0] (%a, %a) : (%a, %a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call checks the result count against the callee's
// results declared at the called phase.
uir.split_func @pc_callee(%x: 0) -> (result: 0) {
  uir.signature (%x) -> (%x)
} []
uir.func @phase_call_result_count(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{has 0 results but callee @pc_callee declares 1 results at phase 0}}
    uir.phase_call @pc_callee[0] (%a) : (%a) -> ()
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that non-first phase entries require a context operand.
uir.split_func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} []
uir.func @phase_call_missing_context(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{requires a context operand: phase 0 is not the callee's first span phase}}
    %r = uir.phase_call @pc_callee2[0] (%a) : (%a) -> (%a)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that first phase entries reject a context operand.
uir.split_func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} []
uir.func @phase_call_spurious_context(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase -1 {
    // expected-error @below {{cannot have a context operand: phase -1 is the callee's first span phase}}
    %ctx = uir.phase_call @pc_callee2[-1] (%a) context %a : (%a) -> () context
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that non-last phase entries require a context result.
uir.split_func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} []
uir.func @phase_call_missing_context_out(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase -1 {
    // expected-error @below {{requires a context result: phase -1 is not the callee's last span phase}}
    uir.phase_call @pc_callee2[-1] (%a) : (%a) -> ()
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that last phase entries reject a context result.
uir.split_func @pc_callee2(%n: -1, %x: 0) -> (result: 0) {
  uir.signature (%n, %x) -> (%x)
} []
uir.func @phase_call_spurious_context_out(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{cannot have a context result: phase 0 is the callee's last span phase}}
    %r, %ctx = uir.phase_call @pc_callee2[0] (%a) context %a : (%a) -> (%a) context
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that a mid-span pass-through entry (a gap phase with no arguments or
// results) still requires a context operand.
uir.split_func @pc_gap_callee(%m: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} []
uir.func @phase_call_gap_missing_context(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase -1 {
    // expected-error @below {{requires a context operand: phase -1 is not the callee's first span phase}}
    %ctx = uir.phase_call @pc_gap_callee[-1] () : () -> () context
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects an entry phase below the declared span.
uir.split_func @pc_gap_callee(%m: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} []
uir.func @phase_call_below_span(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    // expected-error @below {{callee @pc_gap_callee has no phase -3 in its declared span [-2, 0]}}
    %ctx = uir.phase_call @pc_gap_callee[-3] () : () -> () context
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects mismatched arguments/typeOfArgs.
uir.func @phase_call_arg_type_mismatch(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.phase 0 {
    // expected-error @below {{has 1 arguments but 0 typeOfArgs}}
    "uir.phase_call"(%a) {callee = @pc_callee, calleePhase = 0 : si32, operandSegmentSizes = array<i32: 1, 0, 0, 0>, resultSegmentSizes = array<i32: 0, 0>} : (!hir.any) -> ()
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that uir.phase_call rejects mismatched results/typeOfResults.
uir.func @phase_call_result_type_mismatch(%a: 0) -> () {
  uir.signature (%a) -> ()
} {
  uir.phase 0 {
    // expected-error @below {{has 1 results but 0 typeOfResults}}
    %r = "uir.phase_call"() {callee = @pc_callee, calleePhase = 0 : si32, operandSegmentSizes = array<i32: 0, 0, 0, 0>, resultSegmentSizes = array<i32: 1, 0>} : () -> (!hir.any)
    uir.phase_return
  }
  uir.unreachable
}

// -----

// Verify that returns in a uir.func body must match the declared results.
uir.func @return_arity(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // expected-error @below {{provides 0 values but the enclosing function declares 1 results}}
  uir.return
}

// -----

// Verify that nested returns must match the declared results too.
uir.func @return_arity_nested(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.if %c {
    // expected-error @below {{provides 0 values but the enclosing function declares 1 results}}
    uir.return
  }
  uir.return %x : %t
}
