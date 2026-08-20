// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

// # Region Terminators
//
// Test roundtrip parsing for all UIR terminator ops. Since these are
// terminators, they need to appear as the last op in a block. We use
// func.func as the enclosing context.

// CHECK-LABEL: @yield_void
func.func @yield_void() {
  // CHECK: uir.yield
  uir.expr {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @yield_values
func.func @yield_values(%a: !hir.any, %a_ty: !hir.any, %b: !hir.any, %b_ty: !hir.any) {
  // CHECK: uir.yield %{{.*}}, %{{.*}} : %{{.*}}, %{{.*}}
  uir.expr : %a_ty, %b_ty {
    uir.yield %a, %b : %a_ty, %b_ty
  }
  func.return
}

// CHECK-LABEL: @return_void
func.func @return_void() {
  uir.return
}

// CHECK-LABEL: @return_values
func.func @return_values(%a: !hir.any, %a_ty: !hir.any) {
  uir.return %a : %a_ty
}

// CHECK-LABEL: @return_multiple
func.func @return_multiple(%a: !hir.any, %b: !hir.any, %a_ty: !hir.any, %b_ty: !hir.any) {
  uir.return %a, %b : %a_ty, %b_ty
}

// CHECK-LABEL: @unreachable
func.func @unreachable() {
  uir.unreachable
}

// # Structured Control Flow

// CHECK-LABEL: @if_no_results
func.func @if_no_results(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @if_else_no_results
func.func @if_else_no_results(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  } else {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @if_else_with_results
func.func @if_else_with_results(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %a_ty: !hir.any, %b_ty: !hir.any) {
  %r = uir.if %cond : %a_ty {
    uir.yield %a : %a_ty
  } else {
    uir.yield %b : %b_ty
  }
  func.return
}

// CHECK-LABEL: @if_early_return
func.func @if_early_return(%cond: !hir.any, %a: !hir.any, %ty: !hir.any) {
  uir.if %cond {
    uir.return %a : %ty
  } else {
    uir.return %a : %ty
  }
  uir.unreachable
}

// A loop with no carried values and no results.
// CHECK-LABEL: @loop_no_results
// CHECK: uir.loop {
func.func @loop_no_results() {
  uir.loop {
    uir.continue
  }
  func.return
}

// A loop with result type operands but no carried values.
// CHECK-LABEL: @loop_with_results
// CHECK: uir.loop : %{{.*}} {
func.func @loop_with_results(%val: !hir.any, %ty: !hir.any) {
  %r = uir.loop : %ty {
    uir.break %val : %ty
  }
  func.return
}

// A loop carrying values, with the continue nested inside a uir.if to exercise
// the carried-value arity check on a non-top-level continue.
// CHECK-LABEL: @loop_carried
// CHECK: uir.loop (%{{.*}} = %{{.*}}, %{{.*}} = %{{.*}} : %{{.*}}, %{{.*}}) : %{{.*}} {
func.func @loop_carried(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ta: !hir.any, %tb: !hir.any, %r_ty: !hir.any) {
  %r = uir.loop (%x = %a, %y = %b : %ta, %tb) : %r_ty {
    uir.if %cond {
      uir.continue %y, %x : %tb, %ta
    }
    uir.break %x : %r_ty
  }
  func.return
}

// CHECK-LABEL: @nested_if_in_loop
func.func @nested_if_in_loop(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) {
  %r = uir.loop : %ty {
    %v = uir.if %cond : %ty {
      uir.yield %a : %ty
    } else {
      uir.yield %b : %ty
    }
    uir.break %v : %ty
  }
  func.return
}

// # Phase Grouping

// CHECK-LABEL: @expr_floating
func.func @expr_floating(%a: !hir.any, %ty: !hir.any) {
  %r = uir.expr : %ty {
    uir.yield %a : %ty
  }
  func.return
}

// CHECK-LABEL: @expr_floating_void
func.func @expr_floating_void(%a: !hir.any) {
  uir.expr {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @expr_pinned
func.func @expr_pinned(%a: !hir.any, %ty: !hir.any) {
  %r = uir.expr pin : %ty {
    uir.yield %a : %ty
  }
  func.return
}

// CHECK-LABEL: @expr_pinned_const
func.func @expr_pinned_const(%a: !hir.any, %ty: !hir.any) {
  %r = uir.expr pin -1 : %ty {
    uir.yield %a : %ty
  }
  func.return
}

// CHECK-LABEL: @expr_pinned_dyn
func.func @expr_pinned_dyn(%a: !hir.any, %ty: !hir.any) {
  %r = uir.expr pin 1 : %ty {
    uir.yield %a : %ty
  }
  func.return
}

// CHECK-LABEL: @pin_single
func.func @pin_single(%val: !hir.any) {
  %r = uir.pin %val, 0
  func.return
}

// CHECK-LABEL: @pin_const
func.func @pin_const(%val: !hir.any) {
  %r = uir.pin %val, -1
  func.return
}

// CHECK-LABEL: @pin_multiple
func.func @pin_multiple(%a: !hir.any, %b: !hir.any) {
  %r1, %r2 = uir.pin %a, %b, 0
  func.return
}

// # Unified Functions

// CHECK-LABEL: uir.func @simple_func
uir.func @simple_func(%x: 0, %y: 0) -> (result: 0) {
  uir.signature (%x, %y) -> (%x)
} {
  uir.return %x : %x
}

// CHECK-LABEL: uir.func @const_arg
uir.func @const_arg(%N: -1, %x: 0) -> (result: 0) {
  uir.signature (%N, %x) -> (%x)
} {
  uir.return %x : %x
}

// CHECK-LABEL: uir.func @no_args
uir.func @no_args() -> () {
  uir.signature () -> ()
} {
  uir.return
}

// uir.signature is the signature region's return-like exit and may sit under
// structured control flow at any nesting depth, like uir.return in the body.
// CHECK-LABEL: uir.func @sig_under_cf
uir.func @sig_under_cf(%c: 0, %x: 0) -> (result: 0) {
  uir.if %c {
    uir.signature (%c, %x) -> (%x)
  }
  uir.signature (%c, %x) -> (%c)
} {
  uir.return %x : %x
}

// CHECK-LABEL: uir.func @with_module
uir.func @with_module(%x: 0) -> (result: 0) attributes {isModule} {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}

// Stub per-phase functions for the phase list below, which must name functions
// that exist.
hir.func private @split_witness.0() -> () {
  hir.signature () -> ()
} {
  hir.return
}
hir.func private @split_witness.1() -> () {
  hir.signature () -> ()
} {
  hir.return
}

// A uir.split_func in the split state maps each span phase to the function
// implementing it and holds no implementation reference.
// CHECK-LABEL: uir.split_func @split_witness
uir.split_func @split_witness(%a: -1, %b: 0) -> (result: 0) {
  uir.signature (%a, %b) -> (%b)
// CHECK: } [
// CHECK-NEXT: -1: @split_witness.0,
// CHECK-NEXT: 0: @split_witness.1
// CHECK-NEXT: ]
} [
  -1: @split_witness.0,
  0: @split_witness.1
]

// Module-ness is a property of the declaration and thus lives on the
// uir.split_func.
// CHECK-LABEL: uir.split_func @module_witness
// CHECK-SAME: attributes {isModule}
uir.split_func @module_witness(%a: 0) -> (result: 0) attributes {isModule} {
  uir.signature (%a) -> (%a)
} []

// A uir.split_func in the declared state has neither an implementation nor a
// phase list: the interface is known, but nothing has been compiled for it.
// CHECK-LABEL: uir.split_func @declared_func
// CHECK: } []
uir.split_func @declared_func(%a: -1, %b: 0) -> (result: 0) {
  uir.signature (%a, %b) -> (%b)
} []

// A uir.split_func in the sorted state points at the uir.func holding the
// implementation whose phases have not been split apart yet.
// CHECK-LABEL: uir.split_func @sorted_func
// CHECK: } impl @sorted_func.impl []
uir.split_func @sorted_func(%a: -1, %b: 0) -> (result: 0) {
  uir.signature (%a, %b) -> (%b)
} impl @sorted_func.impl []

// CHECK-LABEL: uir.func @sorted_func.impl
uir.func @sorted_func.impl(%a: -1, %b: 0) -> (result: 0) {
  uir.signature (%a, %b) -> (%b)
} {
  uir.return %b : %b
}

// CHECK-LABEL: uir.func @caller
uir.func @caller(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  %r = uir.call @simple_func(%a, %a) : (%a, %a) -> (%a) [0, 0] -> [0]
  uir.return %r : %a
}

// # Sorted Phases
//
// The output contract of phase sorting: function regions hold one uir.phase
// op per phase, values flow between phases as ordinary SSA values, function
// results ride on uir.phase_return values, and calls are decomposed into
// per-phase uir.phase_call ops chained through opaque context values.

// A body sorted into phases: phase -1 computes values (including a type
// value) consumed by phase 0, which delivers the function result; the
// original return is replaced by uir.unreachable.
// CHECK-LABEL: uir.func @sorted_phases
uir.func @sorted_phases(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %p:2 = uir.phase -1 {
    %prod = hir.mul %a, %a : %t
    uir.phase_return yield %prod, %t
  }
  uir.phase 0 {
    %sum = hir.add %p#0, %p#0 : %p#1
    uir.phase_return %sum : %p#1
  }
  uir.unreachable
}

// Signature regions sort into phases too; their phase returns carry no
// function results, only yields consumed by later phases or uir.signature.
// CHECK-LABEL: uir.func @sig_phases
uir.func @sig_phases(%a: -1) -> (result: 0) {
  %p = uir.phase -1 {
    %t = hir.int_type
    uir.phase_return yield %t
  }
  uir.signature (%p) -> (%p)
} {
  %t = hir.int_type
  uir.return %a : %t
}

// Signature and body regions follow the same rules: signature phases may
// use the same control flow and exit arrangements as body phases.
// CHECK-LABEL: uir.func @sig_phase_early_exit
uir.func @sig_phase_early_exit(%c: -1) -> (result: 0) {
  %p = uir.phase -1 {
    uir.if %c {
      %t = hir.int_type
      uir.phase_return yield %t
    } else {
      %t = hir.bool_type
      uir.phase_return yield %t
    }
    uir.unreachable
  }
  uir.signature (%p) -> (%p)
} {
  %t = hir.int_type
  uir.return %c : %t
}

// Early exits: phase returns may sit inside structured CF; when all paths
// exit early, the phase block ends in uir.unreachable. Different exits may
// yield different values.
// CHECK-LABEL: uir.func @phase_early_exit
uir.func @phase_early_exit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %p = uir.phase 0 {
    uir.if %c {
      uir.phase_return %x : %t yield %x
    } else {
      uir.phase_return %x : %t yield %c
    }
    uir.unreachable
  }
  uir.unreachable
}

// uir.unreachable_value stands in for a value a control-flow path proved is
// never produced; it carries the implied !hir.any result and prints no type.
// CHECK-LABEL: @unreachable_value
func.func @unreachable_value() {
  // CHECK: %{{.+}} = uir.unreachable_value
  %v = uir.unreachable_value
  return
}

// A phase return may terminate a loop body inside a phase.
// CHECK-LABEL: uir.func @phase_return_in_loop
uir.func @phase_return_in_loop(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.phase 0 {
    uir.loop {
      uir.phase_return %x : %t
    }
    uir.unreachable
  }
  uir.unreachable
}

// A call to a single-phase callee decomposes into one uir.phase_call with
// no context values. A uir.phase_call is sorted-output vocabulary and must
// live inside a uir.phase region.
// CHECK-LABEL: uir.func @phase_call_single
uir.func @phase_call_single(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  uir.phase 0 {
    %r = uir.phase_call @simple_func[0] (%a, %a) : (%a, %a) -> (%a)
    uir.phase_return %r : %a
  }
  uir.unreachable
}

// A call to a two-phase callee decomposes into chained uir.phase_calls: the
// first entry produces a context result (trailing `context` keyword), the
// last entry consumes it (`context` operand). The entries live in the
// caller phases where they execute.
// CHECK-LABEL: uir.func @phase_call_chain
uir.func @phase_call_chain(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ctx = uir.phase -1 {
    %c = uir.phase_call @const_arg[-1] (%n) : (%t) -> () context
    uir.phase_return yield %c
  }
  uir.phase 0 {
    %r = uir.phase_call @const_arg[0] (%x) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

// A uir.phase_call may target a callee that is already split.
// CHECK-LABEL: uir.func @phase_call_split_callee
uir.func @phase_call_split_callee(%a: -1) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  %ctx = uir.phase -1 {
    %c = uir.phase_call @split_witness[-1] (%a) : (%a) -> () context
    uir.phase_return yield %c
  }
  uir.phase 0 {
    %r = uir.phase_call @split_witness[0] (%a) context %ctx : (%a) -> (%a)
    uir.phase_return %r : %a
  }
  uir.unreachable
}

// A uir.phase_call satisfies its uir.phase ancestor requirement at any
// nesting depth: here it sits under structured control flow inside the phase.
// CHECK-LABEL: uir.func @phase_call_under_cf
uir.func @phase_call_under_cf(%c: 0, %a: 0) -> (result: 0) {
  uir.signature (%c, %a) -> (%a)
} {
  uir.phase 0 {
    uir.if %c {
      // CHECK: uir.phase_call @simple_func[0]
      %r = uir.phase_call @simple_func[0] (%a, %a) : (%a, %a) -> (%a)
      uir.phase_return %r : %a
    } else {
      uir.phase_return %a : %a
    }
    uir.unreachable
  }
  uir.unreachable
}

// A callee whose declared phases have a gap (arguments at -2 and 0, none at
// -1) exposes an internal span phase as a pass-through uir.phase_call: no
// arguments or results, threading only context.
// CHECK-LABEL: uir.func @gap_callee
uir.func @gap_callee(%a: -2, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = hir.add %a, %b : %t
  uir.return %r : %t
}

// CHECK-LABEL: uir.func @phase_call_gap
uir.func @phase_call_gap(%n: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ctx0 = uir.phase -2 {
    %c0 = uir.phase_call @gap_callee[-2] (%n) : (%t) -> () context
    uir.phase_return yield %c0
  }
  %ctx1 = uir.phase -1 {
    // CHECK: uir.phase_call @gap_callee[-1] () context %{{.+}} : () -> () context
    %c1 = uir.phase_call @gap_callee[-1] () context %ctx0 : () -> () context
    uir.phase_return yield %c1
  }
  uir.phase 0 {
    %r = uir.phase_call @gap_callee[0] (%x) context %ctx1 : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

// # Exit Terminators in Replicate/Pick Bodies
//
// Receiving-phase replicate/pick bodies may terminate in uir.return instead
// of hir.yield: the exit path of the producing phase reconstructs itself in
// the receiving phase, and pick results never materialize on such paths.

func.func @pick_with_return(%choice: !hir.any, %ty: !hir.any) {
  %r = hir.pick %choice : !hir.any [
    0 (%a): {
      uir.return %a : %ty
    }
    1 (%b): {
      hir.yield %b : !hir.any
    }
  ]
  return
}

func.func @replicate_with_return(%hits: !hir.any, %ty: !hir.any) {
  hir.replicate (%f) in %hits {
    uir.return %f : %ty
  }
  return
}
