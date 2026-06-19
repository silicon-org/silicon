// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

// # Region Terminators
//
// Test roundtrip parsing for all UIR terminator ops. Since these are
// terminators, they need to appear as the last op in a block. We use
// func.func as the enclosing context.

// CHECK-LABEL: @yield_void
func.func @yield_void() {
  uir.yield
}

// CHECK-LABEL: @yield_values
func.func @yield_values(%a: !hir.any, %a_ty: !hir.any, %b: !hir.any, %b_ty: !hir.any) {
  uir.yield %a, %b : %a_ty, %b_ty
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
  %r = uir.pin %val, 0 : !hir.any
  func.return
}

// CHECK-LABEL: @pin_const
func.func @pin_const(%val: !hir.any) {
  %r = uir.pin %val, -1 : !hir.any
  func.return
}

// CHECK-LABEL: @pin_multiple
func.func @pin_multiple(%a: !hir.any, %b: !hir.any) {
  %r1, %r2 = uir.pin %a, %b, 0 : !hir.any, !hir.any
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

// CHECK-LABEL: uir.func @with_module
uir.func @with_module(%x: 0) -> (result: 0) attributes {isModule} {
  uir.signature (%x) -> (%x)
} {
  uir.return %x : %x
}

// CHECK-LABEL: uir.split_func @split_witness
uir.split_func @split_witness(%a: -1, %b: 0) -> (result: 0) {
  uir.signature (%a, %b) -> (%b)
} [
  -1: @split_witness.0,
  0: @split_witness.1
]

// CHECK-LABEL: uir.func @caller
uir.func @caller(%a: 0) -> (result: 0) {
  uir.signature (%a) -> (%a)
} {
  %r = uir.call @simple_func(%a, %a) : (%a, %a) -> (%a) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
  uir.return %r : %a
}
