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
  uir.return -> ()
}

// CHECK-LABEL: @return_values
func.func @return_values(%a: !hir.any, %a_ty: !hir.any) {
  uir.return %a -> (%a_ty)
}

// CHECK-LABEL: @return_multiple
func.func @return_multiple(%a: !hir.any, %b: !hir.any, %a_ty: !hir.any, %b_ty: !hir.any) {
  uir.return %a, %b -> (%a_ty, %b_ty)
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
    uir.return %a -> (%ty)
  } else {
    uir.return %a -> (%ty)
  }
  uir.unreachable
}

// CHECK-LABEL: @loop_no_results
func.func @loop_no_results(%cond: !hir.any) {
  uir.loop {
    uir.if %cond {
      uir.break
    }
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @loop_with_results
func.func @loop_with_results(%cond: !hir.any, %val: !hir.any, %ty: !hir.any) {
  %r = uir.loop : %ty {
    uir.if %cond {
      uir.break %val : %ty
    }
    uir.yield
  }
  func.return
}

// CHECK-LABEL: @loop_with_continue
func.func @loop_with_continue(%c1: !hir.any, %c2: !hir.any) {
  uir.loop {
    uir.if %c1 {
      uir.continue
    }
    uir.if %c2 {
      uir.break
    }
    uir.yield
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
