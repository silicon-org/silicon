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

// Note: uir.break and uir.continue require a uir.loop ancestor, which will be
// tested in the structured CF commit. For now we just verify the ops parse in
// the error tests below (errors.mlir).
