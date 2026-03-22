// RUN: silicon-opt %s --flatten-cf | FileCheck %s

// # FlattenCF: uir.if lowering

// Both branches yield — merge block has one arg per value result.
// CHECK-LABEL: func.func @if_both_yield
// CHECK:         hir.coerce_to_i1
// CHECK-NEXT:    cf.cond_br
// CHECK:         cf.br ^[[MERGE:.*]](%{{.*}} : !hir.any)
// CHECK:         cf.br ^[[MERGE]](%{{.*}} : !hir.any)
// CHECK:       ^[[MERGE]](%{{.*}}: !hir.any):
func.func @if_both_yield(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %cond : %ty {
    uir.yield %a : %ty
  } else {
    uir.yield %b : %ty
  }
  func.return %r : !hir.any
}

// If without else — else branch goes to continuation.
// CHECK-LABEL: func.func @if_no_else
// CHECK:         hir.coerce_to_i1
// CHECK-NEXT:    cf.cond_br
func.func @if_no_else(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  }
  func.return
}

// Nested if (2 levels) — inner if lowered first (post-order).
// CHECK-LABEL: func.func @nested_if
// CHECK:         hir.coerce_to_i1
// CHECK:         cf.cond_br
// CHECK:         hir.coerce_to_i1
// CHECK:         cf.cond_br
func.func @nested_if(%c1: !hir.any, %c2: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %c1 : %ty {
    %inner = uir.if %c2 : %ty {
      uir.yield %a : %ty
    } else {
      uir.yield %b : %ty
    }
    uir.yield %inner : %ty
  } else {
    uir.yield %a : %ty
  }
  func.return %r : !hir.any
}

// Deeply nested if (3 levels) — tests post-order lowering of inner-to-outer.
// CHECK-LABEL: func.func @deeply_nested_if
// CHECK:         cf.cond_br
// CHECK:         cf.cond_br
// CHECK:         cf.cond_br
func.func @deeply_nested_if(%c1: !hir.any, %c2: !hir.any, %c3: !hir.any,
                             %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %c1 : %ty {
    %mid = uir.if %c2 : %ty {
      %inner = uir.if %c3 : %ty {
        uir.yield %a : %ty
      } else {
        uir.yield %b : %ty
      }
      uir.yield %inner : %ty
    } else {
      uir.yield %b : %ty
    }
    uir.yield %mid : %ty
  } else {
    uir.yield %a : %ty
  }
  func.return %r : !hir.any
}

// If with two results — merge block has 2 block args (values only, not types).
// CHECK-LABEL: func.func @if_two_results
// CHECK:         cf.br ^[[MERGE:.*]](%{{.*}}, %{{.*}} : !hir.any, !hir.any)
// CHECK:       ^[[MERGE]](%{{.*}}: !hir.any, %{{.*}}: !hir.any):
func.func @if_two_results(%cond: !hir.any, %a: !hir.any, %b: !hir.any,
                           %ty1: !hir.any, %ty2: !hir.any) -> (!hir.any, !hir.any) {
  %r1, %r2 = uir.if %cond : %ty1, %ty2 {
    uir.yield %a, %b : %ty1, %ty2
  } else {
    uir.yield %b, %a : %ty1, %ty2
  }
  func.return %r1, %r2 : !hir.any, !hir.any
}

// Two sequential ifs in the same block.
// CHECK-LABEL: func.func @sequential_ifs
// CHECK:         cf.cond_br
// CHECK:         cf.cond_br
func.func @sequential_ifs(%c1: !hir.any, %c2: !hir.any,
                           %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r1 = uir.if %c1 : %ty {
    uir.yield %a : %ty
  } else {
    uir.yield %b : %ty
  }
  %r2 = uir.if %c2 : %ty {
    uir.yield %r1 : %ty
  } else {
    uir.yield %b : %ty
  }
  func.return %r2 : !hir.any
}

// Both branches yield void — merge block with no block args.
// CHECK-LABEL: func.func @if_void_yield_both
// CHECK:         cf.cond_br {{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]]
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE]]
// CHECK:       ^[[MERGE]]:
func.func @if_void_yield_both(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  } else {
    uir.yield
  }
  func.return
}

// Operations after if use the if's results — merge block args replace SSA uses.
// CHECK-LABEL: func.func @if_result_used_after
// CHECK:       ^[[MERGE:.*]](%[[R:.*]]: !hir.any):
// CHECK:         hir.coerce_type %[[R]],
func.func @if_result_used_after(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %cond : %ty {
    uir.yield %a : %ty
  } else {
    uir.yield %b : %ty
  }
  %s = hir.coerce_type %r, %ty
  func.return %s : !hir.any
}

// Note: a uir.if where one branch has uir.unreachable but is actually
// reachable is invalid IR. The pass reports it as a compiler bug.
// That case is tested in flatten-cf-errors.mlir.

// Verify no UIR ops survive.
// CHECK-NOT: uir.if
// CHECK-NOT: uir.yield
// CHECK-NOT: uir.unreachable
