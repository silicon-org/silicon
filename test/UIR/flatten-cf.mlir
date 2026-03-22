// RUN: silicon-opt %s --flatten-cf | FileCheck %s

// # FlattenCF: uir.if lowering

// CHECK-LABEL: func.func @if_both_yield
// CHECK-SAME:    (%[[COND:.*]]: !hir.any, %[[A:.*]]: !hir.any, %[[B:.*]]: !hir.any, %[[TY:.*]]: !hir.any)
// CHECK:         %[[I1:.*]] = hir.coerce_to_i1 %[[COND]]
// CHECK-NEXT:    cf.cond_br %[[I1]], ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]](%[[A]] : !hir.any)
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE]](%[[B]] : !hir.any)
// CHECK:       ^[[MERGE]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @if_both_yield(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %cond : %ty {
    uir.yield %a : %ty
  } else {
    uir.yield %b : %ty
  }
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @if_no_else
// CHECK-SAME:    (%[[COND:.*]]: !hir.any)
// CHECK:         %[[I1:.*]] = hir.coerce_to_i1 %[[COND]]
// CHECK-NEXT:    cf.cond_br %[[I1]], ^[[THEN:.*]], ^[[CONT:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]]
// CHECK:       ^[[MERGE]]:
// CHECK-NEXT:    cf.br ^[[CONT]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    return
func.func @if_no_else(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: func.func @if_two_results
// CHECK-SAME:    (%[[COND:.*]]: !hir.any, %[[A:.*]]: !hir.any, %[[B:.*]]: !hir.any, %[[T1:.*]]: !hir.any, %[[T2:.*]]: !hir.any)
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]](%[[A]], %[[B]] : !hir.any, !hir.any)
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE]](%[[B]], %[[A]] : !hir.any, !hir.any)
// CHECK:       ^[[MERGE]](%[[R1:.*]]: !hir.any, %[[R2:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    return %[[R1]], %[[R2]] : !hir.any, !hir.any
func.func @if_two_results(%cond: !hir.any, %a: !hir.any, %b: !hir.any,
                           %ty1: !hir.any, %ty2: !hir.any) -> (!hir.any, !hir.any) {
  %r1, %r2 = uir.if %cond : %ty1, %ty2 {
    uir.yield %a, %b : %ty1, %ty2
  } else {
    uir.yield %b, %a : %ty1, %ty2
  }
  func.return %r1, %r2 : !hir.any, !hir.any
}

// CHECK-LABEL: func.func @sequential_ifs
// CHECK:         cf.cond_br %{{.*}}, ^[[T1:.*]], ^[[E1:.*]]
// CHECK:       ^[[T1]]:
// CHECK-NEXT:    cf.br ^[[M1:.*]](%{{.*}} : !hir.any)
// CHECK:       ^[[E1]]:
// CHECK-NEXT:    cf.br ^[[M1]](%{{.*}} : !hir.any)
// CHECK:       ^[[M1]](%[[V1:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[MID:.*]]
// CHECK:       ^[[MID]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[T2:.*]], ^[[E2:.*]]
// CHECK:       ^[[T2]]:
// CHECK-NEXT:    cf.br ^[[M2:.*]](%[[V1]] : !hir.any)
// CHECK:       ^[[E2]]:
// CHECK-NEXT:    cf.br ^[[M2]](%{{.*}} : !hir.any)
// CHECK:       ^[[M2]](%[[V2:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    return %[[V2]] : !hir.any
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

// CHECK-LABEL: func.func @if_void_yield_both
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]]
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE]]
// CHECK:       ^[[MERGE]]:
// CHECK-NEXT:    cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    return
func.func @if_void_yield_both(%cond: !hir.any) {
  uir.if %cond {
    uir.yield
  } else {
    uir.yield
  }
  func.return
}

// CHECK-LABEL: func.func @if_result_used_after
// CHECK:       ^[[MERGE:.*]](%[[R:.*]]: !hir.any):
// CHECK:         cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    %{{.*}} = hir.coerce_type %[[R]], %{{.*}}
func.func @if_result_used_after(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.if %cond : %ty {
    uir.yield %a : %ty
  } else {
    uir.yield %b : %ty
  }
  %s = hir.coerce_type %r, %ty
  func.return %s : !hir.any
}

// # FlattenCF: uir.loop lowering

// CHECK-LABEL: func.func @loop_break
// CHECK-SAME:    (%[[COND:.*]]: !hir.any, %[[VAL:.*]]: !hir.any, %[[TY:.*]]: !hir.any)
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BREAK:.*]], ^[[CONT_BODY:.*]]
// CHECK:       ^[[BREAK]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%[[VAL]] : !hir.any)
// CHECK:       ^[[CONT_BODY]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @loop_break(%cond: !hir.any, %val: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    uir.if %cond {
      uir.break %val : %ty
    }
    uir.yield
  }
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @loop_continue
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[CONT1:.*]], ^[[AFTER1:.*]]
// CHECK:       ^[[CONT1]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[AFTER1]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BREAK:.*]], ^[[AFTER2:.*]]
// CHECK:       ^[[BREAK]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]]
// CHECK:       ^[[AFTER2]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]]:
// CHECK-NEXT:    cf.br ^[[FINAL:.*]]
// CHECK:       ^[[FINAL]]:
// CHECK-NEXT:    return
func.func @loop_continue(%c1: !hir.any, %c2: !hir.any) {
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

// CHECK-LABEL: func.func @loop_void
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BREAK:.*]], ^[[BODY:.*]]
// CHECK:       ^[[BREAK]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]]
// CHECK:       ^[[BODY]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]]:
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return
func.func @loop_void(%cond: !hir.any) {
  uir.loop {
    uir.if %cond {
      uir.break
    }
    uir.yield
  }
  func.return
}

// CHECK-LABEL: func.func @if_in_loop
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[T:.*]], ^[[E:.*]]
// CHECK:       ^[[T]]:
// CHECK-NEXT:    cf.br ^[[IFMERGE:.*]](%{{.*}} : !hir.any)
// CHECK:       ^[[E]]:
// CHECK-NEXT:    cf.br ^[[IFMERGE]](%{{.*}} : !hir.any)
// CHECK:       ^[[IFMERGE]](%[[V:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[IFCONT:.*]]
// CHECK:       ^[[IFCONT]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%[[V]] : !hir.any)
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @if_in_loop(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    %v = uir.if %cond : %ty {
      uir.yield %a : %ty
    } else {
      uir.yield %b : %ty
    }
    uir.break %v : %ty
  }
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @nested_loops
// CHECK:         cf.br ^[[OUTER:.*]]
// CHECK:       ^[[OUTER]]:
// CHECK-NEXT:    cf.br ^[[INNER:.*]]
// CHECK:       ^[[INNER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[IBREAK:.*]], ^[[ICONT:.*]]
// CHECK:       ^[[IBREAK]]:
// CHECK-NEXT:    cf.br ^[[IEXIT:.*]]
// CHECK:       ^[[ICONT]]:
// CHECK-NEXT:    cf.br ^[[INNER]]
// CHECK:       ^[[IEXIT]]:
// CHECK-NEXT:    cf.br ^[[IMID:.*]]
// CHECK:       ^[[IMID]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[OBREAK:.*]], ^[[OCONT:.*]]
// CHECK:       ^[[OBREAK]]:
// CHECK-NEXT:    cf.br ^[[OEXIT:.*]]
// CHECK:       ^[[OCONT]]:
// CHECK-NEXT:    cf.br ^[[OUTER]]
// CHECK:       ^[[OEXIT]]:
// CHECK-NEXT:    cf.br ^[[FINAL:.*]]
// CHECK:       ^[[FINAL]]:
// CHECK-NEXT:    return
func.func @nested_loops(%c1: !hir.any, %c2: !hir.any) {
  uir.loop {
    uir.loop {
      uir.if %c2 {
        uir.break
      }
      uir.yield
    }
    uir.if %c1 {
      uir.break
    }
    uir.yield
  }
  func.return
}

// CHECK-LABEL: func.func @loop_result_used
// CHECK:       ^[[EXIT:.*]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @loop_result_used(%cond: !hir.any, %val: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    uir.if %cond {
      uir.break %val : %ty
    }
    uir.yield
  }
  func.return %r : !hir.any
}

// Verify no UIR ops survive.
// CHECK-NOT: uir.if
// CHECK-NOT: uir.loop
// CHECK-NOT: uir.yield
// CHECK-NOT: uir.break
// CHECK-NOT: uir.continue
// CHECK-NOT: uir.unreachable
