// RUN: silicon-opt %s --flatten-cf | FileCheck %s

//===----------------------------------------------------------------------===//
// uir.if lowering
//===----------------------------------------------------------------------===//

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

//===----------------------------------------------------------------------===//
// uir.loop lowering
//===----------------------------------------------------------------------===//

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
    uir.continue
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
    uir.continue
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
    uir.continue
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
      uir.continue
    }
    uir.if %c1 {
      uir.break
    }
    uir.continue
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
    uir.continue
  }
  func.return %r : !hir.any
}

//===----------------------------------------------------------------------===//
// loop with carried values
//===----------------------------------------------------------------------===//

// The loop-carried iteration arguments become block arguments of the loop
// header. The entry branch passes the init values into the header, and each
// uir.continue branches back to the header with the next-iteration values.
// The break exits with the loop's result.
// CHECK-LABEL: func.func @loop_carried
// CHECK:         cf.br ^[[HEADER:.*]](%[[A:.*]], %[[B:.*]] : !hir.any, !hir.any)
// CHECK:       ^[[HEADER]](%[[X:.*]]: !hir.any, %[[Y:.*]]: !hir.any):
// CHECK:         cf.cond_br %{{.*}}, ^[[BREAK:.*]], ^[[CONT:.*]]
// CHECK:       ^[[BREAK]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%[[X]] : !hir.any)
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    cf.br ^[[HEADER]](%[[Y]], %[[X]] : !hir.any, !hir.any)
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @loop_carried(%cond: !hir.any, %a: !hir.any, %b: !hir.any,
                        %ta: !hir.any, %tb: !hir.any, %r_ty: !hir.any) -> !hir.any {
  %r = uir.loop (%x = %a, %y = %b : %ta, %tb) : %r_ty {
    uir.if %cond {
      uir.break %x : %r_ty
    }
    uir.continue %y, %x : %tb, %ta
  }
  func.return %r : !hir.any
}

//===----------------------------------------------------------------------===//
// uir.return lowering
//===----------------------------------------------------------------------===//

// uir.return inside structured CF becomes hir.return. We use hir.func here
// since hir.return requires its parent to be hir.func or hir.unified_func.

// CHECK-LABEL: hir.func @if_both_return
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    hir.return
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    hir.return
hir.func @if_both_return(%cond, %a, %ty) -> (result) {
  hir.signature (%cond, %a, %ty) -> (%ty)
} {
  uir.if %cond {
    uir.return %a -> (%ty)
  } else {
    uir.return %a -> (%ty)
  }
  uir.unreachable
}

// CHECK-LABEL: hir.func @if_one_return_one_yield
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    hir.return
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]](%{{.*}} : !hir.any)
// CHECK:       ^[[MERGE]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[CONT:.*]]
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    hir.return %[[R]] -> (%{{.*}})
hir.func @if_one_return_one_yield(%cond, %a, %b, %ty) -> (result) {
  hir.signature (%cond, %a, %b, %ty) -> (%ty)
} {
  %r = uir.if %cond : %ty {
    uir.return %a -> (%ty)
  } else {
    uir.yield %b : %ty
  }
  hir.return %r -> (%ty)
}

// CHECK-LABEL: hir.func @return_in_loop
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[IFTHEN:.*]], ^[[IFCONT:.*]]
// CHECK:       ^[[IFTHEN]]:
// CHECK-NEXT:    hir.return
// CHECK:       ^[[IFCONT]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
hir.func @return_in_loop(%cond, %val, %ty) -> (result) {
  hir.signature (%cond, %val, %ty) -> (%ty)
} {
  uir.loop {
    uir.if %cond {
      uir.return %val -> (%ty)
    }
    uir.continue
  }
  hir.return %val -> (%ty)
}

//===----------------------------------------------------------------------===//
// if with mixed yield/break inside loop
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @if_yield_and_break_in_loop
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]]({{.*}} : !hir.any)
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]]
// CHECK:       ^[[MERGE]]:
// CHECK-NEXT:    cf.br ^[[IFCONT:.*]]
// CHECK:       ^[[IFCONT]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @if_yield_and_break_in_loop(%cond: !hir.any, %val: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    uir.if %cond {
      uir.break %val : %ty
    } else {
      uir.yield
    }
    uir.continue
  }
  func.return %r : !hir.any
}

//===----------------------------------------------------------------------===//
// multiple break ops from different branches
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @multiple_breaks
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[THEN:.*]], ^[[ELSE:.*]]
// CHECK:       ^[[THEN]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%[[A:.*]] : !hir.any)
// CHECK:       ^[[ELSE]]:
// CHECK-NEXT:    cf.br ^[[EXIT]](%[[B:.*]] : !hir.any)
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @multiple_breaks(%cond: !hir.any, %a: !hir.any, %b: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    uir.if %cond {
      uir.break %a : %ty
    } else {
      uir.break %b : %ty
    }
    uir.unreachable
  }
  func.return %r : !hir.any
}

//===----------------------------------------------------------------------===//
// loop with multiple results
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @loop_multiple_results
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BREAK:.*]], ^[[CONT:.*]]
// CHECK:       ^[[BREAK]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%{{.*}}, %{{.*}} : !hir.any, !hir.any)
// CHECK:       ^[[CONT]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]](%[[R1:.*]]: !hir.any, %[[R2:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R1]], %[[R2]] : !hir.any, !hir.any
func.func @loop_multiple_results(%cond: !hir.any, %a: !hir.any, %b: !hir.any,
                                  %ty1: !hir.any, %ty2: !hir.any) -> (!hir.any, !hir.any) {
  %r1, %r2 = uir.loop : %ty1, %ty2 {
    uir.if %cond {
      uir.break %a, %b : %ty1, %ty2
    }
    uir.continue
  }
  func.return %r1, %r2 : !hir.any, !hir.any
}

//===----------------------------------------------------------------------===//
// deeply nested loop > if > loop > if > break
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @deeply_nested
// CHECK:         cf.br ^[[OUTER:.*]]
// CHECK:       ^[[OUTER]]:
// CHECK:         cf.cond_br
// CHECK:         cf.br ^[[INNER:.*]]
// CHECK:       ^[[INNER]]:
// CHECK:         cf.cond_br
func.func @deeply_nested(%c1: !hir.any, %c2: !hir.any) {
  uir.loop {
    uir.if %c1 {
      uir.loop {
        uir.if %c2 {
          uir.break
        }
        uir.continue
      }
      uir.yield
    }
    uir.continue
  }
  func.return
}

//===----------------------------------------------------------------------===//
// immediate-exit loop (body is just uir.break)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @immediate_break_loop
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]](%{{.*}} : !hir.any)
// CHECK:       ^[[EXIT]](%[[R:.*]]: !hir.any):
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return %[[R]] : !hir.any
func.func @immediate_break_loop(%val: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.loop : %ty {
    uir.break %val : %ty
  }
  func.return %r : !hir.any
}

//===----------------------------------------------------------------------===//
// infinite loop (just uir.continue, no break)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @infinite_loop
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
func.func @infinite_loop() {
  uir.loop {
    uir.continue
  }
  func.return
}

//===----------------------------------------------------------------------===//
// while-loop desugaring pattern
//===----------------------------------------------------------------------===//

// The then-branch yields from the if (continue body), and the else-branch
// breaks out of the loop. After the if, uir.continue advances to the next
// iteration (only reachable from the then-path through the merge block).

// CHECK-LABEL: func.func @while_loop_pattern
// CHECK:         cf.br ^[[HEADER:.*]]
// CHECK:       ^[[HEADER]]:
// CHECK:         cf.cond_br %{{.*}}, ^[[BODY:.*]], ^[[BREAKPATH:.*]]
// CHECK:       ^[[BODY]]:
// CHECK-NEXT:    cf.br ^[[MERGE:.*]]
// CHECK:       ^[[BREAKPATH]]:
// CHECK-NEXT:    cf.br ^[[EXIT:.*]]
// CHECK:       ^[[MERGE]]:
// CHECK-NEXT:    cf.br ^[[LOOPYIELD:.*]]
// CHECK:       ^[[LOOPYIELD]]:
// CHECK-NEXT:    cf.br ^[[HEADER]]
// CHECK:       ^[[EXIT]]:
// CHECK-NEXT:    cf.br ^[[AFTER:.*]]
// CHECK:       ^[[AFTER]]:
// CHECK-NEXT:    return
func.func @while_loop_pattern(%cond: !hir.any) {
  uir.loop {
    uir.if %cond {
      uir.yield
    } else {
      uir.break
    }
    uir.continue
  }
  func.return
}

//===----------------------------------------------------------------------===//
// uir.expr and uir.pin dissolved
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @expr_dissolved
// CHECK-NOT:     uir.expr
// CHECK:         return %{{.*}} : !hir.any
func.func @expr_dissolved(%a: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.expr : %ty {
    uir.yield %a : %ty
  }
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @expr_pinned_dissolved
// CHECK-NOT:     uir.expr
// CHECK:         return %{{.*}} : !hir.any
func.func @expr_pinned_dissolved(%a: !hir.any, %ty: !hir.any) -> !hir.any {
  %r = uir.expr pin -1 : %ty {
    uir.yield %a : %ty
  }
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @pin_dissolved
// CHECK-NOT:     uir.pin
// CHECK:         return %{{.*}} : !hir.any
func.func @pin_dissolved(%a: !hir.any) -> !hir.any {
  %r = uir.pin %a, 0 : !hir.any
  func.return %r : !hir.any
}

// CHECK-LABEL: func.func @pin_multi_dissolved
// CHECK-NOT:     uir.pin
// CHECK:         return %{{.*}}, %{{.*}} : !hir.any, !hir.any
func.func @pin_multi_dissolved(%a: !hir.any, %b: !hir.any) -> (!hir.any, !hir.any) {
  %r1, %r2 = uir.pin %a, %b, -1 : !hir.any, !hir.any
  func.return %r1, %r2 : !hir.any, !hir.any
}

// Verify no UIR ops survive.
// CHECK-NOT: uir.if
// CHECK-NOT: uir.loop
// CHECK-NOT: uir.yield
// CHECK-NOT: uir.break
// CHECK-NOT: uir.continue
// CHECK-NOT: uir.unreachable
// CHECK-NOT: uir.expr
// CHECK-NOT: uir.pin
