// RUN: ! silicon-opt -p unroll --split-input-file %s 2>&1 | FileCheck %s

si.module @UnassignedOutput {
  // CHECK: error: output `x` has not been assigned
  si.output_decl "x" : i32
}

// -----

si.module @UnassignedVar {
  %v = si.var_decl "v" : i32
  // CHECK: error: `v` is unassigned at this point
  si.not %v : i32
}

// -----

si.module @BadAssignLHS {
    // CHECK: error: expression `{{.*}}` cannot appear on left-hand side of `=`
    %0 = si.constant 1 : i32
    si.assign %0, %0 : i32
}

// -----

// CHECK: warning: function `UnusedRoot` is never used
si.func @UnusedRoot() {
  si.call @UnusedInner() : () -> ()
  si.return
}

// CHECK: warning: function `UnusedInner` is never used
si.func @UnusedInner() {
  si.return
}

// -----

// Cannot assign to tuple fields at the moment.
si.module @AssignTupleField {
  %false = si.constant 0 : i1
  %v = si.var_decl "v" : !si.tuple<[i1, i1]>
  %0 = si.tuple_create %false, %false : (i1, i1) -> !si.tuple<[i1, i1]>
  si.assign %v, %0 : !si.tuple<[i1, i1]>
  %1 = si.tuple_get %v, #builtin.int<0> : !si.tuple<[i1, i1]> -> i1
  // CHECK: error: expression `` cannot appear on left-hand side of `=`
  si.assign %1, %false : i1
}
