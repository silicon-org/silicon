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
