// RUN: ! silicon-opt -p unroll --split-input-file --verify-diagnostics %s 2>&1 | FileCheck %s

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
