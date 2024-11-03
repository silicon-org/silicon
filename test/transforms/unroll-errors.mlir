// RUN: silicon-opt --split-input-file %s
// RUN: ! silicon-opt -p unroll --split-input-file %s 2>&1 | FileCheck %s

si.module @UnassignedOutput {
  // CHECK: error: output `x` is unassigned
  si.output_decl "x" : !si.ref<i32>
}

// -----

si.module @UnassignedVar {
  %0 = si.var_decl "v" : !si.ref<i32>
  // CHECK: error: `v` is unassigned at this point
  %1 = si.deref %0 : !si.ref<i32>
  si.not %1 : i32
}

// -----

si.module @UnassignedTupleField {
  %0 = si.var_decl "v" : !si.ref<!si.tuple<[i32]>>
  // CHECK: error: `v.0` is unassigned at this point
  %1 = si.deref %0 : !si.ref<!si.tuple<[i32]>>
}

// -----

si.module @UnassignedNestedTupleField {
  %0 = si.var_decl "v" : !si.ref<!si.tuple<[i32, !si.tuple<[i32]>]>>
  %1 = si.tuple_get_ref %0, #builtin.int<1> : !si.ref<!si.tuple<[i32, !si.tuple<[i32]>]>> -> !si.ref<!si.tuple<[i32]>>
  %2 = si.tuple_get_ref %1, #builtin.int<0> : !si.ref<!si.tuple<[i32]>> -> !si.ref<i32>
  // CHECK: error: `v.1.0` is unassigned at this point
  %3 = si.deref %2 : !si.ref<i32>
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

si.module @DynamicIf {
  // CHECK: error: expression is not a constant
  %0 = si.input "x" : i1
  si.if %0 {
  } {
  }
}
