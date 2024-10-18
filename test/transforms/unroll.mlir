// RUN: silicon-opt -p unroll %s | FileCheck %s

// CHECK-LABEL: si.module @Empty
si.module @Empty {
}

// CHECK-LABEL: si.module @Outputs
si.module @Outputs {
  // CHECK-NOT: si.output_decl
  // CHECK: si.output "x", [[Y:%.+]] : i32
  %x = si.output_decl "x" : !si.ref<i32>
  // CHECK: [[C0:%.+]] = si.constant 0 :
  %c0_i32 = si.constant 0 : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not %c0_i32
  si.assign %x, %c0_i32 : !si.ref<i32>
  %0 = si.deref %x : !si.ref<i32>
  si.not %0 : i32
  // CHECK: [[Y]] = si.input "y" : i32
  %y = si.input "y" : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[Y]]
  si.assign %x, %y : !si.ref<i32>
  %1 = si.deref %x : !si.ref<i32>
  si.not %1 : i32
}

// CHECK-LABEL: si.module @Vars
si.module @Vars {
  // CHECK: [[C0:%.+]] = si.constant 0 :
  // CHECK: [[C1:%.+]] = si.constant 1 :
  // CHECK: [[C2:%.+]] = si.constant 2 :
  %c0_i32 = si.constant 0 : i32
  %c1_i32 = si.constant 1 : i32
  %c2_i32 = si.constant 2 : i32

  // CHECK-NOT: si.var_decl "v0"
  %v0 = si.var_decl "v0" : !si.ref<i32>

  // CHECK-NOT: si.var_decl "v1"
  %v1 = si.var_decl "v1" : !si.ref<i32>
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C0]]
  si.assign %v1, %c0_i32 : !si.ref<i32>
  %0 = si.deref %v1 : !si.ref<i32>
  si.not %0 : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C1]]
  si.assign %v1, %c1_i32 : !si.ref<i32>
  %1 = si.deref %v1 : !si.ref<i32>
  si.not %1 : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C2]]
  si.assign %v1, %c2_i32 : !si.ref<i32>
  %2 = si.deref %v1 : !si.ref<i32>
  si.not %2 : i32
}

// CHECK-LABEL: si.module @Wires
si.module @Wires {
  // CHECK-NOT: si.wire_decl
  %unused = si.wire_decl : !si.wire<i32>

  // CHECK: [[C0:%.+]] = si.constant 0 : i32
  %c0_i32 = si.constant 0 : i32
  // CHECK-NOT: si.wire_decl
  // CHECK-NOT: si.wire_get
  %0 = si.wire_decl : !si.wire<i32>
  %1 = si.wire_decl : !si.wire<i32>
  %2 = si.wire_get %0 : !si.wire<i32>
  %3 = si.wire_get %1 : !si.wire<i32>
  // CHECK: si.not [[C0]]
  si.not %3 : i32
  // CHECK-NOT: si.wire_set
  si.wire_set %1, %2 : !si.wire<i32>
  si.wire_set %0, %c0_i32 : !si.wire<i32>
}

// CHECK-LABEL: si.module @Regs
si.module @Regs {
  // CHECK: [[CLOCK:%.+]] = si.input "clock"
  %clock = si.input "clock" : i1

  // CHECK-NOT: si.reg_decl
  %unused = si.reg_decl %clock : !si.reg<i32>

  // CHECK: [[C0:%.+]] = si.constant 0 : i32
  %c0_i32 = si.constant 0 : i32
  // CHECK-NOT: si.reg_decl
  // CHECK-NOT: si.reg_current
  // CHECK: [[REG0:%.+]] = si.reg [[CLOCK]], [[C0]] : i32
  // CHECK: [[REG1:%.+]] = si.reg [[CLOCK]], [[REG0]] : i32
  %0 = si.reg_decl %clock : !si.reg<i32>
  %1 = si.reg_decl %clock : !si.reg<i32>
  %2 = si.reg_current %0 : !si.reg<i32>
  %3 = si.reg_current %1 : !si.reg<i32>
  // CHECK: si.not [[REG1]]
  si.not %3 : i32
  // CHECK-NOT: si.reg_next
  si.reg_next %1, %2 : !si.reg<i32>
  si.reg_next %0, %c0_i32 : !si.reg<i32>
}

// CHECK-LABEL: si.module @Calls
si.module @Calls {
  // CHECK-NOT: si.call @EmptyFunc()
  si.call @EmptyFunc() : () -> ()

  // CHECK-NOT: si.call @SimpleFunc()
  // CHECK: [[TMP:%.+]] = si.constant 1337
  // CHECK-NOT: si.call @EmptyFunc()
  %0 = si.call @SimpleFunc() : () -> i32
  // CHECK: si.not [[TMP]] : i32
  si.not %0 : i32

  // CHECK: [[X:%.+]] = si.input "x" : i32
  // CHECK: [[Y:%.+]] = si.input "y" : i32
  %x = si.input "x" : i32
  %y = si.input "y" : i32
  // CHECK-NOT: si.call @ComplexFunc
  // CHECK-NOT: si.call @SimpleFunc()
  // CHECK: [[TMP1:%.+]] = si.constant 1337
  // CHECK-NOT: si.call @EmptyFunc()
  // CHECK: [[TMP2:%.+]] = si.add [[TMP1]], [[X]]
  // CHECK: [[TMP1:%.+]] = si.add [[Y]], [[TMP2]]
  // CHECK: [[TMP2:%.+]] = si.sub [[TMP1]], [[TMP]]
  %1 = si.call @ComplexFunc(%x, %y, %0) : (i32, i32, i32) -> i32
  // CHECK: si.not [[TMP2]] : i32
  si.not %1 : i32
}

// CHECK-NOT: si.func @EmptyFunc
si.func @EmptyFunc() {
  si.return
}

// CHECK-NOT: si.func @SimpleFunc
si.func @SimpleFunc() -> i32 {
  %0 = si.constant 1337 : i32
  si.call @EmptyFunc() : () -> ()
  si.return %0 : i32
}

// CHECK-NOT: si.func @ComplexFunc
si.func @ComplexFunc(%arg0 : i32, %arg1 : i32, %arg2 : i32) -> i32 {
  %0 = si.var_decl "x" : !si.ref<i32>
  %1 = si.call @SimpleFunc() : () -> i32
  si.assign %0, %1 : !si.ref<i32>
  %2 = si.deref %0 : !si.ref<i32>
  %3 = si.add %2, %arg0 : i32
  si.assign %0, %3 : !si.ref<i32>
  %4 = si.deref %0 : !si.ref<i32>
  %5 = si.add %arg1, %4 : i32
  si.assign %0, %5 : !si.ref<i32>
  %6 = si.deref %0 : !si.ref<i32>
  %7 = si.sub %6, %arg2 : i32
  si.return %7 : i32
}
