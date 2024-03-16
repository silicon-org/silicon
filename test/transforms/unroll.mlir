// RUN: silicon-opt -p unroll %s | FileCheck %s

// CHECK-LABEL: si.module @Empty
si.module @Empty {
}

// CHECK-LABEL: si.module @Vars
si.module @Vars {
  // CHECK: [[C0:%.+]] = si.constant 0 :
  // CHECK: [[C1:%.+]] = si.constant 1 :
  // CHECK: [[C2:%.+]] = si.constant 2 :
  %c0_i32 = si.constant 0 : i32 : i32
  %c1_i32 = si.constant 1 : i32 : i32
  %c2_i32 = si.constant 2 : i32 : i32

  // CHECK-NOT: si.var_decl "v0"
  %v0 = si.var_decl "v0" : i32

  // CHECK-NOT: si.var_decl "v1"
  %v1 = si.var_decl "v1" : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C0]]
  si.assign %v1, %c0_i32 : i32
  si.not %v1 : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C1]]
  si.assign %v1, %c1_i32 : i32
  si.not %v1 : i32
  // CHECK-NOT: si.assign
  // CHECK: si.not [[C2]]
  si.assign %v1, %c2_i32 : i32
  si.not %v1 : i32
}

// CHECK-LABEL: si.module @Wires
si.module @Wires {
  // CHECK-NOT: si.wire_decl
  %unused = si.wire_decl : !si.wire<i32>

  // CHECK: [[C0:%.+]] = si.constant 0 : i32
  %c0_i32 = si.constant 0 : i32 : i32
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
  // CHECK: [[CLOCK:%.+]] = si.input_decl "clock"
  %clock = si.input_decl "clock" : i1

  // CHECK-NOT: si.reg_decl
  %unused = si.reg_decl %clock : !si.reg<i32>

  // CHECK: [[C0:%.+]] = si.constant 0 : i32
  %c0_i32 = si.constant 0 : i32 : i32
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
