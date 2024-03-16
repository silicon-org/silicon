// RUN: silicon-opt %s | silicon-opt | FileCheck %s

// CHECK-LABEL: si.module @Foo
si.module @Foo {
  // CHECK: [[X:%.+]] = si.input "x" : i42
  %x = si.input "x" : i42
  // CHECK: si.output "u", [[X]] : i42
  si.output "u", %x : i42
  // CHECK: [[Y:%.+]] = si.output_decl "y" : i42
  %y = si.output_decl "y" : i42
  // CHECK: [[Z:%.+]] = si.var_decl "z" : i42
  %z = si.var_decl "z" : i42

  // CHECK: si.assign [[Z]], [[Z]] : i42
  si.assign %z, %z : i42

  // CHECK: [[C0:%.+]] = si.constant false
  %c0_i1 = si.constant false : i1

  // CHECK: [[A:%.+]] = si.wire_decl : !si.wire<i42>
  %a = si.wire_decl : !si.wire<i42>
  // CHECK: [[TMP:%.+]] = si.wire_get [[A]] : !si.wire<i42>
  %0 = si.wire_get %a : !si.wire<i42>
  // CHECK: si.wire_set [[A]], [[TMP]] : !si.wire<i42>
  si.wire_set %a, %0 : !si.wire<i42>

  // CHECK: [[B:%.+]] = si.reg_decl [[C0]] : !si.reg<i42>
  %b = si.reg_decl %c0_i1 : !si.reg<i42>
  // CHECK: [[TMP:%.+]] = si.reg_current [[B]] : !si.reg<i42>
  %1 = si.reg_current %b : !si.reg<i42>
  // CHECK: si.reg_next [[B]], [[TMP]] : !si.reg<i42>
  si.reg_next %b, %1 : !si.reg<i42>

  // CHECK: si.neg [[Z]] : i42
  si.neg %z : i42
  // CHECK: si.not [[Z]] : i42
  si.not %z : i42

  // CHECK: si.add [[X]], [[Z]] : i42
  si.add %x, %z : i42
  // CHECK: si.sub [[X]], [[Z]] : i42
  si.sub %x, %z : i42
  // CHECK: si.concat [[C0]] : (i1) -> i1
  si.concat %c0_i1 : (i1) -> i1
  // CHECK: si.concat [[C0]], [[X]] : (i1, i42) -> i43
  si.concat %c0_i1, %x : (i1, i42) -> i43
  // CHECK: si.concat [[C0]], [[X]], [[C0]] : (i1, i42, i1) -> i44
  si.concat %c0_i1, %x, %c0_i1 : (i1, i42, i1) -> i44
  // CHECK: si.extract [[Z]], #builtin.int<9> : i42 -> i3
  si.extract %z, #builtin.int<9> : i42 -> i3
  // CHECK: si.mux [[C0]], [[X]], [[Z]] : i42
  si.mux %c0_i1, %x, %z : i42
}
