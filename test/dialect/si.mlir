// RUN: silicon-opt %s | silicon-opt | FileCheck %s

// CHECK-LABEL: si.func @Types
si.func @Types(
  // CHECK-SAME: %arg0 : !si.unit
  %arg0 : !si.unit,
  // CHECK-SAME: %arg1 : !si.ref<i42>
  %arg1 : !si.ref<i42>,
  // CHECK-SAME: %arg2 : !si.tuple<[i42, i1337]>
  %arg2 : !si.tuple<[i42, i1337]>
) {
  si.return
}

// CHECK-LABEL: si.module @Foo
si.module @Foo {
  // CHECK: [[W:%.+]] = si.input "w" : i42
  %w = si.input "w" : i42
  // CHECK: [[X:%.+]] = si.input "x" : i42
  %x = si.input "x" : i42
  // CHECK: si.output "u", [[X]] : i42
  si.output "u", %x : i42
  // CHECK: [[Y:%.+]] = si.output_decl "y" : !si.ref<i42>
  %y = si.output_decl "y" : !si.ref<i42>
  // CHECK: [[Z:%.+]] = si.var_decl "z" : !si.ref<i42>
  %z = si.var_decl "z" : !si.ref<i42>

  // CHECK: si.assign [[Z]], [[X]] : !si.ref<i42>
  si.assign %z, %x : !si.ref<i42>

  // CHECK: [[C0:%.+]] = si.constant false
  %c0_i1 = si.constant false
  // CHECK: si.constant_unit : !si.unit
  %unit = si.constant_unit : !si.unit

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

  // CHECK: si.neg [[X]] : i42
  si.neg %x : i42
  // CHECK: si.not [[X]] : i42
  si.not %x : i42

  // CHECK: si.add [[X]], [[W]] : i42
  si.add %x, %w : i42
  // CHECK: si.sub [[X]], [[W]] : i42
  si.sub %x, %w : i42
  // CHECK: si.concat [[C0]] : (i1) -> i1
  si.concat %c0_i1 : (i1) -> i1
  // CHECK: si.concat [[C0]], [[X]] : (i1, i42) -> i43
  si.concat %c0_i1, %x : (i1, i42) -> i43
  // CHECK: si.concat [[C0]], [[X]], [[C0]] : (i1, i42, i1) -> i44
  si.concat %c0_i1, %x, %c0_i1 : (i1, i42, i1) -> i44
  // CHECK: si.extract [[W]], #builtin.int<9> : i42 -> i3
  si.extract %w, #builtin.int<9> : i42 -> i3
  // CHECK: si.mux [[C0]], [[X]], [[W]] : i42
  si.mux %c0_i1, %x, %w : i42

  // CHECK: si.call @f1() : () -> ()
  si.call @f1() : () -> ()
  // CHECK: si.call @f2() : () -> i42
  si.call @f2() : () -> i42
  // CHECK: si.call @f3([[X]]) : (i42) -> ()
  si.call @f3(%x) : (i42) -> ()
  // CHECK: si.call @f4([[X]], [[W]]) : (i42, i42) -> i42
  si.call @f4(%x, %w) : (i42, i42) -> i42
}

// CHECK-LABEL: si.func @f1
// CHECK: () {
si.func @f1() {
  // CHECK: si.return
  si.return
}

// CHECK-LABEL: si.func @f2
// CHECK: () -> i42 {
si.func @f2() -> i42 {
  %43 = si.constant 0 : i42
  // CHECK: si.return {{%.+}} : i42
  si.return %43 : i42
}

// CHECK-LABEL: si.func @f3
// CHECK: ({{%.+}} : i42) {
si.func @f3(%x : i42) {
  // CHECK: si.return
  si.return
}

// CHECK-LABEL: si.func @f4
// CHECK: ([[TMP:%.+]] : i42, {{%.+}} : i42) -> i42 {
si.func @f4(%0 : i42, %1 : i42) -> i42 {
  // CHECK: si.return [[TMP]] : i42
  si.return %0 : i42
}

// CHECK-LABEL: si.func @Tuples
si.func @Tuples(%a : i42, %b : i1, %t1 : !si.ref<!si.tuple<[i42, i1]>>) {
  // CHECK: [[T0:%.+]] = si.tuple_create %a, %b : (i42, i1) -> !si.tuple<[i42, i1]>
  %t0 = si.tuple_create %a, %b : (i42, i1) -> !si.tuple<[i42, i1]>

  // CHECK: si.tuple_get [[T0]], #builtin.int<0> : !si.tuple<[i42, i1]> -> i42
  si.tuple_get %t0, #builtin.int<0> : !si.tuple<[i42, i1]> -> i42
  // CHECK: si.tuple_get [[T0]], #builtin.int<1> : !si.tuple<[i42, i1]> -> i1
  si.tuple_get %t0, #builtin.int<1> : !si.tuple<[i42, i1]> -> i1

  // CHECK: si.tuple_get_ref %t1, #builtin.int<0> : !si.ref<!si.tuple<[i42, i1]>> -> !si.ref<i42>
  si.tuple_get_ref %t1, #builtin.int<0> : !si.ref<!si.tuple<[i42, i1]>> -> !si.ref<i42>
  // CHECK: si.tuple_get_ref %t1, #builtin.int<1> : !si.ref<!si.tuple<[i42, i1]>> -> !si.ref<i1>
  si.tuple_get_ref %t1, #builtin.int<1> : !si.ref<!si.tuple<[i42, i1]>> -> !si.ref<i1>

  si.return
}

// CHECK-LABEL: si.func @Refs
si.func @Refs(%a : i42, %b : !si.ref<i42>) {
  si.ref %a : !si.ref<i42>
  si.deref %b : !si.ref<i42>
  si.return
}

// CHECK-LABEL: si.func @Ifs
si.func @Ifs(%a : i1) {
  // CHECK: si.if %a {
  // CHECK:   si.constant true
  // CHECK: } {
  // CHECK:   si.constant false
  // CHECK: }
  si.if %a {
    si.constant true
  } {
    si.constant false
  }
  si.return
}
