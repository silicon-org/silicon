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

// CHECK-LABEL: si.module @Tuples
si.module @Tuples {
  %c42_i42 = si.constant 42 : i42
  %c9001_i1337 = si.constant 9001 : i1337
  %c1337_i42 = si.constant 1337 : i42
  %x = si.var_decl "x" : !si.ref<!si.tuple<[i42, i1337]>>
  %x0 = si.tuple_get_ref %x, #builtin.int<0> : !si.ref<!si.tuple<[i42, i1337]>> -> !si.ref<i42>
  %x1 = si.tuple_get_ref %x, #builtin.int<1> : !si.ref<!si.tuple<[i42, i1337]>> -> !si.ref<i1337>

  // x.0 = 42; x.1 = 1337; read x.0, x.0, x
  // CHECK-NOT: si.assign
  si.assign %x0, %c42_i42 : !si.ref<i42>
  si.assign %x1, %c9001_i1337 : !si.ref<i1337>
  // CHECK-NOT: si.deref
  // CHECK: si.output "a0", %c42_i42
  %a0 = si.deref %x0 : !si.ref<i42>
  si.output "a0", %a0 : i42
  // CHECK-NOT: si.deref
  // CHECK: si.output "a1", %c9001_i1337
  %a1 = si.deref %x1 : !si.ref<i1337>
  si.output "a1", %a1 : i1337
  // CHECK-NOT: si.deref
  // CHECK: [[TMP:%.+]] = si.tuple_create %c42_i42, %c9001_i1337 :
  // CHECK: si.output "a2", [[TMP]]
  %a2 = si.deref %x : !si.ref<!si.tuple<[i42, i1337]>>
  si.output "a2", %a2 : !si.tuple<[i42, i1337]>

  // x = (42, 9001); read x.0, x.1
  // CHECK-NOT: si.assign
  // CHECK-NOT: si.tuple_create
  %0 = si.tuple_create %c42_i42, %c9001_i1337 : (i42, i1337) -> !si.tuple<[i42, i1337]>
  si.assign %x, %0 : !si.ref<!si.tuple<[i42, i1337]>>
  // CHECK-NOT: si.deref
  // CHECK: si.output "b0", %c42_i42
  %b0 = si.deref %x0 : !si.ref<i42>
  si.output "b0", %b0 : i42
  // CHECK-NOT: si.deref
  // CHECK: si.output "b1", %c9001_i1337
  %b1 = si.deref %x1 : !si.ref<i1337>
  si.output "b1", %b1 : i1337

  // x = (42, 9001); x.0 = 1337; read x.0, x.1
  // CHECK-NOT: si.assign
  si.assign %x, %0 : !si.ref<!si.tuple<[i42, i1337]>>
  si.assign %x0, %c1337_i42 : !si.ref<i42>
  // CHECK-NOT: si.deref
  // CHECK: si.output "c0", %c1337_i42
  %c0 = si.deref %x0 : !si.ref<i42>
  si.output "c0", %c0 : i42
  // CHECK-NOT: si.deref
  // CHECK: si.output "c1", %c9001_i1337
  %c1 = si.deref %x1 : !si.ref<i1337>
  si.output "c1", %c1 : i1337
  // CHECK-NOT: si.deref
  // CHECK: [[TMP:%.+]] = si.tuple_create %c1337_i42, %c9001_i1337
  // CHECK: si.output "c2", [[TMP]]
  %c2 = si.deref %x : !si.ref<!si.tuple<[i42, i1337]>>
  si.output "c2", %c2 : !si.tuple<[i42, i1337]>
}

// CHECK-LABEL: si.module @DerefRefInTuple
si.module @DerefRefInTuple {
  %c0_i0 = si.constant 0 : i0
  %c17_i15 = si.constant 17 : i15
  %c42_i15 = si.constant 42 : i15
  %c1337_i15 = si.constant 1337 : i15
  %c9001_i15 = si.constant 9001 : i15

  // let x = 42, y = 17
  // CHECK-NOT: si.var_decl
  %x = si.var_decl "x" : !si.ref<i15>
  %y = si.var_decl "y" : !si.ref<i15>
  // CHECK-NOT: si.assign
  si.assign %x, %c42_i15 : !si.ref<i15>
  si.assign %y, %c17_i15 : !si.ref<i15>

  // let z = (&x, 0)
  // CHECK-NOT: si.var_decl
  %z = si.var_decl "z" : !si.ref<!si.tuple<[!si.ref<i15>, i0]>>
  // CHECK-NOT: si.tuple_create
  %0 = si.tuple_create %x, %c0_i0 : (!si.ref<i15>, i0) -> !si.tuple<[!si.ref<i15>, i0]>
  // CHECK-NOT: si.assign
  si.assign %z, %0 : !si.ref<!si.tuple<[!si.ref<i15>, i0]>>

  // read *z.0, x
  // CHECK-NOT: si.tuple_get_ref
  %z.0 = si.tuple_get_ref %z, #builtin.int<0> : !si.ref<!si.tuple<[!si.ref<i15>, i0]>> -> !si.ref<!si.ref<i15>>
  // CHECK-NOT: si.deref
  // CHECK: si.output "a0", %c42_i15
  %1 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %2 = si.deref %1 : !si.ref<i15>
  si.output "a0", %2 : i15
  // CHECK-NOT: si.deref
  // CHECK: si.output "a1", %c42_i15
  %3 = si.deref %x : !si.ref<i15>
  si.output "a1", %3 : i15

  // *z.0 = 1337
  // CHECK-NOT: si.deref
  %4 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  // CHECK-NOT: si.assign
  si.assign %4, %c1337_i15 : !si.ref<i15>

  // read *z.0, x
  // CHECK-NOT: si.deref
  // CHECK: si.output "b0", %c1337_i15
  %5 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %6 = si.deref %5 : !si.ref<i15>
  si.output "b0", %6 : i15
  // CHECK-NOT: si.deref
  // CHECK: si.output "b1", %c1337_i15
  %7 = si.deref %x : !si.ref<i15>
  si.output "b1", %7 : i15

  // z.0 = &y
  // CHECK-NOT: si.assign
  si.assign %z.0, %y : !si.ref<!si.ref<i15>>

  // read *z.0, y
  // CHECK-NOT: si.deref
  // CHECK: si.output "c0", %c17_i15
  %8 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %9 = si.deref %8 : !si.ref<i15>
  si.output "c0", %9 : i15
  // CHECK-NOT: si.deref
  // CHECK: si.output "c1", %c17_i15
  %10 = si.deref %y : !si.ref<i15>
  si.output "c1", %10 : i15

  // *z.0 = 9001
  // CHECK-NOT: si.deref
  %11 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  // CHECK-NOT: si.assign
  si.assign %11, %c9001_i15 : !si.ref<i15>

  // read *z.0, y
  // CHECK-NOT: si.deref
  // CHECK: si.output "d0", %c9001_i15
  %12 = si.deref %z.0 : !si.ref<!si.ref<i15>>
  %13 = si.deref %12 : !si.ref<i15>
  si.output "d0", %13 : i15
  // CHECK-NOT: si.deref
  // CHECK: si.output "d1", %c9001_i15
  %14 = si.deref %y : !si.ref<i15>
  si.output "d1", %14 : i15

  // CHECK-NOT: si.tuple_get
  // CHECK: si.output "e", %c0_i0
  %15 = si.tuple_get %0, #builtin.int<1> : !si.tuple<[!si.ref<i15>, i0]> -> i0
  si.output "e", %15 : i0
}

// CHECK-LABEL: si.module @ConstIfs
si.module @ConstIfs {
  %false = si.constant false
  %true = si.constant true
  // CHECK: [[TMP1:%.+]] = si.constant 1337 :
  // CHECK: [[TMP2:%.+]] = si.constant 42 :
  %1 = si.call @ConstIfsFunc(%false) : (i1) -> i32
  %2 = si.call @ConstIfsFunc(%true) : (i1) -> i32
  // CHECK: si.output "a0", [[TMP1]] :
  // CHECK: si.output "a1", [[TMP2]] :
  si.output "a0", %1 : i32
  si.output "a1", %2 : i32
}

si.func @ConstIfsFunc(%a : i1) -> i32 {
  %x = si.var_decl "x" : !si.ref<i32>
  si.if %a {
    %0 = si.constant 42 : i32
    si.assign %x, %0 : !si.ref<i32>
  } {
    %1 = si.constant 1337 : i32
    si.assign %x, %1 : !si.ref<i32>
  }
  %2 = si.deref %x : !si.ref<i32>
  si.return %2 : i32
}
