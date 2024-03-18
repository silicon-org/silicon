// RUN: silicon-opt -t codegen %s | FileCheck %s

// CHECK-LABEL: hw.module @Empty()
si.module @Empty {}

// CHECK-LABEL: hw.module @Foo
// CHECK-SAME: in %a : i16
// CHECK-SAME: in %b : i16
// CHECK-SAME: in %c : i1
// CHECK-SAME: out z : i42
si.module @Foo {
    %a = si.input "a" : i16
    %b = si.input "b" : i16
    %c = si.input "c" : i1
    si.output "z", %u : i42

    // CHECK: [[CLOCK:%.+]] = seq.to_clock %c
    // CHECK: [[REG:%.+]] = seq.compreg %c9001_i42, [[CLOCK]]
    %u = si.reg %c, %c9001 : i42

    // CHECK: hw.constant 9001 : i42
    %c9001 = si.constant 9001 : i42
    // CHECK: comb.sub %c0_i16, %a : i16
    si.neg %a : i16
    // CHECK: comb.xor %c-1_i16, %a : i16
    si.not %a : i16
    // CHECK: comb.add %a, %b : i16
    si.add %a, %b : i16
    // CHECK: comb.sub %a, %b : i16
    si.sub %a, %b : i16
    // CHECK: comb.concat %a, %b : i16, i16
    si.concat %a, %b : (i16, i16) -> i32
    // CHECK: comb.extract %a from 9 : (i16) -> i4
    si.extract %a, #builtin.int<9> : i16 -> i4
    // CHECK: comb.mux %c, %a, %b : i16
    si.mux %c, %a, %b : i16
    // CHECK: hw.output [[REG]]
}
