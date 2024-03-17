// RUN: ! silicon-opt -t codegen --split-input-file %s 2>&1 | FileCheck %s

si.module @Foo {
    // CHECK: info: previous definition of port `x` was here
    si.input "x" : i32
    // CHECK: error: port `x` already defined
    si.input "x" : i32
}

// -----

si.module @Foo {
    %c0_i42 = si.constant 0 : i42 : i42
    // CHECK: info: previous definition of port `x` was here
    si.output "x", %c0_i42 : i42
    // CHECK: error: port `x` already defined
    si.output "x", %c0_i42 : i42
}

// -----

si.module @Foo {
    %c0_i42 = si.constant 0 : i42 : i42
    // CHECK: info: previous definition of port `x` was here
    si.input "x" : i42
    // CHECK: error: port `x` already defined
    si.output "x", %c0_i42 : i42
}
