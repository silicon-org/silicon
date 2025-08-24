// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: hir.func @foo
hir.func @foo {
  // CHECK: [[INT_TYPE:%.+]] = mir.constant #mir.type<!mir.int>
  %0 = hir.int_type
  // CHECK: [[C42_INT:%.+]] = mir.constant #mir.int<42>
  %1 = hir.constant_int 42
  // CHECK: hir.return args([[INT_TYPE]], [[C42_INT]] : !mir.type, !mir.int)
  hir.return args(%0, %1 : !hir.type, !hir.value)
}
