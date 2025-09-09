// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: hir.func @foo
hir.func @foo {
  // CHECK: [[SPEC:%.+]] = mir.constant #mir.specialized_func<@foo, [!mir.int], [], [#mir.int<42> : !mir.int]>
  // CHECK: mir.return [[SPEC]]
  %0 = hir.int_type
  %1 = hir.constant_int 42
  %2 = hir.specialize_func @foo(%0) -> (), %1 : !hir.value
  hir.return %2 : !hir.func
}
