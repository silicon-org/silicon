// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: hir.func @foo
hir.func @foo {
  // CHECK-NEXT: [[TYPE:%.+]] = mir.constant #mir.type<!mir.int>
  %0 = hir.int_type
  // CHECK-NEXT: [[VALUE:%.+]] = mir.constant #mir.int<42>
  %1 = hir.constant_int 42
  // CHECK-NEXT: [[FUNC:%.+]] = mir.constant #mir.func<@bar>
  %2 = hir.constant_func @bar
  // CHECK: [[SPEC:%.+]] = mir.specialize_func @foo([[TYPE]]) -> (), [[VALUE]], [[FUNC]]
  %3 = hir.specialize_func @foo(%0) -> (), %1, %2 : !hir.value, !hir.func
  // CHECK: mir.return [[SPEC]]
  hir.return %3 : !hir.func
}

// CHECK-LABEL: hir.func @Cast
hir.func @Cast {
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant
  %0 = mir.constant #mir.int<42>
  %1 = builtin.unrealized_conversion_cast %0 : !mir.int to !hir.value
  // CHECK-NEXT: mir.return [[TMP]]
  hir.return %1 : !hir.value
}
