// RUN: silicon-opt --interpret %s | FileCheck %s

// CHECK-LABEL: hir.func @foo
hir.func @foo {
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant #mir.specialized_func<@foo, [], [], [#mir.specialized_func<@bar, [!mir.int], [], []> : !mir.specialized_func]>
  // CHECK-NEXT: mir.return [[TMP]]
  %0 = mir.call @bar() : () -> !mir.specialized_func
  %1 = mir.specialize_func @foo() -> (), %0 : !mir.specialized_func
  mir.return %1 : !mir.specialized_func
}

// CHECK-LABEL: hir.func @bar
hir.func @bar {
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant #mir.specialized_func<@bar, [!mir.int], [], []>
  // CHECK-NEXT: mir.return [[TMP]]
  %0 = mir.constant #mir.specialized_func<@bar, [!mir.int], [], []> : !mir.specialized_func
  mir.return %0 : !mir.specialized_func
}
