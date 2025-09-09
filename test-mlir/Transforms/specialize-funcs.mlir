// RUN: silicon-opt --specialize-funcs %s | FileCheck %s

func.func private @use_value(%arg0: !hir.value)

// CHECK: hir.constant_func [[SIMPLE_SPEC:@Simple_.+]]
// CHECK: hir.constant_func [[SIMPLE_SPEC]]
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>

// CHECK-LABEL: hir.func @Simple
// CHECK: hir.func [[SIMPLE_SPEC]]
hir.func @Simple {
// CHECK-NEXT: ^bb0(%arg0: !mir.int):
^bb0(%arg0: !hir.value, %arg1: !hir.value):
  // CHECK-NEXT: [[ARG0:%.+]] = builtin.unrealized_conversion_cast %arg0 : !mir.int to !hir.value
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant #mir.int<42>
  // CHECK-NEXT: [[ARG1:%.+]] = builtin.unrealized_conversion_cast [[TMP]] : !mir.int to !hir.value
  // CHECK-NEXT: func.call @use_value([[ARG0]]) : (!hir.value) -> ()
  func.call @use_value(%arg0) : (!hir.value) -> ()
  // CHECK-NEXT: func.call @use_value([[ARG1]]) : (!hir.value) -> ()
  func.call @use_value(%arg1) : (!hir.value) -> ()
  // CHECK-NEXT: mir.return
  mir.return
}

// CHECK: hir.constant_func [[NESTED_OUTER_SPEC:@NestedOuter_.+]]
// CHECK: hir.constant_func [[NESTED_OUTER_SPEC]]
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>

// CHECK-LABEL: hir.func @NestedOuter
// CHECK: hir.func [[NESTED_OUTER_SPEC]]
hir.func @NestedOuter {
^bb0(%arg0: !hir.func):
  // CHECK-NEXT: hir.constant_func [[NESTED_INNER_SPEC:@NestedInner_.+]]
  mir.return
}

// CHECK-LABEL: hir.func @NestedInner
// CHECK: hir.func [[NESTED_INNER_SPEC]]
hir.func @NestedInner {
^bb0(%arg0: !hir.value):
  // CHECK-NEXT: [[TMP:%.+]] = mir.constant #mir.int<1337>
  // CHECK-NEXT: [[ARG0:%.+]] = builtin.unrealized_conversion_cast [[TMP]] : !mir.int to !hir.value
  // CHECK-NEXT: func.call @use_value([[ARG0]]) : (!hir.value) -> ()
  func.call @use_value(%arg0) : (!hir.value) -> ()
  mir.return
}
