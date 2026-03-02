// RUN: silicon-opt --specialize-funcs %s | FileCheck %s

func.func private @use_value(%arg0: !hir.any)

// CHECK: mir.constant #mir.func<[[SIMPLE_SPEC:@Simple_.+]]> :
// CHECK: mir.constant #mir.func<[[SIMPLE_SPEC]]> :
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>

// CHECK-LABEL: hir.func @Simple
// CHECK: hir.func [[SIMPLE_SPEC]](%arg0) -> ()
hir.func @Simple(%arg0, %arg1) -> () {
  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<42>
  // CHECK-NEXT: [[ARG1:%.+]] = builtin.unrealized_conversion_cast [[TMP]] : !mir.int to !hir.any
  // CHECK-NEXT: func.call @use_value(%arg0) : (!hir.any) -> ()
  func.call @use_value(%arg0) : (!hir.any) -> ()
  // CHECK-NEXT: func.call @use_value([[ARG1]]) : (!hir.any) -> ()
  func.call @use_value(%arg1) : (!hir.any) -> ()
  // CHECK-NEXT: mir.return
  mir.return
}

// CHECK: mir.constant #mir.func<[[NESTED_OUTER_SPEC:@NestedOuter_.+]]> :
// CHECK: mir.constant #mir.func<[[NESTED_OUTER_SPEC]]> :
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>

// CHECK-LABEL: hir.func @NestedOuter
// CHECK: hir.func [[NESTED_OUTER_SPEC]]() -> ()
hir.func @NestedOuter(%arg0) -> () {
  // CHECK: mir.constant #mir.func<[[NESTED_INNER_SPEC:@NestedInner_.+]]> :
  mir.return
}

// CHECK-LABEL: hir.func @NestedInner
// CHECK: hir.func [[NESTED_INNER_SPEC]]() -> ()
hir.func @NestedInner(%arg0) -> () {
  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<1337>
  // CHECK-NEXT: [[ARG0:%.+]] = builtin.unrealized_conversion_cast [[TMP]] : !mir.int to !hir.any
  // CHECK-NEXT: func.call @use_value([[ARG0]]) : (!hir.any) -> ()
  func.call @use_value(%arg0) : (!hir.any) -> ()
  mir.return
}
