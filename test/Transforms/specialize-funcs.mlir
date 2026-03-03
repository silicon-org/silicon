// RUN: silicon-opt --specialize-funcs %s | FileCheck %s

// CHECK: mir.constant #mir.func<[[SIMPLE_SPEC:@Simple_.+]]> :
// CHECK: mir.constant #mir.func<[[SIMPLE_SPEC]]> :
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>
mir.constant #mir.specialized_func<@Simple, [!mir.int], [], [#mir.int<42>]>

// CHECK-LABEL: mir.func @Simple
// CHECK: mir.func [[SIMPLE_SPEC]](%arg0: !mir.int) -> (result: !mir.int)
mir.func @Simple(%arg0: !mir.int, %arg1: !mir.int) -> (result: !mir.int) {
  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<42>
  // CHECK-NEXT: [[RESULT:%.+]] = mir.binary %arg0, [[TMP]]
  %0 = mir.binary %arg0, %arg1 : !mir.int
  // CHECK-NEXT: mir.return [[RESULT]]
  mir.return %0 : !mir.int
}

// CHECK: mir.constant #mir.func<[[NESTED_OUTER_SPEC:@NestedOuter_.+]]> :
// CHECK: mir.constant #mir.func<[[NESTED_OUTER_SPEC]]> :
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>
mir.constant #mir.specialized_func<@NestedOuter, [], [], [#mir.specialized_func<@NestedInner, [], [], [#mir.int<1337>]>]>

// CHECK-LABEL: mir.func @NestedOuter
// CHECK: mir.func [[NESTED_OUTER_SPEC]]() -> ()
mir.func @NestedOuter(%arg0: !mir.specialized_func) -> () {
  // CHECK: mir.constant #mir.func<[[NESTED_INNER_SPEC:@NestedInner_.+]]> :
  mir.return
}

// CHECK-LABEL: mir.func @NestedInner
// CHECK: mir.func [[NESTED_INNER_SPEC]]() -> (result: !mir.int)
mir.func @NestedInner(%arg0: !mir.int) -> (result: !mir.int) {
  // CHECK: [[TMP:%.+]] = mir.constant #mir.int<1337>
  // CHECK-NEXT: mir.return [[TMP]]
  mir.return %arg0 : !mir.int
}
