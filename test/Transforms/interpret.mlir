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

//===----------------------------------------------------------------------===//
// Phase chaining via split_func: the interpret pass evaluates @Chain.const2
// (zero-arg, already lowered to MIR), chains its result into @Chain.const1
// via the split_func phase map, then evaluates @Chain.const1. The
// multiphase_func is collapsed once its sub-functions are evaluated.

// CHECK-LABEL: hir.func private @Chain.const2
// CHECK-NEXT: [[C42:%.+]] = mir.constant #mir.int<42>
// CHECK-NEXT: mir.return [[C42]]
hir.func private @Chain.const2 {
  %0 = mir.constant #mir.int<42> : !mir.int
  mir.return %0 : !mir.int
}

// CHECK-LABEL: hir.func private @Chain.const1
// CHECK-NEXT: [[C52:%.+]] = mir.constant #mir.int<52>
// CHECK-NEXT: mir.return [[C52]]
hir.func private @Chain.const1 {
^bb0(%ctx: !mir.int):
  %0 = mir.constant #mir.int<10> : !mir.int
  %1 = mir.binary %ctx, %0 : !mir.int
  mir.return %1 : !mir.int
}

// The split_func provides the phase map that drives chaining.
// CHECK-LABEL: hir.split_func @Chain
hir.split_func @Chain() -> () {
  hir.signature () -> ()
} [
  -2: @Chain.const2,
  -1: @Chain.const1
]

// CHECK-NOT: hir.multiphase_func @Chain.const
hir.multiphase_func @Chain.const() -> (ctx) [
  @Chain.const2,
  @Chain.const1
]
