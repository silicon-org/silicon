// RUN: silicon-opt --canonicalize %s | FileCheck %s

// Constants must be foldable.
mir.constant #mir.int<42>

// CHECK-LABEL: @ConstSpecializeFunc
func.func @ConstSpecializeFunc() -> !mir.specialized_func {
  // CHECK-NEXT: mir.constant #mir.specialized_func<@foo, [!mir.int], [!mir.type], [#mir.int<42> : !mir.int]>
  // CHECK-NEXT: return
  %0 = mir.constant #mir.type<!mir.int>
  %1 = mir.constant #mir.type<!mir.type>
  %2 = mir.constant #mir.int<42>
  %3 = mir.specialize_func @foo(%0) -> (%1), %2 : !mir.int
  return %3 : !mir.specialized_func
}
