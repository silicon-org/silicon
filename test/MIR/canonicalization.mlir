// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK: mir.constant #si.int<42>
func.func @ConstFold() -> !si.int {
  %0 = mir.constant #si.int<42>
  return %0 : !si.int
}

// CHECK-LABEL: @BoolToI1True
// CHECK-NEXT: %[[TRUE:.*]] = mir.constant true
// CHECK-NEXT: return %[[TRUE]]
func.func @BoolToI1True() -> i1 {
  %0 = mir.constant #si.bool<true>
  %1 = mir.bool_to_i1 %0
  return %1 : i1
}

// CHECK-LABEL: @BoolToI1False
// CHECK-NEXT: %[[FALSE:.*]] = mir.constant false
// CHECK-NEXT: return %[[FALSE]]
func.func @BoolToI1False() -> i1 {
  %0 = mir.constant #si.bool<false>
  %1 = mir.bool_to_i1 %0
  return %1 : i1
}
