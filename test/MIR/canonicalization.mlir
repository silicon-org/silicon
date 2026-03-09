// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK: mir.constant #si.int<42>
func.func @ConstFold() -> !si.int {
  %0 = mir.constant #si.int<42>
  return %0 : !si.int
}

// CHECK-LABEL: @FoldAdd
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<7>
// CHECK-NEXT: return %[[C]]
func.func @FoldAdd() -> !si.int {
  %0 = mir.constant #si.int<3>
  %1 = mir.constant #si.int<4>
  %2 = mir.add %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldSub
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<-1>
// CHECK-NEXT: return %[[C]]
func.func @FoldSub() -> !si.int {
  %0 = mir.constant #si.int<3>
  %1 = mir.constant #si.int<4>
  %2 = mir.sub %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldMul
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<12>
// CHECK-NEXT: return %[[C]]
func.func @FoldMul() -> !si.int {
  %0 = mir.constant #si.int<3>
  %1 = mir.constant #si.int<4>
  %2 = mir.mul %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldDiv
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<3>
// CHECK-NEXT: return %[[C]]
func.func @FoldDiv() -> !si.int {
  %0 = mir.constant #si.int<7>
  %1 = mir.constant #si.int<2>
  %2 = mir.div %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldAnd
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<4>
// CHECK-NEXT: return %[[C]]
func.func @FoldAnd() -> !si.int {
  %0 = mir.constant #si.int<5>
  %1 = mir.constant #si.int<6>
  %2 = mir.and %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldOr
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<7>
// CHECK-NEXT: return %[[C]]
func.func @FoldOr() -> !si.int {
  %0 = mir.constant #si.int<5>
  %1 = mir.constant #si.int<6>
  %2 = mir.or %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldShl
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.int<20>
// CHECK-NEXT: return %[[C]]
func.func @FoldShl() -> !si.int {
  %0 = mir.constant #si.int<5>
  %1 = mir.constant #si.int<2>
  %2 = mir.shl %0, %1 : !si.int
  return %2 : !si.int
}

// CHECK-LABEL: @FoldEq
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.bool<false>
// CHECK-NEXT: return %[[C]]
func.func @FoldEq() -> !si.bool {
  %0 = mir.constant #si.int<3>
  %1 = mir.constant #si.int<4>
  %2 = mir.eq %0, %1 : !si.int
  return %2 : !si.bool
}

// CHECK-LABEL: @FoldLt
// CHECK-NEXT: %[[C:.*]] = mir.constant #si.bool<true>
// CHECK-NEXT: return %[[C]]
func.func @FoldLt() -> !si.bool {
  %0 = mir.constant #si.int<3>
  %1 = mir.constant #si.int<4>
  %2 = mir.lt %0, %1 : !si.int
  return %2 : !si.bool
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
