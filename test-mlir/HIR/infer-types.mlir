// RUN: silicon-opt --infer-types %s | FileCheck %s

func.func private @use_type(%arg0: !hir.type)
func.func private @use_value(%arg0: !hir.value)

// CHECK-LABEL: func @TwoInferrable
func.func @TwoInferrable() {
  // CHECK: [[T:%.+]] = hir.inferrable {a} : !hir.type
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.inferrable {a} : !hir.type
  %1 = hir.inferrable {b} : !hir.type
  %2 = hir.unify %0, %1 : !hir.type
  call @use_type(%0) : (!hir.type) -> ()
  call @use_type(%1) : (!hir.type) -> ()
  call @use_type(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @TwoInferrableReversed
func.func @TwoInferrableReversed() {
  // CHECK: [[T:%.+]] = hir.inferrable {a} : !hir.type
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.inferrable {a} : !hir.type
  %1 = hir.inferrable {b} : !hir.type
  %2 = hir.unify %1, %0 : !hir.type  // reversed
  call @use_type(%0) : (!hir.type) -> ()
  call @use_type(%1) : (!hir.type) -> ()
  call @use_type(%2) : (!hir.type) -> ()
  return
}


// CHECK-LABEL: func @InferrableAndConcrete1
func.func @InferrableAndConcrete1() {
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.int_type
  %1 = hir.inferrable : !hir.type
  %2 = hir.unify %0, %1 : !hir.type
  call @use_type(%0) : (!hir.type) -> ()
  call @use_type(%1) : (!hir.type) -> ()
  call @use_type(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @InferrableAndConcrete2
func.func @InferrableAndConcrete2() {
  // Cannot infer concrete type if it appears after the inferrable type.
  // CHECK: hir.inferrable : !hir.type
  // CHECK: hir.int_type
  %0 = hir.inferrable : !hir.type
  %1 = hir.int_type
  %2 = hir.unify %0, %1 : !hir.type
  call @use_type(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOps1
func.func @UnifyConcreteOps1() {
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[REF:%.+]] = hir.ref_type [[INT]]
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  %0 = hir.int_type
  %1 = hir.inferrable : !hir.type
  %2 = hir.ref_type %0
  %3 = hir.ref_type %1
  %4 = hir.unify %2, %3 : !hir.type
  call @use_type(%2) : (!hir.type) -> ()
  call @use_type(%3) : (!hir.type) -> ()
  call @use_type(%4) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOps2
func.func @UnifyConcreteOps2() {
  // Cannot unify concrete types if the operands of don't dominate both types.
  // CHECK: hir.int_type
  // CHECK: hir.ref_type
  // CHECK: hir.inferrable : !hir.type
  // CHECK: hir.ref_type
  %0 = hir.int_type
  %1 = hir.ref_type %0
  %2 = hir.inferrable : !hir.type
  %3 = hir.ref_type %2
  %4 = hir.unify %1, %3 : !hir.type
  call @use_type(%4) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @InferIfDominates
func.func @InferIfDominates(%arg0: i1) {
  // CHECK: [[TMP:%.+]] = hir.int_type
  %0 = hir.int_type
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  // CHECK: ^bb3:
  // CHECK-NEXT: call @use_type([[TMP]])
  %1 = hir.inferrable : !hir.type
  %2 = hir.unify %0, %1 : !hir.type
  call @use_type(%2) : (!hir.type) -> ()
  // CHECK-NEXT: call @use_type([[TMP]])
  %3 = hir.int_type
  %4 = hir.unify %0, %3 : !hir.type
  call @use_type(%4) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @InferConstantInt
func.func @InferConstantInt() {
  // CHECK: [[TMP:%.+]] = hir.constant_int 42
  // CHECK-NEXT: call @use_value([[TMP]])
  // CHECK-NEXT: call @use_value([[TMP]])
  // CHECK-NEXT: call @use_value([[TMP]])
  %0 = hir.constant_int 42
  %1 = hir.inferrable : !hir.value
  %2 = hir.unify %0, %1 : !hir.value
  call @use_value(%0) : (!hir.value) -> ()
  call @use_value(%1) : (!hir.value) -> ()
  call @use_value(%2) : (!hir.value) -> ()
  return
}
