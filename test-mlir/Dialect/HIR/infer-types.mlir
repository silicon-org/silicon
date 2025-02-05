// RUN: silicon-opt --infer-types %s | FileCheck %s

func.func private @dummy(%arg0: !hir.type)

// CHECK-LABEL: func @TwoInferrable
func.func @TwoInferrable() {
  // CHECK: [[T:%.+]] = hir.inferrable_type {a}
  // CHECK-NOT: hir.inferrable_type {b}
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  %0 = hir.inferrable_type {a}
  %1 = hir.inferrable_type {b}
  %2 = hir.unify_type %0, %1
  call @dummy(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @TwoInferrableReversed
func.func @TwoInferrableReversed() {
  // CHECK: [[T:%.+]] = hir.inferrable_type {a}
  // CHECK-NOT: hir.inferrable_type {b}
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  %0 = hir.inferrable_type {a}
  %1 = hir.inferrable_type {b}
  %2 = hir.unify_type %1, %0  // reversed
  call @dummy(%2) : (!hir.type) -> ()
  return
}


// CHECK-LABEL: func @InferrableAndConcrete1
func.func @InferrableAndConcrete1() {
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK-NOT: hir.inferrable_type
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  %0 = hir.int_type
  %1 = hir.inferrable_type
  %2 = hir.unify_type %0, %1
  call @dummy(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @InferrableAndConcrete2
func.func @InferrableAndConcrete2() {
  // Cannot infer concrete type if it appears after the inferrable type.
  // CHECK: hir.inferrable_type
  // CHECK: hir.int_type
  %0 = hir.inferrable_type
  %1 = hir.int_type
  %2 = hir.unify_type %0, %1
  call @dummy(%2) : (!hir.type) -> ()
  return
}
