// RUN: silicon-opt --infer-types %s | FileCheck %s

func.func private @dummy(%arg0: !hir.type)

// CHECK-LABEL: func @TwoInferrable
func.func @TwoInferrable() {
  // CHECK: [[T:%.+]] = hir.inferrable_type {a}
  // CHECK-NOT: hir.inferrable_type {b}
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  %0 = hir.inferrable_type {a}
  %1 = hir.inferrable_type {b}
  %2 = hir.unify_type %0, %1
  call @dummy(%0) : (!hir.type) -> ()
  call @dummy(%1) : (!hir.type) -> ()
  call @dummy(%2) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @TwoInferrableReversed
func.func @TwoInferrableReversed() {
  // CHECK: [[T:%.+]] = hir.inferrable_type {a}
  // CHECK-NOT: hir.inferrable_type {b}
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  %0 = hir.inferrable_type {a}
  %1 = hir.inferrable_type {b}
  %2 = hir.unify_type %1, %0  // reversed
  call @dummy(%0) : (!hir.type) -> ()
  call @dummy(%1) : (!hir.type) -> ()
  call @dummy(%2) : (!hir.type) -> ()
  return
}


// CHECK-LABEL: func @InferrableAndConcrete1
func.func @InferrableAndConcrete1() {
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK-NOT: hir.inferrable_type
  // CHECK-NOT: hir.unify_type
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  // CHECK: call @dummy([[T]])
  %0 = hir.int_type
  %1 = hir.inferrable_type
  %2 = hir.unify_type %0, %1
  call @dummy(%0) : (!hir.type) -> ()
  call @dummy(%1) : (!hir.type) -> ()
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

// CHECK-LABEL: func @UnifyConcreteOps1
func.func @UnifyConcreteOps1() {
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[REF:%.+]] = hir.ref_type [[INT]]
  // CHECK-NEXT: call @dummy([[REF]])
  // CHECK-NEXT: call @dummy([[REF]])
  // CHECK-NEXT: call @dummy([[REF]])
  %0 = hir.int_type
  %1 = hir.inferrable_type
  %2 = hir.ref_type %0
  %3 = hir.ref_type %1
  %4 = hir.unify_type %2, %3
  call @dummy(%2) : (!hir.type) -> ()
  call @dummy(%3) : (!hir.type) -> ()
  call @dummy(%4) : (!hir.type) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOps2
func.func @UnifyConcreteOps2() {
  // Cannot unify concrete types if the operands of don't dominate both types.
  // CHECK: hir.int_type
  // CHECK: hir.ref_type
  // CHECK: hir.inferrable_type
  // CHECK: hir.ref_type
  %0 = hir.int_type
  %1 = hir.ref_type %0
  %2 = hir.inferrable_type
  %3 = hir.ref_type %2
  %4 = hir.unify_type %1, %3
  call @dummy(%4) : (!hir.type) -> ()
  return
}
