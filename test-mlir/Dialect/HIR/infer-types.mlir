// RUN: silicon-opt --infer-types %s | FileCheck %s

func.func private @dummy(%arg0: !hir.type)

// CHECK: [[T:%.+]] = hir.inferrable_type {a}
// CHECK-NOT: hir.inferrable_type {b}
// CHECK-NOT: hir.unify_type
// CHECK: func.call @dummy([[T]])
%0 = hir.inferrable_type {a}
%1 = hir.inferrable_type {b}
%2 = hir.unify_type %0, %1
func.call @dummy(%2) : (!hir.type) -> ()

// CHECK: [[T:%.+]] = hir.inferrable_type {a}
// CHECK-NOT: hir.inferrable_type {b}
// CHECK-NOT: hir.unify_type
// CHECK: func.call @dummy([[T]])
%3 = hir.inferrable_type {a}
%4 = hir.inferrable_type {b}
%5 = hir.unify_type %4, %3  // reversed
func.call @dummy(%5) : (!hir.type) -> ()
