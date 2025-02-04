// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// CHECK-LABEL: func @Foo
func.func @Foo(%arg0: !hir.type, %arg1: !hir.type, %arg2: !hir.type) {
  // CHECK: hir.inferrable_type
  hir.inferrable_type
  // CHECK: hir.int_type
  hir.int_type
  // CHECK: hir.ref_type %arg0
  hir.ref_type %arg0
  // CHECK: hir.unify_type %arg0, %arg1
  hir.unify_type %arg0, %arg1
  // CHECK: hir.let "x" : %arg0
  hir.let "x" : %arg0
  // CHECK: hir.store %arg0, %arg1 : %arg2
  hir.store %arg0, %arg1 : %arg2
  return
}
