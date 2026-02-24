// RUN: silc --parse-only %s | FileCheck %s
// RUN: silc --format=mlir --parse-only %s | FileCheck %s
// RUN: silc --format=mlir --parse-only - < %s | FileCheck %s

// Test that silc accepts MLIR input directly, auto-detecting from the .mlir
// extension and accepting an explicit --format=mlir flag, including stdin.

// CHECK: module {
// CHECK: func.func @test
// CHECK: hir.constant_int 42

module {
  func.func @test() {
    %0 = hir.constant_int 42
    return
  }
}
