// RUN: silicon-opt %s | silicon-opt | FileCheck %s

// CHECK: hir.int 42 loc("{{.+}}hir.mlir":{{.+}}:{{.+}})
hir.int 42
// CHECK: hir.int 42 loc("hello.si":42:9001)
hir.int 42 loc("hello.si":42:9001)
