// RUN: ! silicon-opt --split-input-file %s 2>&1 | FileCheck %s

si.module @Foo {
  // CHECK: 'si.return' must be nested within 'si.func'
  si.return
}

// -----

si.func @Foo() {
  %0 = si.constant 0 : i42 : i42
  // CHECK: return argument types must match function output types
  si.return %0 : i42
}
