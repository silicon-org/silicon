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

// -----

// CHECK: @UnknownFunc does not exist
si.call @UnknownFunc() : () -> ()

// -----

si.module @Foo {}

// CHECK: @Foo cannot be called
si.call @Foo() : () -> ()

// -----

si.func @Foo(%arg0 : i42) -> i42 {
  si.return %arg0 : i42
}
%0 = si.constant 0 : i43 : i43
// CHECK: call type (i43) -> i44 differs from callee type (i42) -> i42
si.call @Foo(%0) : (i43) -> (i44)
