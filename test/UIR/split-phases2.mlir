// RUN: silicon-opt --split-phases2 %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//
// Single-phase function: all ops at phase 0, no phase shifts.
// Produces one hir.func and a uir.split_func with a single entry.

// CHECK-LABEL: hir.func private @SinglePhase.0() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       } {
// CHECK:         func.call @dummyA
// CHECK:         hir.return -> ()

// CHECK-NOT: uir.func
// CHECK-LABEL: uir.split_func @SinglePhase() -> ()
// CHECK:         uir.signature () -> ()
// CHECK:       [
// CHECK:         0: @SinglePhase.0
uir.func @SinglePhase() -> () {
  uir.signature () -> ()
} {
  func.call @dummyA() : () -> ()
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Two-phase function: const arg at phase -1, result at phase 0.
// The type value %T from phase -1 must flow through opaque context.

// CHECK-LABEL: hir.func private @TwoPhase.0(%T) -> (ctx)
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @TwoPhase.1(%x, %ctx) -> (result)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return

// CHECK-NOT: uir.func
// CHECK-LABEL: uir.split_func @TwoPhase(%T: -1, %x: 0) -> (result: 0)
// CHECK:       -1: @TwoPhase.0
// CHECK:        0: @TwoPhase.1
uir.func @TwoPhase(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  uir.signature (%type_type, %T) -> (%T)
} {
  uir.return %x -> (%T)
}

//===----------------------------------------------------------------------===//
// Internal phases via uir.expr: a const { ... } block creates an internal
// phase -1 within a function whose args/results are all at phase 0.
// This produces a multiphase_func grouping two sub-phases under phase 0.

// CHECK-LABEL: hir.func private @InternalPhase.0a() -> (ctx)
// CHECK:         func.call @dummyA
// CHECK:         hir.opaque_pack()
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @InternalPhase.0b(%ctx) -> ()
// CHECK:         func.call @dummyB
// CHECK:         hir.return

// CHECK-LABEL: uir.split_func @InternalPhase() -> ()
// CHECK:       0: @InternalPhase.0

// CHECK-LABEL: hir.multiphase_func @InternalPhase.0() -> ()
// CHECK:       @InternalPhase.0a
// CHECK:       @InternalPhase.0b
uir.func @InternalPhase() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  %v = uir.expr pin -1 : %type_type {
    func.call @dummyA() : () -> ()
    %r = hir.int_type
    uir.yield %r : %r
  }
  func.call @dummyB() : () -> ()
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Cross-phase expr result: a value computed at phase -1 (inside const block)
// is used at phase 0. The value must flow through opaque context.

// CHECK-LABEL: hir.func private @ExprCrossPhase.0a() -> (ctx)
// CHECK:         func.call @dummyA
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @ExprCrossPhase.0b(%ctx) -> (result)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return
uir.func @ExprCrossPhase() -> (result: 0) {
  %type_type = hir.type_type
  uir.signature () -> (%type_type)
} {
  %type_type = hir.type_type
  %v = uir.expr pin -1 : %type_type {
    func.call @dummyA() : () -> ()
    %r = hir.int_type
    uir.yield %r : %r
  }
  uir.return %v -> (%type_type)
}
