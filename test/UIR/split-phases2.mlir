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
// Multiple args at the same phase: both %a and %b at phase -1.
// Only %a is used as the return value; %b is unused in the body.

// CHECK-LABEL: hir.func private @TwoArgsOnePhase.0(%a, %b) -> (ctx)
// CHECK:         hir.opaque_pack(%a)
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @TwoArgsOnePhase.1(%ctx) -> (result)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return
uir.func @TwoArgsOnePhase(%a: -1, %b: -1) -> (result: 0) {
  %type_type = hir.type_type
  uir.signature (%type_type, %type_type) -> (%type_type)
} {
  %type_type2 = hir.type_type
  uir.return %a -> (%type_type2)
}

//===----------------------------------------------------------------------===//
// Call decomposition: @Caller calls @TwoPhase which has been split into
// phases -1 and 0. The unified call becomes two hir.call ops, one per
// callee split entry.

// CHECK-LABEL: hir.func private @Caller.0a() -> (ctx)
// CHECK:         hir.call @TwoPhase.0(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @Caller.0b(%ctx) -> ()
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @TwoPhase.1(
// CHECK:         hir.return
uir.func @Caller() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  %T = hir.int_type
  %x = hir.int_type
  %r = uir.call @TwoPhase(%T, %x) : (%type_type, %T) -> (%T) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
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

//===----------------------------------------------------------------------===//
// Additional callees for call decomposition tests.

// A callee with 3 args at different phases.
uir.func @ThreeArgCallee(%a: -2, %b: -1, %c: 0) -> (r: 0) {
  %tt = hir.type_type
  uir.signature (%tt, %tt, %tt) -> (%tt)
} {
  %tt = hir.type_type
  uir.return %c -> (%tt)
}

// A callee with 3 results (all at phase 0).
uir.func @ThreeResultCallee(%x: 0) -> (r1: 0, r2: 0, r3: 0) {
  %tt = hir.type_type
  uir.signature (%tt) -> (%tt, %tt, %tt)
} {
  %tt = hir.type_type
  uir.return %x, %x, %x -> (%tt, %tt, %tt)
}

//===----------------------------------------------------------------------===//
// Call with zero args, zero results: void call to void callee.

// CHECK-LABEL: hir.func private @CallVoid.0() -> ()
// CHECK:         hir.call @SinglePhase.0()
// CHECK:         hir.return
uir.func @CallVoid() -> () {
  uir.signature () -> ()
} {
  uir.call @SinglePhase() : () -> () () -> () [] -> []
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Call with 3 args at different phases: tests partitioning args across 3
// split entries and chaining opaque context through them.

// CHECK-LABEL: hir.func private @CallThreeArgs.0a() -> (ctx)
// CHECK:         hir.call @ThreeArgCallee.0(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @CallThreeArgs.0b(%ctx) -> (ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @ThreeArgCallee.1(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @CallThreeArgs.0c(%ctx) -> ()
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @ThreeArgCallee.2(
// CHECK:         hir.return
uir.func @CallThreeArgs() -> () {
  uir.signature () -> ()
} {
  %tt = hir.type_type
  %a = hir.int_type
  %b = hir.int_type
  %c = hir.int_type
  %r = uir.call @ThreeArgCallee(%a, %b, %c) : (%tt, %tt, %tt) -> (%tt) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Call with 3 results: tests that multiple results are correctly handled.

// CHECK-LABEL: hir.func private @CallThreeResults.0() -> ()
// CHECK:         hir.call @ThreeResultCallee.0(
// CHECK-SAME:    -> (
// CHECK:         hir.return
uir.func @CallThreeResults() -> () {
  uir.signature () -> ()
} {
  %tt = hir.type_type
  %x = hir.int_type
  %r1, %r2, %r3 = uir.call @ThreeResultCallee(%x) : (%tt) -> (%tt, %tt, %tt) (!hir.any) -> (!hir.any, !hir.any, !hir.any) [0] -> [0, 0, 0]
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Nested calls: result of one call feeds directly into another call's arg.
// @TwoPhase(%T: -1, %x: 0) -> (result: 0)
// We call @TwoPhase twice, feeding the result of the first into the second.

// CHECK-LABEL: hir.func private @NestedCalls.0a() -> (ctx)
// CHECK:         hir.call @TwoPhase.0(
// CHECK:         hir.call @TwoPhase.0(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @NestedCalls.0b(%ctx) -> ()
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @TwoPhase.1(
// CHECK:         hir.call @TwoPhase.1(
// CHECK:         hir.return
uir.func @NestedCalls() -> () {
  uir.signature () -> ()
} {
  %tt = hir.type_type
  %T1 = hir.int_type
  %x1 = hir.int_type
  %r1 = uir.call @TwoPhase(%T1, %x1) : (%tt, %T1) -> (%T1) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %T2 = hir.int_type
  %r2 = uir.call @TwoPhase(%T2, %r1) : (%tt, %T2) -> (%T2) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  uir.return -> ()
}
