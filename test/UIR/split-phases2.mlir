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

// Signature for phase -1: %T's type is hir.type_type (trivially materializable).
// CHECK-LABEL: hir.func private @TwoPhase.0(%T) -> (ctx)
// CHECK:         hir.type_type
// CHECK:         hir.opaque_type
// CHECK:         hir.signature
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// Signature for phase 0: %x's type comes from context (%T = unpack result).
// CHECK-LABEL: hir.func private @TwoPhase.1(%x, %ctx) -> (result)
// CHECK:         %[[T:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.opaque_type
// CHECK:         hir.signature (%[[T]], %{{.+}}) -> (%[[T]])
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

//===----------------------------------------------------------------------===//
// Complex callee: 3 args at phases -3, -1, 1 and 3 results at phases -2, 0, 2.
// Plus an internal phase at -4 from a const block.
// External phases: {-3, -2, -1, 0, 1, 2}. Internal: {-4}.
// Phase range: [-4, 2].
// Groups: {-4, -3}, {-2}, {-1}, {0}, {1}, {2}
//   Group 0 = multiphase [-4, -3]
//   Groups 1-5 = single phase each
// Split func maps: -3: @.0, -2: @.1, -1: @.2, 0: @.3, 1: @.4, 2: @.5

// CHECK-LABEL: hir.func private @SixPhase.0a() -> (ctx)
// CHECK:         hir.opaque_pack()
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.0b(%a, %ctx) -> (ctx)
// CHECK:         hir.opaque_pack(%a)
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.1(%ctx) -> (r1, ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.2(%b, %ctx) -> (ctx)
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.3(%ctx) -> (r2, ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.4(%c, %ctx) -> (ctx)
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// CHECK-LABEL: hir.func private @SixPhase.5(%ctx) -> (r3)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.return

// CHECK-LABEL: uir.split_func @SixPhase(%a: -3, %b: -1, %c: 1) -> (r1: -2, r2: 0, r3: 2)
// CHECK:       -3: @SixPhase.0
// CHECK:       -2: @SixPhase.1
// CHECK:       -1: @SixPhase.2
// CHECK:        0: @SixPhase.3
// CHECK:        1: @SixPhase.4
// CHECK:        2: @SixPhase.5

// CHECK-LABEL: hir.multiphase_func @SixPhase.0
// CHECK:       @SixPhase.0a
// CHECK:       @SixPhase.0b
uir.func @SixPhase(%a: -3, %b: -1, %c: 1) -> (r1: -2, r2: 0, r3: 2) {
  %tt = hir.type_type
  uir.signature (%tt, %tt, %tt) -> (%tt, %tt, %tt)
} {
  %tt = hir.type_type
  // Internal phase at -4: creates a value that is unused (just exercises
  // the internal phase grouping into multiphase_func).
  %v = uir.expr pin -4 : %tt {
    %unused = hir.int_type
    uir.yield %unused : %unused
  }
  uir.return %a, %b, %c -> (%tt, %tt, %tt)
}

//===----------------------------------------------------------------------===//
// Caller of @SixPhase: exercises full call decomposition across 6 entries.
// The caller provides 3 args and receives 3 results.
// With callOpPhase=0, the split entries map to absolute caller phases
// -3, -2, -1, 0, 1, 2. Since the caller has no external phases besides 0,
// groups are: {-3, -2, -1, 0} (multiphase), {1, 2} (trailing multiphase).

// Phase -3 entry: %a (tagged {arg_a}) is passed to @SixPhase.0.
// CHECK-LABEL: hir.func private @CallSixPhase.0a() -> (ctx)
// CHECK:         %[[A:.+]] = hir.int_type {arg_a}
// CHECK:         hir.call @SixPhase.0(%[[A]])
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase -2 entry: no user args, just context from previous.
// CHECK-LABEL: hir.func private @CallSixPhase.0b(%ctx) -> (ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @SixPhase.1(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase -1 entry: %b (tagged {arg_b}) is passed to @SixPhase.2.
// CHECK-LABEL: hir.func private @CallSixPhase.0c(%ctx) -> (ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         %[[B:.+]] = hir.int_type {arg_b}
// CHECK:         hir.call @SixPhase.2(%[[B]],
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase 0 entry: no user args, just context.
// CHECK-LABEL: hir.func private @CallSixPhase.0d(%ctx) -> (ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @SixPhase.3(
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase 1 entry: %c (tagged {arg_c}) is passed to @SixPhase.4.
// CHECK-LABEL: hir.func private @CallSixPhase.1a(%ctx) -> (ctx)
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         %[[C:.+]] = hir.int_type {arg_c}
// CHECK:         hir.call @SixPhase.4(%[[C]],
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase 2 entry: no user args, receives last result.
// CHECK-LABEL: hir.func private @CallSixPhase.1b(%ctx) -> ()
// CHECK:         hir.opaque_unpack %ctx
// CHECK:         hir.call @SixPhase.5(
// CHECK:         hir.return
uir.func @CallSixPhase() -> () {
  uir.signature () -> ()
} {
  %tt = hir.type_type
  %a = hir.int_type {arg_a}
  %b = hir.int_type {arg_b}
  %c = hir.int_type {arg_c}
  %r1, %r2, %r3 = uir.call @SixPhase(%a, %b, %c) : (%tt, %tt, %tt) -> (%tt, %tt, %tt) (!hir.any, !hir.any, !hir.any) -> (!hir.any, !hir.any, !hir.any) [-3, -1, 1] -> [-2, 0, 2]
  uir.return -> ()
}
