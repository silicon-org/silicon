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
// CHECK:         %[[TT:.+]] = hir.type_type
// CHECK:         hir.signature (%{{.+}}) -> (%[[TT]])
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
// Dependent type: type of %B depends on value of %N from an earlier phase.
// The signature must compute `hir.uint_type %N` where %N flows through
// opaque context.

// CHECK-LABEL: hir.func private @DepType.0(%N) -> (ctx)
// CHECK:         %[[INT:.+]] = hir.int_type
// CHECK:         hir.opaque_type
// CHECK:         hir.signature (%[[INT]]) -> (
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase 0 sig: both %N and uint_type(%N) flow from context. The sig
// picks the pre-computed uint_type result directly from the unpack.
// CHECK-LABEL: hir.func private @DepType.1(%B, %ctx) -> (result)
// CHECK:         %[[U:.+]]:2 = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U]]#1, %{{.+}}) -> (%[[U]]#0)
// CHECK:         hir.return
uir.func @DepType(%N: -1, %B: 0) -> (result: 0) {
  %int_type = hir.int_type
  %b_type_sig = hir.uint_type %N
  uir.signature (%int_type, %b_type_sig) -> (%b_type_sig)
} {
  %b_type_body = hir.uint_type %N
  uir.return %B -> (%b_type_body)
}

//===----------------------------------------------------------------------===//
// Chained dependent types: type of %C depends on type of %B which depends on
// %A. Both type computations use hir.uint_type. The sig for phase 0 must
// reconstruct the full chain from context.

// CHECK-LABEL: hir.func private @ChainedDepType.0(%A) -> (ctx)
// CHECK:         hir.int_type
// CHECK:         hir.signature
// CHECK:         hir.opaque_pack(
// CHECK:         hir.return

// Phase 0 sig: both types (uint_type(%A) and uint_type(uint_type(%A)))
// flow pre-computed from context. The sig picks them directly.
// CHECK-LABEL: hir.func private @ChainedDepType.1(%B, %C, %ctx) -> ()
// CHECK:         %[[U:.+]]:2 = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U]]#0, %[[U]]#1, %{{.+}}) -> ()
// CHECK:         hir.return
uir.func @ChainedDepType(%A: -1, %B: 0, %C: 0) -> () {
  %int_type = hir.int_type
  %b_type = hir.uint_type %A
  %c_type = hir.uint_type %b_type
  uir.signature (%int_type, %b_type, %c_type) -> ()
} {
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Signature with a call: the type of %B is computed by a func.call in the sig.
// The call should be cloned into the body, distributed, and its result
// threaded through opaque context to the per-phase sig.

func.func private @computeType(!hir.any) -> !hir.any

// Phase -1: call @computeType in the body, pack result into context.
// CHECK-LABEL: hir.func private @SigWithCall.0(%A) -> (ctx)
// CHECK:         hir.int_type
// CHECK:         hir.signature
// CHECK:       } {
// CHECK:         func.call @computeType(%A)
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// Phase 0 sig: the call result flows from context, used as %B's type.
// CHECK-LABEL: hir.func private @SigWithCall.1(%B, %ctx) -> ()
// CHECK:         %[[U:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U]],
// CHECK:         hir.return
uir.func @SigWithCall(%A: -1, %B: 0) -> () {
  %int_type = hir.int_type
  %b_type = func.call @computeType(%A) : (!hir.any) -> !hir.any
  uir.signature (%int_type, %b_type) -> ()
} {
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Shared non-trivial type: both %B and %C share the same computed type.
// The sig reconstruction must clone the call result once, not twice.

// CHECK-LABEL: hir.func private @SharedType.1(%B, %C, %ctx) -> ()
// CHECK:         %[[U:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U]], %[[U]],
// CHECK:         hir.return
uir.func @SharedType(%A: -1, %B: 0, %C: 0) -> () {
  %int_type = hir.int_type
  %bt = func.call @computeType(%A) : (!hir.any) -> !hir.any
  uir.signature (%int_type, %bt, %bt) -> ()
} {
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Deep type computation chain: each arg's type depends on the previous arg,
// computed via func.call @computeType. Each call is at a different phase.

func.func private @derive(!hir.any) -> !hir.any

// Phase -3: provides %A, calls @derive(%A) to get %bt.
// CHECK-LABEL: hir.func private @DeepChain.0(%A) -> (ctx)
// CHECK:         hir.int_type
// CHECK:         hir.signature
// CHECK:         @derive(%A)
// CHECK:         hir.opaque_pack
// CHECK:         hir.return

// Phase -2: unpacks %bt from ctx, calls @derive(%bt) to get %ct.
// CHECK-LABEL: hir.func private @DeepChain.1(%B, %ctx) -> (ctx)
// CHECK:         %[[U1:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U1]],
// CHECK:         hir.return

// Phase -1: unpacks %ct from ctx, calls @derive(%ct) to get %dt.
// CHECK-LABEL: hir.func private @DeepChain.2(%C, %ctx) -> (ctx)
// CHECK:         %[[U2:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U2]],
// CHECK:         hir.return

// Phase 0: unpacks %dt from ctx, uses as %D's type.
// CHECK-LABEL: hir.func private @DeepChain.3(%D, %ctx) -> ()
// CHECK:         %[[U3:.+]] = hir.opaque_unpack %ctx
// CHECK:         hir.signature (%[[U3]],
// CHECK:         hir.return
uir.func @DeepChain(%A: -3, %B: -2, %C: -1, %D: 0) -> () {
  %int_type = hir.int_type
  %bt = func.call @derive(%A) : (!hir.any) -> !hir.any
  %ct = func.call @derive(%bt) : (!hir.any) -> !hir.any
  %dt = func.call @derive(%ct) : (!hir.any) -> !hir.any
  uir.signature (%int_type, %bt, %ct, %dt) -> ()
} {
  uir.return -> ()
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
// CHECK:         %[[TT:.+]] = hir.type_type
// CHECK:         hir.signature (%{{.+}}) -> (%[[TT]], %{{.+}})
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

//===----------------------------------------------------------------------===//
// Structured CF survival: uir.if at phase 0 moves as a whole into the
// per-phase function. Then/else regions and yield are preserved.

// CHECK-LABEL: hir.func private @IfSurvival.0() -> ()
// CHECK:       } {
// CHECK:         uir.if %{{.+}} {
// CHECK:           func.call @dummyA
// CHECK:           uir.yield
// CHECK:         } else {
// CHECK:           func.call @dummyB
// CHECK:           uir.yield
// CHECK:         }
// CHECK:         hir.return
uir.func @IfSurvival() -> () {
  uir.signature () -> ()
} {
  %cond = hir.int_type
  uir.if %cond {
    func.call @dummyA() : () -> ()
    uir.yield
  } else {
    func.call @dummyB() : () -> ()
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Structured CF with results: uir.if produces a value via yield.

// CHECK-LABEL: hir.func private @IfWithResult.0() -> (result)
// CHECK:       } {
// CHECK:         uir.if %{{.+}} : %{{.+}} {
// CHECK:           %[[A:.+]] = hir.int_type
// CHECK:           uir.yield %[[A]] : %[[A]]
// CHECK:         } else {
// CHECK:           %[[B:.+]] = hir.int_type
// CHECK:           uir.yield %[[B]] : %[[B]]
// CHECK:         }
// CHECK:         hir.return
uir.func @IfWithResult() -> (result: 0) {
  %tt = hir.type_type
  uir.signature () -> (%tt)
} {
  %cond = hir.int_type
  %tt = hir.type_type
  %r = uir.if %cond : %tt {
    %a = hir.int_type
    uir.yield %a : %a
  } else {
    %b = hir.int_type
    uir.yield %b : %b
  }
  uir.return %r -> (%tt)
}

//===----------------------------------------------------------------------===//
// Loop survival: uir.loop with uir.break inside a nested uir.if.

// CHECK-LABEL: hir.func private @LoopSurvival.0() -> ()
// CHECK:       } {
// CHECK:         uir.loop {
// CHECK:           uir.if %{{.+}} {
// CHECK:             uir.break
// CHECK:           }
// CHECK:           uir.yield
// CHECK:         }
// CHECK:         hir.return
uir.func @LoopSurvival() -> () {
  uir.signature () -> ()
} {
  %cond = hir.int_type
  uir.loop {
    uir.if %cond {
      uir.break
    }
    uir.yield
  }
  uir.return -> ()
}

//===----------------------------------------------------------------------===//
// Pin dissolution: uir.pin outputs are replaced by inputs.

// CHECK-LABEL: hir.func private @PinDissolution.0() -> (result)
// CHECK:       } {
// CHECK-NOT:     uir.pin
// CHECK:         %[[V:.+]] = hir.int_type
// CHECK:         hir.return %[[V]]
uir.func @PinDissolution() -> (result: 0) {
  %tt = hir.type_type
  uir.signature () -> (%tt)
} {
  %v = hir.int_type
  %tt = hir.type_type
  %pinned = uir.pin %v, 0 : !hir.any
  uir.return %pinned -> (%tt)
}

//===----------------------------------------------------------------------===//
// Early return inside uir.if: the uir.return inside the then branch
// survives splitting (FlattenCF lowers it later).

// CHECK-LABEL: hir.func private @EarlyReturn.0() -> (result)
// CHECK:       } {
// CHECK:         uir.if %{{.+}} {
// CHECK:           uir.return %{{.+}} -> (
// CHECK:         }
// CHECK:         uir.unreachable
uir.func @EarlyReturn() -> (result: 0) {
  %tt = hir.type_type
  uir.signature () -> (%tt)
} {
  %cond = hir.int_type
  %v = hir.int_type
  %tt = hir.type_type
  uir.if %cond {
    uir.return %v -> (%tt)
  }
  uir.unreachable
}
