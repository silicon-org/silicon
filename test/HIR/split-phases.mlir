// RUN: silicon-opt --split-phases %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//
// Dependent types: result type operand is a block argument (%T), not type_of.
// After splitting, the phase-0 function should use the threaded %T value as
// the return type operand, not fall back to hir.type_of.

// CHECK-LABEL: hir.func private @Identity.0(%T) -> (ctx)
// CHECK:      [[TT:%.+]] = hir.type_type
// CHECK:      [[CT:%.+]] = hir.coerce_type %T, [[TT]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[CT]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : (%0) -> ([[OT]])

// CHECK-LABEL: hir.func private @Identity.1(%x, %ctx) -> (result)
// CHECK:      [[UNPACK:%.+]] = hir.opaque_unpack %ctx
// CHECK:      [[CTXTY:%.+]] = hir.opaque_type
// CHECK:      [[X0:%.+]] = hir.coerce_type %x, [[UNPACK]]
// CHECK:      hir.return [[X0]] : ([[UNPACK]], [[CTXTY]]) -> ([[UNPACK]])

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @Identity(%T: -1, %x: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -1: @Identity.0
// CHECK:       0: @Identity.1
hir.unified_func @Identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.unified_signature (%type_type, %T) -> (%T)
} {
  hir.return %x : () -> (%T)
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @SinglePhase.0() -> ()
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @SinglePhase() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @SinglePhase.0
hir.unified_func @SinglePhase() -> () {
  hir.unified_signature () -> ()
} {
  func.call @dummyA() : () -> ()
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.0a() -> ()
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return : () -> ()

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.0b() -> ()
// CHECK-NEXT: func.call @dummyB
// CHECK-NEXT: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @TwoUnrelatedPhases() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @TwoUnrelatedPhases.0
// CHECK-LABEL: hir.multiphase_func @TwoUnrelatedPhases.0() -> ()
// CHECK:       @TwoUnrelatedPhases.0a
// CHECK:       @TwoUnrelatedPhases.0b
hir.unified_func @TwoUnrelatedPhases() -> () {
  hir.unified_signature () -> ()
} {
  func.call @dummyB() : () -> ()
  hir.expr -1 {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.0a() -> ()
// CHECK: [[C42:%.+]] = hir.constant_int 42
// CHECK: [[TMP:%.+]] = hir.constant_int 1337
// CHECK: hir.add [[C42]], [[TMP]] :
// CHECK: hir.return : () -> ()

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.0b() -> ()
// CHECK: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ValueUseAcrossPhases() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @ValueUseAcrossPhases.0
// CHECK-LABEL: hir.multiphase_func @ValueUseAcrossPhases.0() -> ()
// CHECK:       @ValueUseAcrossPhases.0a
// CHECK:       @ValueUseAcrossPhases.0b
hir.unified_func @ValueUseAcrossPhases() -> () {
  hir.unified_signature () -> ()
} {
  %0 = hir.constant_int 42
  %1 = hir.expr -1 : !hir.any {
    %3 = hir.constant_int 1337
    hir.yield %3 : !hir.any
  }
  %t0 = hir.type_of %0
  %t1 = hir.type_of %1
  %t = hir.unify %t0, %t1
  %2 = hir.add %0, %1 : %t
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Constness-aware split: a const argument flows from the const phase to the
// runtime phase.

// CHECK-LABEL: hir.func private @ConstArg.0(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[A0:%.+]] = hir.coerce_type %a, [[INT]]
// CHECK:      [[TA:%.+]] = hir.type_of [[A0]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[TA]], [[A0]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : (%0) -> ([[OT]])

// CHECK-LABEL: hir.func private @ConstArg.1(%b, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[R:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      hir.return [[R]] : ([[INT]], {{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ConstArg(%a: -1, %b: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -1: @ConstArg.0
// CHECK:       0: @ConstArg.1
hir.unified_func @ConstArg(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  hir.unified_signature (%0, %1) -> (%2)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t = hir.unify %ta, %tb
  %0 = hir.add %a, %b : %t
  %t0 = hir.type_of %0
  hir.return %0 : () -> (%t0)
}

//===----------------------------------------------------------------------===//
// Const arg pass-through: verifies that opaque_pack operands use the coerced
// value instead of the raw block arg. This is the pattern from add-const.si
// where a const arg flows into the next phase and the coerce_type was dead.

// CHECK-LABEL: hir.func private @ConstArgPassThrough.0(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[A0:%.+]] = hir.coerce_type %a, [[INT]]
// CHECK:      [[TA:%.+]] = hir.type_of [[A0]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[TA]], [[A0]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : (%0) -> ([[OT]])

// CHECK-LABEL: hir.func private @ConstArgPassThrough.1(%b, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      hir.return {{.*}} : ([[INT]], {{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ConstArgPassThrough(%a: -1, %b: 0) -> (result: 0)
hir.unified_func @ConstArgPassThrough(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.int_type
  %2 = hir.int_type
  hir.unified_signature (%0, %1) -> (%2)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t = hir.unify %ta, %tb
  %0 = hir.add %a, %b : %t
  %t0 = hir.type_of %0
  hir.return %0 : () -> (%t0)
}

//===----------------------------------------------------------------------===//
// Three-phase split: all three phases are externally visible (-2, -1, 0).
// No multiphase_func needed — each phase gets its own standalone entry in
// the split_func.

// CHECK-LABEL: hir.func private @ThreePhase.0(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[A0:%.+]] = hir.coerce_type %a, [[INT]]
// CHECK:      [[TA:%.+]] = hir.type_of [[A0]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[TA]], [[A0]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : (%0) -> ([[OT]])

// CHECK-LABEL: hir.func private @ThreePhase.1(%b, %ctx) -> (ctx)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[TMP:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      [[PACK:%.+]] = hir.opaque_pack({{.*}}, [[TMP]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : ([[INT]], {{.*}}) -> ([[OT]])

// CHECK-LABEL: hir.func private @ThreePhase.2(%c, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[C0:%.+]] = hir.coerce_type %c, [[INT]]
// CHECK:      [[RES:%.+]] = hir.add {{.*}}, [[C0]] :
// CHECK:      hir.return [[RES]] : ([[INT]], {{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ThreePhase(%a: -2, %b: -1, %c: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -2: @ThreePhase.0
// CHECK:       -1: @ThreePhase.1
// CHECK:       0: @ThreePhase.2
// CHECK-NOT: hir.multiphase_func @ThreePhase
hir.unified_func @ThreePhase(%a: -2, %b: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t0 = hir.unify %ta, %tb
  %0 = hir.add %a, %b : %t0
  %t0b = hir.type_of %0
  %tc = hir.type_of %c
  %t1 = hir.unify %t0b, %tc
  %1 = hir.add %0, %c : %t1
  %t1b = hir.type_of %1
  hir.return %1 : () -> (%t1b)
}

//===----------------------------------------------------------------------===//
// Three-phase call rewriting: a unified_call to a 3-phase function causes the
// caller to be split into matching phases. Each split function hosts a single
// per-phase call to the corresponding callee split function.

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0a
// CHECK: hir.call @ThreePhase.0(
// CHECK: hir.opaque_pack(
// CHECK: hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0b
// CHECK: hir.opaque_unpack
// CHECK: hir.call @ThreePhase.1(
// CHECK: hir.opaque_pack(
// CHECK: hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0c
// CHECK: hir.opaque_unpack
// CHECK: hir.coerce_type %z,
// CHECK: hir.call @ThreePhase.2(
// CHECK: hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ThreePhaseCaller(%z: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       0: @ThreePhaseCaller.0
// CHECK-LABEL: hir.multiphase_func @ThreePhaseCaller.0(last z) -> (result)
// CHECK:       @ThreePhaseCaller.0a
// CHECK:       @ThreePhaseCaller.0b
// CHECK:       @ThreePhaseCaller.0c
hir.unified_func @ThreePhaseCaller(%z: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  %a = hir.constant_int 10
  %b = hir.constant_int 20
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %t3 = hir.inferrable
  %r = hir.unified_call @ThreePhase(%a, %b, %z) : (%t0, %t1, %t2) -> (%t3) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  %tr = hir.type_of %r
  hir.return %r : () -> (%tr)
}

//===----------------------------------------------------------------------===//
// Internal compile-time phase: a function with no compile-time args that calls
// a const-arg function with constant arguments, creating an internal phase -1.
// The multiphase_func should be emitted even though there's only one
// compile-time phase.

// CHECK-LABEL: hir.func private @InternalPhase.0a() -> (ctx)
// CHECK: hir.call @ConstArg.0(
// CHECK: hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @InternalPhase.0b(%y, %ctx) -> (result)
// CHECK: hir.call @ConstArg.1(
// CHECK: hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @InternalPhase(%y: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       0: @InternalPhase.0
// CHECK-LABEL: hir.multiphase_func @InternalPhase.0(last y) -> (result)
// CHECK:       @InternalPhase.0a
// CHECK:       @InternalPhase.0b
hir.unified_func @InternalPhase(%y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  %a = hir.constant_int 42
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %r = hir.unified_call @ConstArg(%a, %y) : (%t0, %t1) -> (%t2) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %tr = hir.type_of %r
  hir.return %r : () -> (%tr)
}

//===----------------------------------------------------------------------===//
// Leading external then internal: phases -2, -1, 0. External at -2 and 0,
// internal at -1 (created by const=-1 on an expr inside a phase-0 context).
// Groups: [-2], [-1, 0].

// CHECK-LABEL: hir.func private @LeadingExternal.0(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      hir.coerce_type %a, [[INT]]
// CHECK:      hir.opaque_pack(
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-LABEL: hir.func private @LeadingExternal.1a(%ctx) -> (ctx)
// CHECK:      hir.opaque_unpack %ctx
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-LABEL: hir.func private @LeadingExternal.1b(%c, %ctx) -> (result)
// CHECK:      hir.opaque_unpack %ctx
// CHECK:      hir.coerce_type %c,
// CHECK:      hir.add
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @LeadingExternal(%a: -2, %c: 0) -> (result: 0)
// CHECK:       -2: @LeadingExternal.0
// CHECK:       0: @LeadingExternal.1
// CHECK-LABEL: hir.multiphase_func @LeadingExternal.1(first ctx, last c) -> (result)
// CHECK:       @LeadingExternal.1a
// CHECK:       @LeadingExternal.1b
hir.unified_func @LeadingExternal(%a: -2, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tc = hir.type_of %c
  %t = hir.unify %ta, %tc
  hir.expr -1 {
    hir.yield
  }
  %0 = hir.add %a, %c : %t
  %t0 = hir.type_of %0
  hir.return %0 : () -> (%t0)
}

//===----------------------------------------------------------------------===//
// Unified call targeting a pre-existing split_func: the callee is already split
// in the input IR (not a unified_func). The pass should decompose the
// unified_call using the split_func's phase mapping.

hir.func private @PreSplit.0(%a) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.coerce_type %a, %0
  %2 = hir.type_of %1
  %3 = hir.opaque_pack(%1)
  %4 = hir.opaque_type
  hir.return %3 : (%0) -> (%4)
}
hir.func private @PreSplit.1(%b, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.coerce_type %b, %1
  %3 = hir.add %0, %2 : %1
  %4 = hir.opaque_type
  hir.return %3 : (%1, %4) -> (%1)
}
hir.split_func @PreSplit(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0, %0) -> (%0)
} [
  -1: @PreSplit.0,
  0: @PreSplit.1
]

//===----------------------------------------------------------------------===//
// Visibility inheritance: a public unified_func should produce a public
// split_func, but multiphase_func ops should remain private.

// CHECK-LABEL: hir.func private @PublicVis.0a() -> ()
// CHECK: hir.return : () -> ()

// CHECK-LABEL: hir.func private @PublicVis.0b() -> ()
// CHECK: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func public @PublicVis() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @PublicVis.0
// CHECK-LABEL: hir.multiphase_func @PublicVis.0() -> ()
// CHECK:       @PublicVis.0a
// CHECK:       @PublicVis.0b
hir.unified_func public @PublicVis() -> () {
  hir.unified_signature () -> ()
} {
  func.call @dummyB() : () -> ()
  hir.expr -1 {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Shifted result phase: the return value lives at phase 0, but the declared
// result phase is 1 (as with `-> dyn int`). The value should flow through
// as a context return from phase 0 and appear as the result of phase 1.

// CHECK-LABEL: hir.func private @DynReturn.0(%x) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[X0:%.+]] = hir.coerce_type %x, [[INT]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[X0]]
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : (%0) -> ([[OT]])

// CHECK-LABEL: hir.func private @DynReturn.1(%ctx) -> (result)
// CHECK:      [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[CTXTY2:%.+]] = hir.opaque_type
// CHECK:      hir.return [[UNPACK]]#0 : ([[CTXTY2]]) -> ([[UNPACK]]#1)

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @DynReturn(%x: 0) -> (result: 1)
// CHECK:         hir.signature
// CHECK:       0: @DynReturn.0
// CHECK:       1: @DynReturn.1
hir.unified_func @DynReturn(%x: 0) -> (result: 1) {
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  %tx = hir.type_of %x
  hir.return %x : () -> (%tx)
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @CallsPreSplit.0a
// CHECK: hir.call @PreSplit.0(
// CHECK: hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @CallsPreSplit.0b(%y, %ctx) -> (result)
// CHECK: hir.call @PreSplit.1(
// CHECK: hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @CallsPreSplit(%y: 0) -> (result: 0)
// CHECK:       0: @CallsPreSplit.0
// CHECK-LABEL: hir.multiphase_func @CallsPreSplit.0(last y) -> (result)
// CHECK:       @CallsPreSplit.0a
// CHECK:       @CallsPreSplit.0b
hir.unified_func @CallsPreSplit(%y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  %a = hir.constant_int 99
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %r = hir.unified_call @PreSplit(%a, %y) : (%t0, %t1) -> (%t2) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %tr = hir.type_of %r
  hir.return %r : () -> (%tr)
}

//===----------------------------------------------------------------------===//
// Phase back-propagation: ExprOp wrapping a call to a single-phase function
// with only floating operands, passed to a const arg. The ExprOp should be
// pulled to the required phase (-1) so no error is emitted.

// CHECK-LABEL: hir.func private @Adder.0
// CHECK:      hir.add
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func @Adder
// CHECK-LABEL: hir.split_func @Adder(%a: 0, %b: 0) -> (result: 0)
hir.unified_func @Adder(%a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t = hir.unify %ta, %tb
  %r = hir.add %a, %b : %t
  hir.return %r : () -> (%t)
}

// CHECK-LABEL: hir.func private @PullExpr.0a() -> (ctx)
// CHECK:      hir.call @Adder.0(
// CHECK:      hir.call @ConstArg.0(
// CHECK:      hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @PullExpr.0b(%y, %ctx) -> (result)
// CHECK:      hir.opaque_unpack
// CHECK:      hir.call @ConstArg.1(
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @PullExpr(%y: 0) -> (result: 0)
// CHECK:       0: @PullExpr.0
// CHECK-LABEL: hir.multiphase_func @PullExpr.0(last y) -> (result)
// CHECK:       @PullExpr.0a
// CHECK:       @PullExpr.0b
hir.unified_func @PullExpr(%y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  %c19 = hir.constant_int 19
  %c23 = hir.constant_int 23
  %key = hir.expr 0 : !hir.any {
    %t0 = hir.int_type
    %t1 = hir.int_type
    %ti = hir.inferrable
    %t = hir.unified_call @Adder(%c19, %c23) : (%t0, %t1) -> (%ti) (!hir.any, !hir.any) -> !hir.any [0, 0] -> [0]
    hir.yield %t : !hir.any
  }
  %kt = hir.type_of %key
  %yt = hir.type_of %y
  %rt = hir.inferrable
  %r = hir.unified_call @ConstArg(%key, %y) : (%kt, %yt) -> (%rt) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %rrt = hir.type_of %r
  hir.return %r : () -> (%rrt)
}

//===----------------------------------------------------------------------===//
// Nested ExprOps: an outer ExprOp at the current phase containing an inner
// ExprOp shifted to an earlier phase. After splitting, both ExprOps are
// dissolved — the inner's ops land in the earlier phase function and the
// outer's remaining ops stay in the current phase.

// CHECK-LABEL: hir.func private @NestedExpr.0a() -> ()
// CHECK-NOT: hir.expr
// CHECK: func.call @dummyA
// CHECK: hir.return : () -> ()

// CHECK-LABEL: hir.func private @NestedExpr.0b() -> ()
// CHECK-NOT: hir.expr
// CHECK: func.call @dummyB
// CHECK: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @NestedExpr() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @NestedExpr.0
// CHECK-LABEL: hir.multiphase_func @NestedExpr.0() -> ()
// CHECK:       @NestedExpr.0a
// CHECK:       @NestedExpr.0b
hir.unified_func @NestedExpr() -> () {
  hir.unified_signature () -> ()
} {
  hir.expr 0 {
    hir.expr -1 {
      func.call @dummyA() : () -> ()
      hir.yield
    }
    func.call @dummyB() : () -> ()
    hir.yield
  }
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Nested ExprOps with value flow: the inner ExprOp produces a value at an
// earlier phase that flows through cross-phase threading to the outer ExprOp.

// CHECK-LABEL: hir.func private @NestedExprValue.0a() -> ()
// CHECK-NOT: hir.expr
// CHECK: hir.constant_int 100
// CHECK: hir.return : () -> ()

// CHECK-LABEL: hir.func private @NestedExprValue.0b() -> ()
// CHECK-NOT: hir.expr
// CHECK: [[V:%.+]] = hir.constant_int 100
// CHECK: hir.add [[V]], [[V]] :
// CHECK: hir.return : () -> ()

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @NestedExprValue() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @NestedExprValue.0
// CHECK-LABEL: hir.multiphase_func @NestedExprValue.0() -> ()
// CHECK:       @NestedExprValue.0a
// CHECK:       @NestedExprValue.0b
hir.unified_func @NestedExprValue() -> () {
  hir.unified_signature () -> ()
} {
  hir.expr 0 {
    %0 = hir.expr -1 : !hir.any {
      %1 = hir.constant_int 100
      hir.yield %1 : !hir.any
    }
    %t = hir.type_of %0
    %2 = hir.add %0, %0 : %t
    hir.yield
  }
  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Const block with a call: `const { noargs() }` shifts the call's phases by
// the ExprOp's phaseShift. The callee's phase-0 function should be called in
// the phase -1 split of the caller, and the result forwarded to phase 0.

// CHECK-LABEL: hir.func private @ConstBlockCallee.0() -> (result)
// CHECK:         hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @ConstBlockCall.0a() -> (ctx)
// CHECK:         hir.call @ConstBlockCallee.0()
// CHECK:         hir.opaque_pack
// CHECK:         hir.return {{.*}} : () -> ({{.*}})

// CHECK-LABEL: hir.func private @ConstBlockCall.0b(%ctx) -> (result)
// CHECK:         hir.opaque_unpack
// CHECK:         hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ConstBlockCall() -> (result: 0)
// CHECK:       0: @ConstBlockCall.0
// CHECK-LABEL: hir.multiphase_func @ConstBlockCall.0() -> (result)
// CHECK:       @ConstBlockCall.0a
// CHECK:       @ConstBlockCall.0b

hir.unified_func @ConstBlockCallee() -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  %0 = hir.constant_int 42
  %1 = hir.int_type
  hir.return %0 : () -> (%1)
}

hir.unified_func @ConstBlockCall() -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  %0 = hir.expr -1 : !hir.any {
    %4 = hir.int_type
    %5 = hir.unified_call @ConstBlockCallee() : () -> (%4) () -> !hir.any [] -> [0]
    hir.yield %5 : !hir.any
  }
  %1 = hir.type_of %0
  %2 = hir.int_type
  %3 = hir.unify %1, %2
  hir.return %0 : () -> (%3)
}

//===----------------------------------------------------------------------===//
// Parameterized type in signature: `uint_type` depends on a `constant_int`
// operand. Both must be cloned transitively into the split function.

// CHECK-LABEL: hir.func private @UIntLiteral.0(%a) -> (result)
// CHECK:         hir.constant_int 42
// CHECK-NEXT:    hir.uint_type
// CHECK:         hir.return

// CHECK-LABEL: hir.split_func private @UIntLiteral(%a: 0) -> (result: 0)
// CHECK:         hir.constant_int 42
// CHECK-NEXT:    hir.uint_type
// CHECK:         hir.signature

hir.unified_func private @UIntLiteral(%a: 0) -> (result: 0) attributes {isModule} {
  %0 = hir.constant_int 42
  %1 = hir.uint_type %0
  %2 = hir.unit_type
  hir.unified_signature (%1) -> (%2)
} {
  %0 = hir.constant_unit
  %1 = hir.unit_type
  hir.return %0 : () -> (%1)
}


//===----------------------------------------------------------------------===//
// Dyn return with phase-0 value: `fn make_dyn(x: int) -> dyn int { x }`.
// The return value `x` is phase 0 but the declared result phase is 1 (dyn).
// The effective result phase stays at 1 (max(1, 0) = 1), so the value flows
// through as context from phase 0 to phase 1.

// CHECK-LABEL: hir.func private @DynReturnPhase0Val.0(%x) -> (ctx)
// CHECK:      hir.coerce_type %x,
// CHECK:      hir.opaque_pack(
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-LABEL: hir.func private @DynReturnPhase0Val.1(%ctx) -> (result)
// CHECK:      hir.opaque_unpack %ctx
// CHECK:      hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @DynReturnPhase0Val(%x: 0) -> (result: 1)
// CHECK:         hir.signature
// CHECK:       0: @DynReturnPhase0Val.0
// CHECK:       1: @DynReturnPhase0Val.1
hir.unified_func @DynReturnPhase0Val(%x: 0) -> (result: 1) {
  %0 = hir.int_type
  hir.unified_signature (%0) -> (%0)
} {
  %tx = hir.type_of %x
  hir.return %x : () -> (%tx)
}
