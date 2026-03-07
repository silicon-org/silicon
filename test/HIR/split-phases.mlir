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
// CHECK:      hir.return [[PACK]] : [[OT]]

// CHECK-LABEL: hir.func private @Identity.1(%x, %ctx) -> (result)
// CHECK:      [[UNPACK:%.+]] = hir.opaque_unpack %ctx
// CHECK:      [[X0:%.+]] = hir.coerce_type %x, [[UNPACK]]
// CHECK:      hir.return [[X0]] : [[UNPACK]]

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @Identity(%T: -1, %x: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -1: @Identity.0
// CHECK:       0: @Identity.1
hir.unified_func @Identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.unified_signature (%type_type, %T) -> (%T)
} {
  hir.unified_return %x : %T
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @SinglePhase.0() -> ()
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @SinglePhase() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @SinglePhase.0
hir.unified_func @SinglePhase() -> () {
  hir.unified_signature () -> ()
} {
  func.call @dummyA() : () -> ()
  hir.unified_return
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.0a() -> ()
// CHECK-NEXT: hir.expr
// CHECK-NEXT:   func.call @dummyA
// CHECK-NEXT:   hir.yield
// CHECK-NEXT: }
// CHECK-NEXT: hir.return

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.0b() -> ()
// CHECK-NEXT: func.call @dummyB
// CHECK-NEXT: hir.return

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
  hir.expr attributes {const = -1} {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.unified_return
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.0a() -> ()
// CHECK: [[C42:%.+]] = hir.constant_int 42
// CHECK: [[TMP:%.+]] = hir.expr
// CHECK:   hir.constant_int 1337
// CHECK: hir.add [[C42]], [[TMP]] :
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.0b() -> ()
// CHECK: hir.return

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
  %1 = hir.expr : !hir.any attributes {const = -1} {
    %3 = hir.constant_int 1337
    hir.yield %3 : !hir.any
  }
  %t0 = hir.type_of %0
  %t1 = hir.type_of %1
  %t = hir.unify %t0, %t1
  %2 = hir.add %0, %1 : %t
  hir.unified_return
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
// CHECK:      hir.return [[PACK]] : [[OT]]

// CHECK-LABEL: hir.func private @ConstArg.1(%b, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[R:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      hir.return [[R]] : {{.*}}

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
  hir.unified_return %0 : %t0
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
// CHECK:      hir.return [[PACK]] : [[OT]]

// CHECK-LABEL: hir.func private @ConstArgPassThrough.1(%b, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      hir.return

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
  hir.unified_return %0 : %t0
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
// CHECK:      hir.return [[PACK]] : [[OT]]

// CHECK-LABEL: hir.func private @ThreePhase.1(%b, %ctx) -> (ctx)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[TMP:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      [[PACK:%.+]] = hir.opaque_pack({{.*}}, [[TMP]])
// CHECK:      [[OT:%.+]] = hir.opaque_type
// CHECK:      hir.return [[PACK]] : [[OT]]

// CHECK-LABEL: hir.func private @ThreePhase.2(%c, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[C0:%.+]] = hir.coerce_type %c, [[INT]]
// CHECK:      [[RES:%.+]] = hir.add {{.*}}, [[C0]] :
// CHECK:      hir.return [[RES]] : {{.*}}

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
  hir.unified_return %1 : %t1b
}

//===----------------------------------------------------------------------===//
// Three-phase call rewriting: a unified_call to a 3-phase function causes the
// caller to be split into matching phases. Each split function hosts a single
// per-phase call to the corresponding callee split function.

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0a
// CHECK: hir.call @ThreePhase.0(
// CHECK: hir.opaque_pack(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0b
// CHECK: hir.opaque_unpack
// CHECK: hir.call @ThreePhase.1(
// CHECK: hir.opaque_pack(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ThreePhaseCaller.0c
// CHECK: hir.opaque_unpack
// CHECK: hir.coerce_type %z,
// CHECK: hir.call @ThreePhase.2(
// CHECK: hir.return

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
  hir.unified_return %r : %tr
}

//===----------------------------------------------------------------------===//
// Internal compile-time phase: a function with no compile-time args that calls
// a const-arg function with constant arguments, creating an internal phase -1.
// The multiphase_func should be emitted even though there's only one
// compile-time phase.

// CHECK-LABEL: hir.func private @InternalPhase.0a() -> (ctx)
// CHECK: hir.call @ConstArg.0(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @InternalPhase.0b(%y, %ctx) -> (result)
// CHECK: hir.call @ConstArg.1(
// CHECK: hir.return

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
  hir.unified_return %r : %tr
}

//===----------------------------------------------------------------------===//
// Leading external then internal: phases -2, -1, 0. External at -2 and 0,
// internal at -1 (created by const=-1 on an expr inside a phase-0 context).
// Groups: [-2], [-1, 0].

// CHECK-LABEL: hir.func private @LeadingExternal.0(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      hir.coerce_type %a, [[INT]]
// CHECK:      hir.opaque_pack(
// CHECK:      hir.return

// CHECK-LABEL: hir.func private @LeadingExternal.1a(%ctx) -> (ctx)
// CHECK:      hir.opaque_unpack %ctx
// CHECK:      hir.return

// CHECK-LABEL: hir.func private @LeadingExternal.1b(%c, %ctx) -> (result)
// CHECK:      hir.opaque_unpack %ctx
// CHECK:      hir.coerce_type %c,
// CHECK:      hir.add
// CHECK:      hir.return

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
  hir.expr attributes {const = -1} {
    hir.yield
  }
  %0 = hir.add %a, %c : %t
  %t0 = hir.type_of %0
  hir.unified_return %0 : %t0
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
  hir.return %3 : %4
}
hir.func private @PreSplit.1(%b, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.coerce_type %b, %1
  %3 = hir.add %0, %2 : %1
  hir.return %3 : %1
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
// CHECK: hir.return

// CHECK-LABEL: hir.func private @PublicVis.0b() -> ()
// CHECK: hir.return

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
  hir.expr attributes {const = -1} {
    func.call @dummyA() : () -> ()
    hir.yield
  }
  hir.unified_return
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @CallsPreSplit.0a
// CHECK: hir.call @PreSplit.0(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @CallsPreSplit.0b(%y, %ctx) -> (result)
// CHECK: hir.call @PreSplit.1(
// CHECK: hir.return

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
  hir.unified_return %r : %tr
}
