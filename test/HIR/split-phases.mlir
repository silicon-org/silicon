// RUN: silicon-opt --split-phases %s | FileCheck %s

func.func private @dummyA()
func.func private @dummyB()

//===----------------------------------------------------------------------===//
// Dependent types: result type operand is a block argument (%T), not type_of.
// After splitting, the phase-0 function should use the threaded %T value as
// the return type operand, not fall back to hir.type_of.

// CHECK-LABEL: hir.func private @Identity.const1(%T) -> (ctx)
// CHECK:      [[TT:%.+]] = hir.type_type
// CHECK:      hir.coerce_type %T, [[TT]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack(%T)
// CHECK:      hir.return([[PACK]]) : ({{.*}})

// CHECK-LABEL: hir.func private @Identity.const0(%x, %ctx) -> (result)
// CHECK:      [[UNPACK:%.+]] = hir.opaque_unpack %ctx
// CHECK:      [[X0:%.+]] = hir.coerce_type %x, [[UNPACK]]
// CHECK:      hir.return([[X0]]) : ([[UNPACK]])

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @Identity(%T: -1, %x: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -1: @Identity.const1
// CHECK:       0: @Identity.const0
hir.unified_func @Identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.unified_signature (%type_type, %T) -> (%T)
} {
  hir.unified_return (%x) : (%T)
}

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @SinglePhase.const0() -> ()
// CHECK-NEXT: func.call @dummyA
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @SinglePhase() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       0: @SinglePhase.const0
hir.unified_func @SinglePhase() -> () {
  hir.unified_signature () -> ()
} {
  func.call @dummyA() : () -> ()
  hir.unified_return
}

//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.const1() -> ()
// CHECK-NEXT: hir.expr
// CHECK-NEXT:   func.call @dummyA
// CHECK-NEXT:   hir.yield
// CHECK-NEXT: }
// CHECK-NEXT: hir.return

// CHECK-LABEL: hir.func private @TwoUnrelatedPhases.const0() -> ()
// CHECK-NEXT: func.call @dummyB
// CHECK-NEXT: hir.return

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @TwoUnrelatedPhases() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       -1: @TwoUnrelatedPhases.const1
// CHECK:       0: @TwoUnrelatedPhases.const0
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

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const1() -> ()
// CHECK: [[C42:%.+]] = hir.constant_int 42
// CHECK: [[TMP:%.+]] = hir.expr
// CHECK:   hir.constant_int 1337
// CHECK: hir.add [[C42]], [[TMP]] :
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ValueUseAcrossPhases.const0() -> ()
// CHECK: hir.return

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ValueUseAcrossPhases() -> ()
// CHECK:         hir.signature () -> ()
// CHECK:       -1: @ValueUseAcrossPhases.const1
// CHECK:       0: @ValueUseAcrossPhases.const0
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

// CHECK-LABEL: hir.func private @ConstArg.const1(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[A0:%.+]] = hir.coerce_type %a, [[INT]]
// CHECK:      [[TA:%.+]] = hir.type_of [[A0]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[TA]], %a)
// CHECK:      hir.return([[PACK]]) : ({{.*}})

// CHECK-LABEL: hir.func private @ConstArg.const0(%b, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[R:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      hir.return([[R]]) : ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ConstArg(%a: -1, %b: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -1: @ConstArg.const1
// CHECK:       0: @ConstArg.const0
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
  hir.unified_return (%0) : (%t0)
}

//===----------------------------------------------------------------------===//
// Three-phase split: const const arg at phase -2, const arg at phase -1,
// runtime arg at phase 0. Values thread through adjacent phases.

// CHECK-LABEL: hir.func private @ThreePhase.const2(%a) -> (ctx)
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[A0:%.+]] = hir.coerce_type %a, [[INT]]
// CHECK:      [[TA:%.+]] = hir.type_of [[A0]]
// CHECK:      [[PACK:%.+]] = hir.opaque_pack([[TA]], %a)
// CHECK:      hir.return([[PACK]]) : ({{.*}})

// CHECK-LABEL: hir.func private @ThreePhase.const1(%b, %ctx) -> (ctx)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[B0:%.+]] = hir.coerce_type %b, [[INT]]
// CHECK:      [[TMP:%.+]] = hir.add {{.*}}, [[B0]] :
// CHECK:      [[PACK:%.+]] = hir.opaque_pack({{.*}}, [[TMP]])
// CHECK:      hir.return([[PACK]]) : ({{.*}})

// CHECK-LABEL: hir.func private @ThreePhase.const0(%c, %ctx) -> (result)
// CHECK-NEXT: [[UNPACK:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:      [[INT:%.+]] = hir.int_type
// CHECK:      [[C0:%.+]] = hir.coerce_type %c, [[INT]]
// CHECK:      [[RES:%.+]] = hir.add {{.*}}, [[C0]] :
// CHECK:      hir.return([[RES]]) : ({{.*}})

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ThreePhase(%a: -2, %b: -1, %c: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -2: @ThreePhase.const2
// CHECK:       -1: @ThreePhase.const1
// CHECK:       0: @ThreePhase.const0
// CHECK-LABEL: hir.multiphase_func @ThreePhase.const(first a, last b) -> (ctx)
// CHECK:       @ThreePhase.const2
// CHECK:       @ThreePhase.const1
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
  hir.unified_return (%1) : (%t1b)
}

//===----------------------------------------------------------------------===//
// Three-phase call rewriting: a unified_call to a 3-phase function causes the
// caller to be split into matching phases. Each split function hosts a single
// per-phase call to the corresponding callee split function.

// CHECK-LABEL: hir.func private @ThreePhaseCaller.const2
// CHECK: hir.call @ThreePhase.const2(
// CHECK: hir.opaque_pack(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ThreePhaseCaller.const1
// CHECK: hir.opaque_unpack
// CHECK: hir.call @ThreePhase.const1(
// CHECK: hir.opaque_pack(
// CHECK: hir.return

// CHECK-LABEL: hir.func private @ThreePhaseCaller.const0
// CHECK: hir.opaque_unpack
// CHECK: hir.coerce_type %z,
// CHECK: hir.call @ThreePhase.const0(
// CHECK: hir.return

// CHECK-NOT: hir.unified_func
// CHECK-LABEL: hir.split_func @ThreePhaseCaller(%z: 0) -> (result: 0)
// CHECK:         hir.signature
// CHECK:       -2: @ThreePhaseCaller.const2
// CHECK:       -1: @ThreePhaseCaller.const1
// CHECK:       0: @ThreePhaseCaller.const0
// CHECK-LABEL: hir.multiphase_func @ThreePhaseCaller.const() -> (ctx)
// CHECK:       @ThreePhaseCaller.const2
// CHECK:       @ThreePhaseCaller.const1
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
  hir.unified_return (%r) : (%tr)
}
