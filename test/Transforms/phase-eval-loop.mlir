// RUN: silicon-opt --phase-eval-loop %s | FileCheck %s

//===----------------------------------------------------------------------===//
// No-op: input has no HIR functions or multiphase ops to process.
// The pass should leave the IR unchanged.

// CHECK-LABEL: mir.func private @noop(%x: !si.int) -> (result: !si.int)
// CHECK-NEXT:    mir.return %x
// CHECK:       hir.split_func @noop_fn(%x: 0) -> (result: 0)
// CHECK:         0: @noop
mir.func private @noop(%x: !si.int) -> (result: !si.int) {
  mir.return %x : !si.int
}
hir.split_func @noop_fn(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @noop
]

//===----------------------------------------------------------------------===//
// Single iteration: a zero-arg hir.func is lowered to MIR and interpreted,
// producing an evaluated_func. Then SpecializeFuncs chains the result into the
// next sub-function, which itself gets lowered.

// CHECK-NOT:   hir.func private @add42.0a
// CHECK-NOT:   hir.multiphase_func @add42.0

// CHECK-LABEL: mir.func private @add42.0b(%x: !si.int) -> (result: !si.int)
// CHECK:         %c42_int = mir.constant #si.int<42> : !si.int
// CHECK:         mir.add %x, %c42_int : !si.int
// CHECK:         mir.return

// CHECK:       hir.split_func @add42(%x: 0) -> (result: 0)
// CHECK:         0: @add42.0b

hir.func private @add42.0a() -> (ctx) {
  %0 = hir.int_type
  %1 = hir.constant_int 42
  %2 = hir.opaque_pack(%0, %1)
  %3 = hir.opaque_type
  hir.return %2 : %3
}

hir.func private @add42.0b(%x, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %2 = hir.coerce_type %x, %0
  %3 = hir.add %2, %1 : %0
  hir.return %3 : %0
}

hir.split_func @add42(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @add42.0
]

hir.multiphase_func @add42.0(last x) -> (result) [
  @add42.0a,
  @add42.0b
]

//===----------------------------------------------------------------------===//
// Single iteration: a standalone zero-arg hir.func (no multiphase) is lowered
// and interpreted directly. The result is an evaluated_func with an empty list.

// CHECK-NOT:   hir.func private @standalone.0
// CHECK:       mir.evaluated_func private @standalone.0 []
// CHECK:       hir.split_func @standalone() -> ()
// CHECK:         0: @standalone.0

hir.func private @standalone.0() -> () {
  hir.return
}

hir.split_func @standalone() -> () {
  hir.signature () -> ()
} [
  0: @standalone.0
]

//===----------------------------------------------------------------------===//
// Multi-iteration: @outer calls @inner with a constant argument. In iteration
// 0, @outer.0a and @inner.0 are lowered and @outer.0a is interpreted. In
// iteration 1, SpecializeFuncs chains the evaluated result into @outer.0b,
// which is then lowered. The specialized @inner.1_0 is also produced.

// CHECK-NOT:   hir.func private @outer.0a
// CHECK-NOT:   hir.func private @outer.0b
// CHECK-NOT:   hir.multiphase_func @outer.0

// The lowered inner.0 function remains as a mir.func.
// CHECK:       mir.func private @inner.0(%a: !si.int) -> (ctx: !si.opaque)

// The specialized version of inner.1 with the constant 10 baked in.
// CHECK:       mir.func private @inner.1_0(%b: !si.int) -> (result: !si.int)
// CHECK:         %c10_int = mir.constant #si.int<10> : !si.int
// CHECK:         mir.add %c10_int, %b : !si.int

// The final lowered outer.0b calls the specialized inner.1_0.
// CHECK:       mir.func private @outer.0b(%x: !si.int) -> (result: !si.int)
// CHECK:         mir.call @inner.1_0(%x) : (!si.int) -> !si.int

// CHECK:       hir.split_func @outer(%x: 0) -> (result: 0)
// CHECK:         0: @outer.0b

hir.func private @inner.0(%a) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.coerce_type %a, %0
  %2 = hir.opaque_pack(%0, %1)
  %3 = hir.opaque_type
  hir.return %2 : %3
}

hir.func private @inner.1(%b, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %2 = hir.coerce_type %b, %0
  %3 = hir.add %1, %2 : %0
  hir.return %3 : %0
}

hir.split_func @inner(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0, %0) -> (%0)
} [
  -1: @inner.0,
  0: @inner.1
]

hir.func private @outer.0a() -> (ctx) {
  %0 = hir.int_type
  %1 = hir.constant_int 10
  %2 = hir.opaque_type
  %3 = hir.call @inner.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  %5 = hir.opaque_type
  hir.return %4 : %5
}

hir.func private @outer.0b(%x, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.coerce_type %x, %1
  %3 = hir.opaque_type
  %4 = hir.call @inner.1(%2, %0) : (%1, %3) -> (%1)
  hir.return %4 : %1
}

hir.split_func @outer(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @outer.0
]

hir.multiphase_func @outer.0(last x) -> (result) [
  @outer.0a,
  @outer.0b
]
