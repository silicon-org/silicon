// RUN: silicon-opt --sort-phases --thread-phases --split-funcs %s | FileCheck %s
// RUN: silicon-opt --sort-phases --thread-phases --split-funcs %s | silicon-opt --verify-roundtrip

// Sorting, threading, and splitting compose end to end: sorting produces the
// jumping sorted form, threading densifies it into the fully threaded contract,
// and splitting emits per-phase `hir.func` ops plus the structural
// `uir.split_func`. These pre-sort inputs exercise all three passes back to
// back on more than one function.

//===----------------------------------------------------------------------===//
// A three-phase chain whose phase -2 value skips to phase 0: threading routes
// it through the materialized gap phase -1 (which carries a receiving `hir.var`
// write), and splitting turns each phase into a context-threaded func.

// The signature's type for the phase -2 argument %a is computed one phase
// earlier still, so the chain reaches phase -3 and heads an eager chain.
// CHECK-LABEL: hir.func private @"Chain.-3"() -> (ctx) {
// CHECK:         [[T:%.+]] = hir.int_type
// CHECK:         hir.opaque_pack([[T]])

// CHECK-LABEL: hir.func private @"Chain.-2.tail"(%a, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK:         [[SQ:%.+]] = hir.mul %a, %a
// CHECK:         hir.opaque_pack([[SQ]])

// CHECK-LABEL: hir.func private @"Chain.-1"(%ctx) -> (ctx) {
// CHECK:       } {
// CHECK:         [[U:%.+]] = hir.opaque_unpack %ctx
// CHECK:         [[XTY:%.+]] = hir.int_type
// CHECK:         hir.write_var {{%.+}}, [[U]]
// CHECK:         hir.opaque_pack([[XTY]], [[U]])

// CHECK-LABEL: hir.func private @Chain.0(%x, %ctx) -> (result) {
// CHECK:       } {
// CHECK:         [[U:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:         hir.add [[U]]#1, %x

// CHECK-LABEL: hir.eager_func @"Chain.-2"(a) -> (ctx) [
// CHECK-NEXT:    @"Chain.-3",
// CHECK-NEXT:    @"Chain.-2.tail"
// CHECK-NEXT:  ]

// CHECK-LABEL: uir.split_func @Chain(%a: -2, %x: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    @"Chain.-2",
// CHECK-NEXT:    @"Chain.-1",
// CHECK-NEXT:    @Chain.0
// CHECK-NEXT:  ]
uir.func @Chain(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  uir.expr pin -1 {
    %v = hir.var : %t
    hir.write_var %v, %sq : %t
    uir.yield
  }
  %r = hir.add %sq, %x : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A caller: sorting decomposes the `uir.call` into per-phase `uir.phase_call`
// entries, and splitting rewrites each into an `hir.call` to the callee's
// concrete per-phase func. The callee context threads across the caller's own
// phase boundary.

// CHECK-LABEL: hir.func private @"Caller.-1.tail"(%x, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK:         [[CC:%.+]] = hir.call @"Chain2.-1"(%x)
// CHECK:         hir.opaque_pack({{%.+}}, [[CC]])

// CHECK-LABEL: hir.func private @Caller.0(%y, %ctx) -> (result) {
// CHECK:       } {
// CHECK:         [[CC:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:         hir.call @Chain2.0(%y, [[CC]]#1)

// CHECK-LABEL: uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.call @Chain2(%x, %y) : (%t, %t) -> (%t) [-1, 0] -> [0]
  uir.return %r : %t
}

// CHECK-LABEL: uir.split_func @Chain2(%a: -1, %b: 0) -> (result: 0) {
uir.func @Chain2(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %sq = hir.mul %a, %a : %t
  %r = hir.add %sq, %b : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A precompiled callee: `@Lib` is already split, so only its `uir.split_func`
// interface and per-phase funcs exist, with no `uir.func` implementation for
// sorting to take custody of. Decomposition reads the declared span off that
// interface exactly as it does off a freshly snapshotted one, and splitting
// binds each entry through its existing phase list.

// CHECK-LABEL: hir.func private @"LibCaller.-1.tail"(%x, %ctx) -> (ctx) {
// CHECK:       } {
// CHECK:         [[CC:%.+]] = hir.call @"Lib.-1"(%x)
// CHECK:         hir.opaque_pack({{%.+}}, [[CC]])

// CHECK-LABEL: hir.func private @LibCaller.0(%y, %ctx) -> (result) {
// CHECK:       } {
// CHECK:         [[CC:%.+]]:2 = hir.opaque_unpack %ctx
// CHECK:         hir.call @Lib.0(%y, [[CC]]#1)

// CHECK-LABEL: uir.split_func @LibCaller(%x: -1, %y: 0) -> (result: 0) {
uir.func @LibCaller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %r = uir.call @Lib(%x, %y) : (%t, %t) -> (%t) [-1, 0] -> [0]
  uir.return %r : %t
}

// The precompiled library, left untouched by all three passes.
// CHECK-LABEL: uir.split_func @Lib(%a: -1, %b: 0) -> (result: 0) {
// CHECK:       } [
// CHECK-NEXT:    @"Lib.-1",
// CHECK-NEXT:    @Lib.0
// CHECK-NEXT:  ]
uir.split_func @Lib(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} [
  @"Lib.-1",
  @Lib.0
]

hir.func private @"Lib.-1"(%a) -> (ctx) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  hir.signature (%1) -> (%0)
} {
  %0 = hir.opaque_type
  %1 = hir.int_type
  %2 = hir.mul %a, %a : %1
  %3 = hir.opaque_pack(%1, %2)
  hir.return %3 : %0
}

hir.func private @Lib.0(%b, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%1#0, %0) -> (%1#0)
} {
  %0:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %1 = hir.int_type
  %2 = hir.add %0#1, %b : %1
  hir.return %2 : %1
}
