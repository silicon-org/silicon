// RUN: silicon-opt --specialize-funcs %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Basic multiphase chaining: an evaluated_func result is chained into the next
// sub-function by replacing the opaque context arg with hir.mir_constant ops.
// The multiphase_func dissolves when only one sub-function remains.

hir.func private @BasicChain.0b(%x, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %2 = hir.add %x, %1 : %0
  %3 = hir.opaque_type
  hir.return %2 : (%0, %3) -> (%0)
}

mir.evaluated_func @BasicChain.0a [#si.opaque<[#si.type<!si.int>, #si.int<42>]> : !si.opaque]

hir.split_func @BasicChain(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @BasicChain.0
]

hir.multiphase_func @BasicChain.0(last x) -> (result) [
  @BasicChain.0a,
  @BasicChain.0b
]

// CHECK-NOT: mir.evaluated_func @BasicChain.0a
// CHECK-NOT: hir.multiphase_func @BasicChain.0

// CHECK-LABEL: hir.func private @BasicChain.0b(%x) -> (result)
// CHECK-NEXT:    %{{.*}} = hir.mir_constant #si.type<!si.int> : !si.type
// CHECK-NEXT:    %{{.*}} = hir.mir_constant #si.int<42> : !si.int
// CHECK-NEXT:    %{{.*}} = hir.add %x, %{{.*}} : %{{.*}}
// CHECK:         hir.return %{{.*}} : (%{{.*}}) -> (%{{.*}})

// CHECK-LABEL: hir.split_func @BasicChain(%x: 0) -> (result: 0)
// CHECK:         0: @BasicChain.0b

//===----------------------------------------------------------------------===//
// Multiphase with three sub-functions: only the first evaluated phase is
// consumed; the remaining two sub-functions stay in a multiphase_func.

hir.func private @ThreePhase.0b(%ctx) -> (ctx) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %2 = hir.opaque_pack(%0, %1)
  %3 = hir.opaque_type
  hir.return %2 : (%3) -> (%3)
}

hir.func private @ThreePhase.0c(%y, %ctx) -> (result) {
  %0:2 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %1 = hir.add %y, %0#1 : %0#0
  %2 = hir.opaque_type
  hir.return %1 : (%0#0, %2) -> (%0#0)
}

mir.evaluated_func @ThreePhase.0a [#si.opaque<[#si.type<!si.int>, #si.int<10>]> : !si.opaque]

hir.split_func @ThreePhase(%y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @ThreePhase.0
]

hir.multiphase_func @ThreePhase.0(last y) -> (result) [
  @ThreePhase.0a,
  @ThreePhase.0b,
  @ThreePhase.0c
]

// CHECK-NOT: mir.evaluated_func @ThreePhase.0a

// CHECK-LABEL: hir.func private @ThreePhase.0b() -> (ctx)
// CHECK:         hir.mir_constant #si.type<!si.int>
// CHECK:         hir.mir_constant #si.int<10>

// CHECK-LABEL: hir.func private @ThreePhase.0c(%y, %ctx) -> (result)

// CHECK-LABEL: hir.multiphase_func @ThreePhase.0(first y) -> (result)
// CHECK:         @ThreePhase.0b
// CHECK:         @ThreePhase.0c

//===----------------------------------------------------------------------===//
// No opaque_unpack: when the context arg has no opaque_unpack user, the pass
// inserts a single hir.mir_constant for the whole opaque and relies on
// canonicalization to expand it later.

hir.func private @NoUnpack.0b(%x, %ctx) -> (result) {
  %0 = hir.coerce_type %ctx, %x
  hir.return %0 : (%x, %x) -> (%x)
}

mir.evaluated_func @NoUnpack.0a [#si.opaque<[#si.int<99>]> : !si.opaque]

hir.split_func @NoUnpack(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @NoUnpack.0
]

hir.multiphase_func @NoUnpack.0(last x) -> (result) [
  @NoUnpack.0a,
  @NoUnpack.0b
]

// CHECK-NOT: mir.evaluated_func @NoUnpack.0a
// CHECK-NOT: hir.multiphase_func @NoUnpack.0

// CHECK-LABEL: hir.func private @NoUnpack.0b(%x) -> (result)
// CHECK:         hir.mir_constant #si.opaque<[#si.int<99> : !si.int]>
// CHECK:         hir.coerce_type
// CHECK:         hir.return {{.*}} : ({{.*}}) -> ({{.*}})

// CHECK-LABEL: hir.split_func @NoUnpack(%x: 0) -> (result: 0)
// CHECK:         0: @NoUnpack.0b

//===----------------------------------------------------------------------===//
// Transitive specialization: after chaining, if a call passes an
// hir.mir_constant with an opaque attribute as the last argument, the callee is
// cloned and specialized, and the call drops the last argument.

hir.func private @callee(%a, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.add %a, %0 : %0
  %2 = hir.opaque_type
  hir.return %1 : (%0, %2) -> (%0)
}

// This function calls @callee with an hir.mir_constant opaque as the last arg.
// After transitive specialization, @callee is cloned, specialized, and the call
// is updated to drop the last arg.
hir.func private @TransSpec.0b(%x, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %opaque = hir.mir_constant #si.opaque<[#si.int<7>]> : !si.opaque
  %3 = hir.call @callee(%x, %opaque) : (%0, %0) -> (%0)
  %4 = hir.opaque_type
  hir.return %3 : (%0, %4) -> (%0)
}

mir.evaluated_func @TransSpec.0a [#si.opaque<[#si.type<!si.int>, #si.int<5>]> : !si.opaque]

hir.split_func @TransSpec(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @TransSpec.0
]

hir.multiphase_func @TransSpec.0(last x) -> (result) [
  @TransSpec.0a,
  @TransSpec.0b
]

// CHECK-NOT: mir.evaluated_func @TransSpec.0a
// CHECK-NOT: hir.multiphase_func @TransSpec.0

// The cloned and specialized callee has the opaque context arg removed.
// CHECK-LABEL: hir.func private @callee_0(%a) -> (result)
// CHECK:         hir.mir_constant #si.int<7>
// CHECK:         hir.add

// The specialized function has its context expanded.
// CHECK-LABEL: hir.func private @TransSpec.0b(%x) -> (result)
// CHECK:         hir.mir_constant #si.type<!si.int>
// CHECK:         hir.mir_constant #si.int<5>
// CHECK:         hir.call @callee_0(%x)
// CHECK:         hir.return {{.*}} : ({{.*}}) -> ({{.*}})

//===----------------------------------------------------------------------===//
// Single sub-function multiphase with evaluated result: when a multiphase_func
// has only one sub-function and it has been evaluated, the evaluated_func is
// erased and the multiphase_func is erased.

mir.evaluated_func @SingleSub.0a [#si.opaque<[#si.int<1>]> : !si.opaque]

hir.split_func @SingleSub() -> () {
  hir.signature () -> ()
} [
  0: @SingleSub.0
]

hir.multiphase_func @SingleSub.0() -> () [
  @SingleSub.0a
]

// CHECK-NOT: mir.evaluated_func @SingleSub.0a
// CHECK-NOT: hir.multiphase_func @SingleSub.0

//===----------------------------------------------------------------------===//
// Two calls to the same callee with distinct opaque constants produce two
// distinct specializations.

hir.func private @callee2(%a, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.add %a, %0 : %0
  %2 = hir.opaque_type
  hir.return %1 : (%0, %2) -> (%0)
}

hir.func private @TwoCalls.0b(%x, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %opaqueA = hir.mir_constant #si.opaque<[#si.int<7>]> : !si.opaque
  %opaqueB = hir.mir_constant #si.opaque<[#si.int<8>]> : !si.opaque
  %3 = hir.call @callee2(%x, %opaqueA) : (%0, %0) -> (%0)
  %4 = hir.call @callee2(%x, %opaqueB) : (%0, %0) -> (%0)
  %5 = hir.opaque_type
  hir.return %4 : (%0, %5) -> (%0)
}

mir.evaluated_func @TwoCalls.0a [#si.opaque<[#si.type<!si.int>, #si.int<5>]> : !si.opaque]

hir.split_func @TwoCalls(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @TwoCalls.0
]

hir.multiphase_func @TwoCalls.0(last x) -> (result) [
  @TwoCalls.0a,
  @TwoCalls.0b
]

// CHECK-LABEL: hir.func private @callee2_{{[0-9]+}}(%a) -> (result)
// CHECK:         hir.mir_constant #si.int<8>
// CHECK-LABEL: hir.func private @callee2_{{[0-9]+}}(%a) -> (result)
// CHECK:         hir.mir_constant #si.int<7>

//===----------------------------------------------------------------------===//
// Two calls to the same callee with identical opaque constants reuse a single
// specialization.

hir.func private @callee3(%a, %ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.add %a, %0 : %0
  %2 = hir.opaque_type
  hir.return %1 : (%0, %2) -> (%0)
}

hir.func private @Dedup.0b(%x1, %x2, %ctx) -> (result) {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %opaqueA = hir.mir_constant #si.opaque<[#si.int<7>]> : !si.opaque
  %opaqueB = hir.mir_constant #si.opaque<[#si.int<7>]> : !si.opaque
  %3 = hir.call @callee3(%x1, %opaqueA) : (%0, %0) -> (%0)
  %4 = hir.call @callee3(%x2, %opaqueB) : (%0, %0) -> (%0)
  %5 = hir.opaque_type
  hir.return %4 : (%0, %0, %5) -> (%0)
}

mir.evaluated_func @Dedup.0a [#si.opaque<[#si.type<!si.int>, #si.int<5>]> : !si.opaque]

hir.split_func @Dedup(%x1: 0, %x2: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0, %0) -> (%0)
} [
  0: @Dedup.0
]

hir.multiphase_func @Dedup.0(last x1, last x2) -> (result) [
  @Dedup.0a,
  @Dedup.0b
]

// Only one specialization of callee3 should be produced.
// CHECK-LABEL: hir.func private @callee3_{{[0-9]+}}(%a) -> (result)
// CHECK:         hir.mir_constant #si.int<7>
// CHECK-NOT:   hir.func private @callee3_{{[0-9]+}}
