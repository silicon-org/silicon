// RUN: silicon-opt --phase-eval-loop %s | FileCheck %s
// XFAIL: *

// Test three-phase dependent type threading: phase -2 computes a value, phase
// -1 uses it to construct a uint<N> type, phase 0 uses the result. This uses
// pre-split IR to bypass the SplitPhases limitation.

// Still XFAIL because the caller main.0b passes an int-typed literal where the
// specialized callee expects uint<8>. Fixing this requires coercion at the
// mir.call boundary or updating the caller literal type during specialization.
//
// Corresponds to Example 7 in docs/design/cross-phase-types.md.

// CHECK: mir.evaluated_func {{.*}}@main.0c [#si.int<42> : !si.int]

// triple phase -2: receives N (int), packs it.
hir.func private @triple.0(%N) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.int_type
  %1 = hir.coerce_type %N, %0
  %2 = hir.opaque_pack(%1)
  %3 = hir.opaque_type
  hir.return %2 -> (%3)
}

// triple phase -1: receives x, unpacks N, computes uint<N>, coerces x.
hir.func private @triple.1(%x, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %2 = hir.opaque_type
  hir.signature (%0, %1) -> (%2)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.uint_type %0
  %2 = hir.coerce_type %x, %1
  hir.return %2 -> (%1)
}

hir.split_func private @triple(%N: -2, %x: -1) -> (result: -1) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1) -> (%1)
} [
  -2: @triple.0,
  -1: @triple.1
]

// main phase 0a: calls triple.0(8), packs context.
hir.func private @main.0a() -> (ctx) {
  %0 = hir.opaque_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.opaque_type
  %3 = hir.call @triple.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  %5 = hir.opaque_type
  hir.return %4 -> (%5)
}

// main phase 0b: calls triple.1(42, ctx).
hir.func private @main.0b(%ctx) -> (ctx) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.constant_int 42 : %1
  %3 = hir.opaque_type
  %4 = hir.opaque_type
  %5 = hir.call @triple.1(%2, %0) : (%1, %3) -> (%4)
  %6 = hir.opaque_pack(%5)
  %7 = hir.opaque_type
  hir.return %6 -> (%7)
}

// main phase 0c: unpacks the result.
hir.func private @main.0c(%ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.type_of %0
  hir.return %0 -> (%1)
}

hir.split_func @main() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} [
  0: @main.0
]

hir.multiphase_func @main.0() -> (result) [
  @main.0a,
  @main.0b,
  @main.0c
]
