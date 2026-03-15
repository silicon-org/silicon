// RUN: silicon-opt --phase-eval-loop %s | FileCheck %s

// Test dependent types where a const argument N determines a uint<N> type.
// This uses pre-split IR to bypass the SplitPhases limitation where
// `uint_type %N` (with block arg width) ends up in the const-phase function,
// preventing HIRToMIR from lowering it. In the pre-split form, the const phase
// packs the raw int value, and the runtime phase computes the uint type from
// the specialized constant.

//===----------------------------------------------------------------------===//
// Example 1: id(const N: int, x: uint<N>) -> uint<N>
//===----------------------------------------------------------------------===//

// CHECK: mir.evaluated_func {{.*}}@main_ex1.0b [#si.int<42> : !si.int]

// id phase -1: receives N, packs it for type computation in the next phase.
hir.func private @id.0(%N) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.int_type
  %1 = hir.coerce_type %N, %0
  %2 = hir.opaque_pack(%1)
  hir.return %2 -> ()
}

// id phase 0: unpacks N, computes uint<N>, coerces x, returns it.
hir.func private @id.1(%x, %ctx) -> (result) {
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

hir.split_func private @id(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1) -> (%1)
} [
  -1: @id.0,
  0: @id.1
]

// main calls id(8, 42).
hir.func private @main_ex1.0a() -> (ctx) {
  %0 = hir.opaque_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.opaque_type
  %3 = hir.call @id.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  hir.return %4 -> ()
}

hir.func private @main_ex1.0b(%ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  %2 = hir.constant_int 8 : %1
  %3 = hir.uint_type %2
  hir.signature (%0) -> (%3)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.constant_int 8 : %1
  %3 = hir.uint_type %2
  %4 = hir.constant_int 42 : %3
  %5 = hir.opaque_type
  %6 = hir.call @id.1(%4, %0) : (%3, %5) -> (%3)
  hir.return %6 -> ()
}

hir.split_func @main_ex1() -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.uint_type %1
  hir.signature () -> (%2)
} [
  0: @main_ex1.0
]

hir.multiphase_func @main_ex1.0() -> (result) [
  @main_ex1.0a,
  @main_ex1.0b
]

//===----------------------------------------------------------------------===//
// Example 8: typed_add(const N: int, a: uint<N>, b: uint<N>) -> uint<N>
//===----------------------------------------------------------------------===//

// CHECK: mir.evaluated_func {{.*}}@main_ex8.0b [#si.int<30> : !si.int]

// typed_add phase -1: receives N, packs it.
hir.func private @typed_add.0(%N) -> (ctx) {
  %0 = hir.int_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %0 = hir.int_type
  %1 = hir.coerce_type %N, %0
  %2 = hir.opaque_pack(%1)
  hir.return %2 -> ()
}

// typed_add phase 0: unpacks N, computes uint<N>, coerces a and b, adds them.
hir.func private @typed_add.1(%a, %b, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %2 = hir.opaque_type
  %3 = hir.opaque_type
  hir.signature (%0, %1, %2) -> (%3)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.uint_type %0
  %2 = hir.coerce_type %a, %1
  %3 = hir.coerce_type %b, %1
  %4 = hir.add %2, %3 : %1
  %5 = hir.coerce_type %4, %1
  hir.return %5 -> (%1)
}

hir.split_func private @typed_add(%N: -1, %a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1, %1) -> (%1)
} [
  -1: @typed_add.0,
  0: @typed_add.1
]

// main calls typed_add(8, 10, 20).
hir.func private @main_ex8.0a() -> (ctx) {
  %0 = hir.opaque_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.opaque_type
  %3 = hir.call @typed_add.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  hir.return %4 -> ()
}

hir.func private @main_ex8.0b(%ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  %2 = hir.constant_int 8 : %1
  %3 = hir.uint_type %2
  hir.signature (%0) -> (%3)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %1 = hir.int_type
  %2 = hir.constant_int 8 : %1
  %3 = hir.uint_type %2
  %4 = hir.constant_int 10 : %3
  %5 = hir.constant_int 20 : %3
  %6 = hir.opaque_type
  %7 = hir.call @typed_add.1(%4, %5, %0) : (%3, %3, %6) -> (%3)
  hir.return %7 -> ()
}

hir.split_func @main_ex8() -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.uint_type %1
  hir.signature () -> (%2)
} [
  0: @main_ex8.0
]

hir.multiphase_func @main_ex8.0() -> (result) [
  @main_ex8.0a,
  @main_ex8.0b
]
