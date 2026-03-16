// RUN: silicon-opt --phase-eval-loop %s | FileCheck %s

// Test three-phase dependent type threading: phase -2 computes a value, phase
// -1 uses it to construct a uint<N> type and coerce a const value, phase 0
// uses the const value together with a runtime value. This uses pre-split IR
// to bypass the SplitPhases limitation.
//
// The Silicon source this corresponds to:
//   fn foo(const const N: int, const x: uint<N>, y: uint<N>) -> uint<N> {
//     x + y
//   }
//   pub fn main() -> uint<8> { foo(8, 42, 100) }
//
// Corresponds to Example 7 in docs/design/cross-phase-types.md.

// CHECK: mir.evaluated_func {{.*}}@main.0c [#si.uint<8, 142> : !si.uint<8>]

//===----------------------------------------------------------------------===//
// foo: three-phase function
//===----------------------------------------------------------------------===//

// foo phase -2: receives N (int), packs it.
hir.func private @foo.0(%N) -> (ctx) {
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

// foo phase -1: receives x, unpacks N, computes uint<N>, coerces x. Packs
// both N and x into context so that phase 0 can reconstruct the type.
hir.func private @foo.1(%x, %ctx) -> (ctx) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %2 = hir.opaque_type
  hir.signature (%0, %1) -> (%2)
} {
  %N = hir.opaque_unpack %ctx : !hir.any
  %T = hir.uint_type %N
  %x_coerced = hir.coerce_type %x, %T
  %packed = hir.opaque_pack(%N, %x_coerced)
  %0 = hir.opaque_type
  hir.return %packed -> (%0)
}

// foo phase 0: receives y, unpacks N and x from context, adds x + y.
hir.func private @foo.2(%y, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %2 = hir.opaque_type
  hir.signature (%0, %1) -> (%2)
} {
  %N, %x = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %T = hir.uint_type %N
  %y_coerced = hir.coerce_type %y, %T
  %sum = hir.add %x, %y_coerced : %T
  hir.return %sum -> (%T)
}

hir.split_func private @foo(%N: -2, %x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  hir.signature (%0, %1, %1) -> (%1)
} [
  -2: @foo.0,
  -1: @foo.1,
  0: @foo.2
]

//===----------------------------------------------------------------------===//
// main: calls foo(8, 42, 100)
//===----------------------------------------------------------------------===//

// main phase 0a: calls foo.0(8), packs context.
hir.func private @main.0a() -> (ctx) {
  %0 = hir.opaque_type
  hir.signature () -> (%0)
} {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.opaque_type
  %3 = hir.call @foo.0(%1) : (%0) -> (%2)
  %4 = hir.opaque_pack(%3)
  %5 = hir.opaque_type
  hir.return %4 -> (%5)
}

// main phase 0b: calls foo.1(42, ctx). The literal 42 is typed as uint<N>
// where N is extracted from the context (the result of foo.0). Packs both N
// and the foo.1 result into context so that main.0c can reconstruct the type.
hir.func private @main.0b(%ctx) -> (ctx) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0) -> (%1)
} {
  %foo_ctx = hir.opaque_unpack %ctx : !hir.any
  %N = hir.opaque_unpack %foo_ctx : !hir.any
  %T = hir.uint_type %N
  %c42 = hir.constant_int 42 : %T
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  %foo1_result = hir.call @foo.1(%c42, %foo_ctx) : (%T, %0) -> (%1)
  %packed = hir.opaque_pack(%N, %foo1_result)
  %2 = hir.opaque_type
  hir.return %packed -> (%2)
}

// main phase 0c: calls foo.2(100, ctx). Unpacks N from context to construct
// uint<N> for the literal 100. Passes the foo.1 result as foo.2's context.
hir.func private @main.0c(%ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.int_type
  %2 = hir.constant_int 8 : %1
  %3 = hir.uint_type %2
  hir.signature (%0) -> (%3)
} {
  %N, %foo1_ctx = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %T = hir.uint_type %N
  %c100 = hir.constant_int 100 : %T
  %0 = hir.opaque_type
  %result = hir.call @foo.2(%c100, %foo1_ctx) : (%T, %0) -> (%T)
  hir.return %result -> (%T)
}

hir.split_func @main() -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.uint_type %1
  hir.signature () -> (%2)
} [
  0: @main.0
]

hir.multiphase_func @main.0() -> (result) [
  @main.0a,
  @main.0b,
  @main.0c
]
