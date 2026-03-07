// RUN: silicon-opt --specialize-funcs --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error: evaluated_func has wrong number of results (expected exactly 1 opaque
// pack).

hir.func private @BadCount.0b(%ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  hir.return %0 : %0
}

// expected-error @below {{compiler bug: evaluated_func has 2 results, expected 1 (opaque pack)}}
// expected-note @below {{on line}}
mir.evaluated_func @BadCount.0a [#si.int<1> : !si.int, #si.int<2> : !si.int]

hir.split_func @BadCount() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} [
  0: @BadCount.0
]

hir.multiphase_func @BadCount.0() -> (result) [
  @BadCount.0a,
  @BadCount.0b
]

// -----

//===----------------------------------------------------------------------===//
// Error: evaluated_func result is not an opaque attribute.

hir.func private @NotOpaque.0b(%ctx) -> (result) {
  %0 = hir.opaque_unpack %ctx : !hir.any
  hir.return %0 : %0
}

// expected-error @below {{compiler bug: evaluated_func result is not an opaque attribute}}
// expected-note @below {{on line}}
mir.evaluated_func @NotOpaque.0a [#si.int<42> : !si.int]

hir.split_func @NotOpaque() -> (result: 0) {
  %0 = hir.int_type
  hir.signature () -> (%0)
} [
  0: @NotOpaque.0
]

hir.multiphase_func @NotOpaque.0() -> (result) [
  @NotOpaque.0a,
  @NotOpaque.0b
]
