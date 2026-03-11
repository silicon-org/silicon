// RUN: silicon-opt --specialize-funcs --split-input-file --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Error: cycle in the call graph during transitive specialization. Function
// CycleA calls CycleB with an opaque constant, and CycleB calls CycleA with
// an opaque constant, creating unbounded mutual recursion.

// expected-error @below {{compiler bug: cycle detected during transitive specialization of @CycleA}}
// expected-note @below {{on line}}
hir.func private @CycleA(%x, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0, %1) -> (%0)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %opaque = hir.mir_constant #si.opaque<[#si.int<1>]> : !si.opaque
  %1 = hir.call @CycleB(%x, %opaque) : (%0, %0) -> (%0)
  %2 = hir.opaque_type
  hir.return %1 -> (%0)
}

hir.func private @CycleB(%x, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0, %1) -> (%0)
} {
  %0 = hir.opaque_unpack %ctx : !hir.any
  %opaque = hir.mir_constant #si.opaque<[#si.int<2>]> : !si.opaque
  %1 = hir.call @CycleA(%x, %opaque) : (%0, %0) -> (%0)
  %2 = hir.opaque_type
  hir.return %1 -> (%0)
}

hir.func private @CycleCaller.0b(%x, %ctx) -> (result) {
  %0 = hir.opaque_type
  %1 = hir.opaque_type
  hir.signature (%0, %1) -> (%0)
} {
  %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %opaque = hir.mir_constant #si.opaque<[#si.int<99>]> : !si.opaque
  %2 = hir.call @CycleA(%x, %opaque) : (%0, %0) -> (%0)
  %3 = hir.opaque_type
  hir.return %2 -> (%0)
}

mir.evaluated_func @CycleCaller.0a [#si.opaque<[#si.type<!si.int>, #si.int<5>]> : !si.opaque]

hir.split_func @CycleCaller(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.signature (%0) -> (%0)
} [
  0: @CycleCaller.0
]

hir.multiphase_func @CycleCaller.0(last x) -> (result) [
  @CycleCaller.0a,
  @CycleCaller.0b
]
