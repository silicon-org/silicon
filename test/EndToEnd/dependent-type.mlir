// RUN: silc %s | FileCheck %s
// XFAIL: *

// Test dependent types in function signatures: the result type of `identity`
// depends on its first argument T. When called with T = int, the compiler
// resolves the result type to !mir.int through the full pipeline:
//   check-calls → infer-types → split-phases → phase-eval-loop

// CHECK-LABEL: mir.func private @main.0b() -> (r0: !mir.int, r1: !mir.unit)
// CHECK-NEXT:    %c42_int = mir.constant #mir.int<42> : !mir.int
// CHECK-NEXT:    %unit = mir.constant #mir.unit : !mir.unit
// CHECK-NEXT:    mir.return %c42_int, %unit : !mir.int, !mir.unit

hir.unified_func @identity(%T: -1, %x: 0) -> (result: 0) {
  %type_type = hir.type_type
  hir.unified_signature (%type_type, %T) -> (%T)
} {
  hir.unified_return (%x) : (%T)
}

hir.unified_func @main() -> (r0: 0, r1: 0) {
  %0 = hir.int_type
  %1 = hir.unit_type
  hir.unified_signature () -> (%0, %1)
} {
  %type_type = hir.type_type

  // Call @identity with an int argument.
  %int_type = hir.int_type
  %c42 = hir.constant_int 42
  %r0 = hir.unified_call @identity(%int_type, %c42) : (%type_type, %int_type) -> (%int_type) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]

  // Call @identity with a unit argument.
  %unit_type = hir.unit_type
  %cunit = hir.constant_unit
  %r1 = hir.unified_call @identity(%unit_type, %cunit) : (%type_type, %unit_type) -> (%unit_type) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]

  hir.unified_return (%r0, %r1) : (%int_type, %unit_type)
}
