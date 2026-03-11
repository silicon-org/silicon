// RUN: silicon-opt --phase-eval-loop="max-iterations=1" --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// Max-iterations exhaustion: the multiphase_func @add42.0 needs two loop
// iterations to fully converge (iteration 0 lowers and evaluates @add42.0a,
// iteration 1 lowers the specialized @add42.0b). With max-iterations=1, only
// iteration 0 runs and the pass reports failure.

// expected-error @below {{phase evaluation did not converge after 1 iterations}}
module {
  hir.func private @add42.0a() -> (ctx) {
    %0 = hir.opaque_type
    hir.signature () -> (%0)
  } {
    %0 = hir.int_type
    %1 = hir.constant_int 42 : %0
    %2 = hir.opaque_pack(%0, %1)
    %3 = hir.opaque_type
    hir.return %2 -> (%3)
  }

  // expected-note @below {{hir.func @add42.0b still present}}
  hir.func private @add42.0b(%x, %ctx) -> (result) {
    %0 = hir.int_type
    %1 = hir.opaque_type
    hir.signature (%0, %1) -> (%0)
  } {
    %0, %1 = hir.opaque_unpack %ctx : !hir.any, !hir.any
    %2 = hir.coerce_type %x, %0
    %3 = hir.add %2, %1 : %0
    %4 = hir.opaque_type
    hir.return %3 -> (%0)
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
}
