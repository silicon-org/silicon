// RUN: silicon-opt --phase-eval-loop --verify-diagnostics %s

//===----------------------------------------------------------------------===//
// No-progress error: the loop enters because there are actionable ops, but
// the sub-pipeline cannot make progress. Here @good.0 is evaluated in
// iteration 0, but @stuck.0 has an unresolvable coerce_type (block arg as
// type operand) so HIRToMIR skips it. In iteration 1 the counts are unchanged
// and the pass reports an error.

// expected-error @below {{compiler bug: phase eval loop made no progress in iteration 1}}
// expected-note @below {{PhaseEvalLoopPass::runOnOperation()}}
module {
  hir.func private @good.0() -> () {
    hir.return : () -> ()
  }

  hir.split_func @good() -> () {
    hir.signature () -> ()
  } [
    0: @good.0
  ]

  // expected-note @below {{hir.func @stuck.0 could not be lowered to MIR}}
  hir.func private @stuck.0(%T) -> (result) {
    %0 = hir.constant_int 0
    %1 = hir.coerce_type %0, %T
    hir.return %1 : () -> (%T)
  }

  hir.split_func @stuck(%T: 0) -> (result: 0) {
    %0 = hir.int_type
    hir.signature (%0) -> (%0)
  } [
    0: @stuck.0
  ]
}
