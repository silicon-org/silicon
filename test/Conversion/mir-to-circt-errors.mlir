// RUN: silicon-opt --lower-mir-to-circt --split-input-file --verify-diagnostics %s

// Multi-result mir.if cannot be lowered to comb.mux.
mir.func @multi_result_if(%cond: !si.uint<1>, %a: !si.int, %b: !si.int) -> (x: !si.int, y: !si.int) {
  // expected-error @below {{mir.if with multiple results cannot be lowered to comb.mux; only single-result is supported}}
  // expected-error @below {{failed to legalize operation 'mir.if'}}
  %0, %1 = mir.if %cond : !si.uint<1>, !si.int, !si.int {
    mir.yield %a, %b : !si.int, !si.int
  } else {
    mir.yield %b, %a : !si.int, !si.int
  }
  mir.return %0, %1 : !si.int, !si.int
}
