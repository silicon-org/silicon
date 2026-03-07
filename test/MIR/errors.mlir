// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// YieldOp must be inside an IfOp
mir.func @yield_outside() -> () {
  // expected-error @below {{'mir.yield' op expects parent op 'mir.if'}}
  mir.yield
}
