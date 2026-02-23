// RUN: silicon-opt --split-phases --split-input-file --verify-diagnostics %s

// Phase mismatch: passing phase-0 func args as phase -2 and -1 callee args.

hir.unified_func @ThreePhase [-2, -1, 0] -> [0] attributes {argNames = ["a", "b", "c"]} {
^bb0(%a: !hir.any, %b: !hir.any, %c: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
^bb0(%a: !hir.any, %b: !hir.any, %c: !hir.any):
  %0 = hir.binary %a, %b
  %1 = hir.binary %0, %c
  hir.unified_return %1
}

hir.unified_func @BadCaller [0, 0, 0] -> [0] attributes {argNames = ["x", "y", "z"]} {
^bb0(%x: !hir.any, %y: !hir.any, %z: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
^bb0(%x: !hir.any, %y: !hir.any, %z: !hir.any):
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %t3 = hir.inferrable
  // expected-error @below {{call argument requires phase -2 but value is only available at phase 0}}
  // expected-error @below {{call argument requires phase -1 but value is only available at phase 0}}
  %r = hir.unified_call @ThreePhase(%x, %y, %z) : (%t0, %t1, %t2) -> (%t3) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  hir.unified_return %r
}
