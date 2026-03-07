// RUN: silicon-opt --split-phases --split-input-file --verify-diagnostics %s

// Phase mismatch: passing phase-0 func args as phase -2 and -1 callee args.

hir.unified_func @ThreePhase(%a: -2, %b: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
  %ta = hir.type_of %a
  %tb = hir.type_of %b
  %t0 = hir.unify %ta, %tb
  %0 = hir.add %a, %b : %t0
  %t0b = hir.type_of %0
  %tc = hir.type_of %c
  %t1 = hir.unify %t0b, %tc
  %1 = hir.add %0, %c : %t1
  %t1b = hir.type_of %1
  hir.unified_return %1 : %t1b
}

hir.unified_func @BadCaller(%x: 0, %y: 0, %z: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0, %0) -> (%0)
} {
  %t0 = hir.inferrable
  %t1 = hir.inferrable
  %t2 = hir.inferrable
  %t3 = hir.inferrable
  // expected-error @below {{call argument requires phase -2 but value is only available at phase 0}}
  // expected-error @below {{call argument requires phase -1 but value is only available at phase 0}}
  %r = hir.unified_call @ThreePhase(%x, %y, %z) : (%t0, %t1, %t2) -> (%t3) (!hir.any, !hir.any, !hir.any) -> !hir.any [-2, -1, 0] -> [0]
  %tr = hir.type_of %r
  hir.unified_return %r : %tr
}

// -----

// Phase back-propagation failure: ExprOp captures a phase-0 block arg, so it
// cannot be pulled to phase -1.

hir.unified_func @ConstArgCallee(%key: -1, %a: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %t = hir.type_of %key
  %t2 = hir.type_of %a
  %t3 = hir.unify %t, %t2
  %r = hir.add %key, %a : %t3
  hir.unified_return %r : %t3
}

hir.unified_func @PullFails(%x: 0, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %val = hir.expr 0 : !hir.any {
    %t0 = hir.type_of %x
    %t1 = hir.type_of %y
    %t = hir.unify %t0, %t1
    %r = hir.add %x, %y : %t
    hir.yield %r : !hir.any
  }
  %vt = hir.type_of %val
  %yt = hir.type_of %y
  %rt = hir.inferrable
  // expected-error @below {{call argument requires phase -1 but value is only available at phase 0}}
  %r = hir.unified_call @ConstArgCallee(%val, %y) : (%vt, %yt) -> (%rt) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  %rrt = hir.type_of %r
  hir.unified_return %r : %rrt
}
