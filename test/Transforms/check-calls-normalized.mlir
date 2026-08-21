// RUN: silicon-opt --normalize-exits --check-calls %s | FileCheck %s
// RUN: silicon-opt --normalize-exits --check-calls %s | silicon-opt --verify-roundtrip

// Call checking consumes exit-normalized input, so a signature that declares
// its types on several paths reaches it as a single exit fed by picks. The
// whole computation, control flow included, is copied to the use.

uir.func private @MultiExit(%c: 0, %x: 0) -> (result: 0) {
  %b = hir.bool_type
  %i = hir.int_type
  uir.if %c {
    uir.signature (%b, %i) -> (%i)
  }
  %u = hir.uint_type %x
  uir.signature (%b, %i) -> (%u)
} {
  %t = hir.int_type
  uir.return %x : %t
}

// CHECK-LABEL: uir.func @CallMultiExit
uir.func @CallMultiExit(%c: 0, %v: 0) -> () {
  %b = hir.bool_type
  %i = hir.int_type
  uir.signature (%b, %i) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CC:%.+]] = hir.coerce_type %c
  // CHECK-NEXT: [[CV:%.+]] = hir.coerce_type %v
  // The copied signature keeps the normalized branch that decides the declared
  // types, with the caller's own values substituted for the arguments.
  // CHECK:      [[IF:%.+]]:4 = uir.if [[CC]]
  // CHECK:      [[PICK:%.+]]:3 = hir.pick [[IF]]#3
  // CHECK:        hir.uint_type [[CV]]
  // CHECK:      [[UA0:%.+]] = hir.unify [[PICK]]#0, {{%.+}}
  // CHECK-NEXT: [[UA1:%.+]] = hir.unify [[PICK]]#1, {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[PICK]]#2, {{%.+}}
  // CHECK-NEXT: uir.call @MultiExit([[CC]], [[CV]]) : ([[UA0]], [[UA1]]) -> ([[UR]]) [0, 0] -> [0]
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r = uir.call @MultiExit(%c, %v) : (%i0, %i1) -> (%i2) [0, 0] -> [0]
  uir.return
}
