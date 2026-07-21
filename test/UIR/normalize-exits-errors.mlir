// RUN: silicon-opt %s --normalize-exits --split-input-file --verify-diagnostics

// An exit inside a region op that is not structured UIR control flow cannot
// be encoded through.
uir.func @ReturnInPick(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{'hir.pick' op holds a function exit but is not a structured control flow op that exit normalization can encode through}}
  %v = hir.pick %c : !hir.any [0 (): {
    uir.return %x : %t
  }]
  uir.return %v : %t
}

// -----

// The same rule holds in the signature region: a signature exit inside a
// region op that is not structured UIR control flow cannot be encoded
// through.
uir.func @SignatureInPick(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  // expected-error @below {{'hir.pick' op holds a function exit but is not a structured control flow op that exit normalization can encode through}}
  %v = hir.pick %c : !hir.any [0 (): {
    uir.signature (%0, %0) -> (%0)
  }]
  uir.signature (%0, %0) -> (%v)
} {
  %t = hir.int_type
  uir.return %x : %t
}
