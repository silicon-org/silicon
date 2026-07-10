// RUN: silicon-opt --split-phases2="sort-only=true" --split-input-file --verify-diagnostics %s

// A call whose callee has not been split cannot be decomposed for sorting.
uir.func @unsplit(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.return %x : %t
}

uir.func @NoWitness(%x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{callee @unsplit has no uir.split_func witness}}
  %r = uir.call @unsplit(%x) : (%t) -> (%t) (!hir.any) -> !hir.any [0] -> [0]
  uir.return %r : %t
}
