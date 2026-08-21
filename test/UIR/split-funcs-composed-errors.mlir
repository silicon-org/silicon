// RUN: silicon-opt --split-input-file --verify-diagnostics --sort-phases --thread-phases --split-funcs %s

// Errors the splitter reports about calls it cannot bind, reached the way the
// compiler reaches them: from a `uir.call` in unsorted input, decomposed by
// sorting, rather than from hand-written `uir.phase_call` entries.

// Calling a declaration whose implementation was never compiled: sorting
// decomposes the call against the declared span like any other, and splitting
// finds no phase list to bind the entries through.
uir.split_func @never_compiled(%a: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} []

uir.func @calls_never_compiled(%x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int) -> (%int)
} {
  %t = hir.int_type
  // expected-error @below {{callee @never_compiled has no phase list; its implementation is not available}}
  %r = uir.call @never_compiled(%x) : (%t) -> (%t) [0] -> [0]
  uir.return %r : %t
}
