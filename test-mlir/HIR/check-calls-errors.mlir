// RUN: silicon-opt --check-calls --split-input-file --verify-diagnostics %s

// Trivial recursion
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unchecked_func @foo {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unchecked_call @foo() : () -> ()
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}

// -----

// Multiple levels
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unchecked_func @foo {
  // expected-remark @below {{called through `bar`}}
  hir.unchecked_call @bar() : () -> ()
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}
hir.unchecked_func @bar {
  // expected-remark @below {{called through `gux`}}
  hir.unchecked_call @gux() : () -> ()
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}
hir.unchecked_func @gux {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unchecked_call @foo() : () -> ()
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}
