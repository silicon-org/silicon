// RUN: silicon-opt --check-calls --split-input-file --verify-diagnostics %s

// Trivial recursion
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unified_func @foo() -> () {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unified_call @foo() : () -> () () -> () [] -> []
  hir.signature () -> ()
} {
  hir.return : () -> ()
}

// -----

// Multiple levels
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unified_func @foo() -> () {
  // expected-remark @below {{called through `bar`}}
  hir.unified_call @bar() : () -> () () -> () [] -> []
  hir.signature () -> ()
} {
  hir.return : () -> ()
}
hir.unified_func @bar() -> () {
  // expected-remark @below {{called through `gux`}}
  hir.unified_call @gux() : () -> () () -> () [] -> []
  hir.signature () -> ()
} {
  hir.return : () -> ()
}
hir.unified_func @gux() -> () {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unified_call @foo() : () -> () () -> () [] -> []
  hir.signature () -> ()
} {
  hir.return : () -> ()
}
