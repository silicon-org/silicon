// RUN: silicon-opt --check-calls --split-input-file --verify-diagnostics %s

// Trivial recursion
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unified_func @foo [] -> [] attributes {argNames = []} {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unified_call @foo() : () -> () () -> () [] -> []
  hir.unified_signature () -> ()
} {
  hir.unified_return
}

// -----

// Multiple levels
// expected-error @below {{signature of `foo` cannot call itself}}
hir.unified_func @foo [] -> [] attributes {argNames = []} {
  // expected-remark @below {{called through `bar`}}
  hir.unified_call @bar() : () -> () () -> () [] -> []
  hir.unified_signature () -> ()
} {
  hir.unified_return
}
hir.unified_func @bar [] -> [] attributes {argNames = []} {
  // expected-remark @below {{called through `gux`}}
  hir.unified_call @gux() : () -> () () -> () [] -> []
  hir.unified_signature () -> ()
} {
  hir.unified_return
}
hir.unified_func @gux [] -> [] attributes {argNames = []} {
  // expected-remark @below {{calling `foo` itself here}}
  hir.unified_call @foo() : () -> () () -> () [] -> []
  hir.unified_signature () -> ()
} {
  hir.unified_return
}
