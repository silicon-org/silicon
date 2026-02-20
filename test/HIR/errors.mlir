// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{does not reference a valid `hir.unchecked_func`}}
hir.unchecked_call @foo() : () -> ()

// -----

hir.unchecked_func @foo {
  %0 = hir.int_type
  %1 = hir.unchecked_arg "a", %0, 0 : !hir.type
  hir.unchecked_signature (%1 : !hir.value) -> ()
} {
  hir.unchecked_return
}
// expected-error @below {{has 0 arguments, but @foo expects 1}}
hir.unchecked_call @foo() : () -> ()

// -----

hir.unchecked_func @foo {
  %0 = hir.int_type
  hir.unchecked_signature () -> (%0 : !hir.type)
} {
  hir.unchecked_return
}
// expected-error @below {{has 0 results, but @foo expects 1}}
hir.unchecked_call @foo() : () -> ()

// -----

hir.unchecked_func @foo {
  // expected-error @below {{can only appear in the last block}}
  hir.unchecked_signature () -> ()
^bb1:
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_return
}

// -----

hir.unchecked_func @foo {
  hir.unchecked_signature () -> ()
} {
  // expected-error @below {{can only appear in the last block}}
  hir.unchecked_return
^bb1:
  hir.unchecked_return
}

// -----

// expected-error @below {{requires `hir.unchecked_return` terminator in the body}}
hir.unchecked_func @foo {
  hir.unchecked_signature () -> ()
} {
  hir.unchecked_signature () -> ()
}

// -----

// expected-error @below {{requires `hir.unchecked_signature` terminator in the signature}}
hir.unchecked_func @foo {
  hir.unchecked_return
} {
  hir.unchecked_return
}
