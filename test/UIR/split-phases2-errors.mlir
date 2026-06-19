// RUN: silicon-opt --split-input-file --verify-diagnostics --split-phases2 %s

// -----

// Recursive call cycle: @A calls @B, @B calls @A.
// expected-error @below {{recursive call cycle detected}}
uir.func @A() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  %r = uir.call @B() : () -> () () -> !hir.any [] -> [0]
  uir.return
}

uir.func @B() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  %r = uir.call @A() : () -> () () -> !hir.any [] -> [0]
  uir.return
}
