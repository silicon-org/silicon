// RUN: silicon-opt --split-input-file --verify-diagnostics --split-phases2 %s

// -----

// Recursive call cycle: @A calls @B, @B calls @A.
// expected-error @below {{recursive call cycle detected}}
uir.func @A() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  uir.call @B() : () -> () () -> () [] -> []
  uir.return
}

uir.func @B() -> () {
  uir.signature () -> ()
} {
  %type_type = hir.type_type
  uir.call @A() : () -> () () -> () [] -> []
  uir.return
}
