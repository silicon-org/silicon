// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{does not reference a valid `hir.unified_func`}}
hir.unified_call @foo() : () -> () [] -> []

// -----

hir.unified_func @foo [0] -> [] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0) -> ()
} {
  hir.unified_return
}
// expected-error @below {{has 0 arguments, but @foo expects 1}}
hir.unified_call @foo() : () -> () [] -> []

// -----

hir.unified_func @foo [] -> [0] attributes {argNames = []} {
  %0 = hir.int_type
  hir.unified_signature () -> (%0)
} {
  hir.unified_return
}
// expected-error @below {{has 0 results, but @foo expects 1}}
hir.unified_call @foo() : () -> () [] -> []

// -----

hir.unified_func @foo [] -> [] attributes {argNames = []} {
  // expected-error @below {{can only appear in the last block}}
  hir.unified_signature () -> ()
^bb1:
  hir.unified_signature () -> ()
} {
  hir.unified_return
}

// -----

hir.unified_func @foo [] -> [] attributes {argNames = []} {
  hir.unified_signature () -> ()
} {
  // expected-error @below {{can only appear in the last block}}
  hir.unified_return
^bb1:
  hir.unified_return
}

// -----

// expected-error @below {{requires `hir.unified_return` terminator in the body}}
hir.unified_func @foo [] -> [] attributes {argNames = []} {
  hir.unified_signature () -> ()
} {
  hir.unified_signature () -> ()
}

// -----

// expected-error @below {{requires `hir.unified_signature` terminator in the signature}}
hir.unified_func @foo [] -> [] attributes {argNames = []} {
  hir.unified_return
} {
  hir.unified_return
}

// -----

// expected-error @below {{argPhases has 2 entries but function has 1 arguments}}
hir.unified_func @foo [0, -1] -> [0] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}

// -----

// expected-error @below {{resultPhases has 2 entries but function has 1 results}}
hir.unified_func @foo [0] -> [0, -1] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}

// -----

hir.unified_func @foo [0] -> [0] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
// expected-error @below {{argPhases has 0 entries but call has 1 arguments}}
hir.unified_call @foo(%arg) : (!hir.any) -> (!hir.any) [] -> [0]

// -----

hir.unified_func @foo [0] -> [0] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
// expected-error @below {{resultPhases has 0 entries but call has 1 results}}
hir.unified_call @foo(%arg) : (!hir.any) -> (!hir.any) [0] -> []

// -----

hir.unified_func @foo [0] -> [0] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
// expected-error @below {{argPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (!hir.any) -> (!hir.any) [-1] -> [0]

// -----

hir.unified_func @foo [0] -> [0] attributes {argNames = ["a"]} {
^bb0(%a: !hir.any):
  %0 = hir.int_type
  %1 = hir.int_type
  hir.unified_signature (%0) -> (%1)
} {
  hir.unified_return
}
%arg = hir.constant_int 42
// expected-error @below {{resultPhases do not match callee @foo}}
hir.unified_call @foo(%arg) : (!hir.any) -> (!hir.any) [0] -> [-1]
