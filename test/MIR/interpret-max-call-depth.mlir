// RUN: silicon-opt --interpret="max-call-depth=5" %s --split-input-file --verify-diagnostics

// Recursive function: calls itself until the call depth limit is reached.

mir.func @Recurse(%n: !si.int) -> (result: !si.int) {
  %one = mir.constant #si.int<1> : !si.int
  %next = mir.sub %n, %one : !si.int
  // expected-error @below {{interpreter exceeded call depth of 5; possible unbounded recursion}}
  %result = mir.call @Recurse(%next) : (!si.int) -> !si.int
  mir.return %result : !si.int
}

mir.func @Main() -> (result: !si.int) {
  %start = mir.constant #si.int<100> : !si.int
  %result = mir.call @Recurse(%start) : (!si.int) -> !si.int
  mir.return %result : !si.int
}
