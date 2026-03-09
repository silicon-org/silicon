// RUN: silicon-opt --interpret="max-steps=10" %s --split-input-file --verify-diagnostics

// Infinite loop: a function with a back-edge that never terminates hits the
// step limit and produces an error diagnostic.

mir.func @InfiniteLoop() -> (result: !si.int) {
  %0 = mir.constant #si.int<0> : !si.int
  cf.br ^loop(%0 : !si.int)
^loop(%i: !si.int):
  %one = mir.constant #si.int<1> : !si.int
  %next = mir.add %i, %one : !si.int
  // expected-error @below {{interpreter exceeded 10 steps; possible infinite loop}}
  cf.br ^loop(%next : !si.int)
^exit:
  mir.return %0 : !si.int
}
