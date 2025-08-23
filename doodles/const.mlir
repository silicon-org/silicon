hir.func @foo {
  %0 = hir.int_type
  %1 = hir.const_wrap %0 : !hir.type
  %2 = hir.arg "x" : %1 : !hir.const<!hir.type> -> !hir.const<!hir.value>
  %3 = hir.constant_int 1 -> !hir.value
  %4 = hir.binary %2, %3 : !hir.const<!hir.value>
  %5 = hir.uint_type %4 : (!hir.const<!hir.value>) -> !hir.type
  %6 = hir.arg "y" : %5 : !hir.type -> !hir.value
  hir.args %2, %6
} {
  %0 = hir.constant_unit
  hir.return %0 : !hir.type
}

x: const type
%0 = !hir.const<!hir.higher_type>
%x = hir.arg "x" : %0 : !hir.const<!hir.higher_type> -> !hir.const<!hir.type>

y: Foo<x>
%1 = hir.const_unwrap %x : !hir.const<!hir.type> -> !hir.type
%2 = hir.param_type Foo, $1 : !hir.type -> !hir.type
%3 = hir.const_wrap %2 : !hir.type -> !hir.const<!hir.type>
%y = hir.arg "y" : %3 : !hir.const<!hir.type> -> !hir.value

//===--- Calls ------------------------------------------------------------===//

// Input
fn foo() {
  let x = const { const { 42 } };
  let y = const { 1337 };
  let z = 9001;
  bar(x, y, z);
}
fn bar(a: const const int, b: const int, c: int) {
  const { const { print(a) } }
  const { print(a+b) }
  print(a+b+c)
}

// Split into separate execution phases
func @foo {
  %x = constant_int 42
  %0 = call_phase @bar(%x)
  return freeze (%0)
} {
^bb0(%0: func):
  %y = constant_int 1337
  %1 = call_phase %0(%y)
  return freeze (%1)
} {
^bb0(%0: func):
  %z = constant_int 9001
  %1 = call_phase %0(%z)
}
func @bar {
  %0 = int_type
  return args (%0)
} {
^bb0(%a: value):
  print %a
  %0 = int_type
  return freeze (%a) args (%0)
} {
^bb0(%a: value, %b: value):
  %0 = add %a, %b
  print %0
  %1 = int_type
  return freeze (%a, %b) args (%1)
} {
^bb0(%a: value, %b: value, %c: value):
  %0 = add %a, %b
  %1 = add %0, %c
  print %1
  return
}

//===--- Execution --------------------------------------------------------===//

// dump 42
// dump 1379

hir.func @bar {
^bb0(%arg0: !mir.int):
  %0 = mir.constant_int 1379 : !mir.int
  %1 = mir.binary %0, %arg0 : !mir.int
  mir.dump_int %1 : !mir.int
  hir.return
}
