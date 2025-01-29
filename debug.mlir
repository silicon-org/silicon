// let x = 42;
// let y: uint<x> = 1337;

%x = untyped.let "x"
%c42 = untyped.constant 42
untyped.store %x, %c42

// let x: uint<40+2>
%0 = untyped.int 40
%1 = untyped.int 2
%2 = untyped.add %0, %1
%3 = untyped.uint_type %2
%x = untyped.let "x", %3

// let a = 41;
// let b = 1;
// let x: uint<a+b>

// fn foo(a: int, b: int) -> int { a + b }
// let a = 40;
// let b = 2;
// let c = foo(a, b);
// let x: uint<c>;

// fn foo(a: int, b: int) {
//   let x: uint<a>;
//   let y: uint<b>;
//   x = y;
// }

untyped.func @foo {
  %0 = untyped.int_type
  %a = untyped.func_arg "a", %0
  %b = untyped.func_arg "b", %0
  %1 = untyped.eq %a, %b
  untyped.require %1
} {
  %0 = untyped.uint_type %a
  %1 = untyped.uint_type %b
  %x = untyped.let "x", %0
  %y = untyped.let "y", %1
  %2 = untyped.load %y
  untyped.store %x, %2
}

// insert type constraints

untyped.func @foo {
  %0 = untyped.int_type
  %a = untyped.func_arg "a", %0
  %b = untyped.func_arg "b", %0
  %1 = untyped.eq %a, %b
  untyped.require %1
} {
  %0 = untyped.int_type
  %1 = untyped.type_of %a
  %2 = untyped.type_of %b
  untyped.unify %0, %1
  untyped.unify %0, %2
  %ai = untyped.as_type %a : !untyped.int
  %bi = untyped.as_type %b : !untyped.int
  %3 = untyped.uint_type %a
  %4 = untyped.uint_type %b
  %x = untyped.let "x", %3
  %y = untyped.let "y", %4
  %5 = untyped.type_of %y
  %6 = untyped.unpack_ref %5
  %7 = untyped.load %y, %6
  %8 = untyped.type_of %x
  %9 = untyped.unpack_ref %8
  %10 = untyped.type_of %7
  %11 = untyped.unify %9, %10
  untyped.store %x, %7, %11
}

// canonicalize (round 1)

untyped.func @foo {
  %a = untyped.func_arg "a" : !untyped.int
  %b = untyped.func_arg "b" : !untyped.int
  %0 = untyped.eq %a, %b
  untyped.require %0
} {
  %0 = untyped.uint_type %a : !untyped.int -> !untyped.some_uint
  %1 = untyped.uint_type %b : !untyped.int -> !untyped.some_uint
  %x = untyped.let "x", %0
  %y = untyped.let "y", %1
  %2 = untyped.load %y, %1
  %3 = untyped.unify %0, %1
  untyped.store %x, %2, %3
}

// canonicalize (round 2)

untyped.func @foo {
  %a = untyped.func_arg "a" : !untyped.int
  %b = untyped.func_arg "b" : !untyped.int
  %0 = untyped.eq %a, %b
  untyped.require %0
} {
  %0 = untyped.uint_type %a : !untyped.int -> !untyped.some_uint
  %1 = untyped.uint_type %b : !untyped.int -> !untyped.some_uint
  %2 = untyped.unify %0, %1
  %x = untyped.let "x", %2
  %y = untyped.let "y", %2
  %3 = untyped.load %y, %2
  untyped.store %x, %3, %2
}

// canonicalize (round 3)

untyped.func @foo {
  %a = untyped.func_arg "a" : !untyped.int
  %b = untyped.func_arg "b" : !untyped.int
  %0 = untyped.eq %a, %b
  untyped.require %0
} {
  %0 = untyped.unify %a, %b
  %1 = untyped.uint_type %0 : !untyped.int -> !untyped.some_uint
  %x = untyped.let "x", %1
  %y = untyped.let "y", %1
  %2 = untyped.load %y, %1
  untyped.store %x, %2, %1
}

// canonicalize (round 4)

untyped.func @foo {
  %a = untyped.func_arg "a" : !untyped.int
  %b = untyped.func_arg "b" : !untyped.int
  %0 = untyped.eq %a, %b
  untyped.require %0
} {
  %0 = untyped.uint_type %a : !untyped.int -> !untyped.some_uint
  %x = untyped.let "x", %0
  %y = untyped.let "y", %0
  %1 = untyped.load %y, %0
  untyped.store %x, %1, %0
}
