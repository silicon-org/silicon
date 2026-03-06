// RUN: silicon-opt --interpret %s | FileCheck %s

// CHECK: mir.evaluated_func @foo [#mir.specialized_func<@foo, [], [], [#mir.specialized_func<@bar, [!mir.int], [], []> : !mir.specialized_func]> : !mir.specialized_func]
mir.func @foo() -> (result: !mir.specialized_func) {
  %0 = mir.call @bar() : () -> !mir.specialized_func
  %1 = mir.specialize_func @foo() -> (), %0 : !mir.specialized_func
  mir.return %1 : !mir.specialized_func
}

// CHECK: mir.evaluated_func @bar [#mir.specialized_func<@bar, [!mir.int], [], []> : !mir.specialized_func]
mir.func @bar() -> (result: !mir.specialized_func) {
  %0 = mir.constant #mir.specialized_func<@bar, [!mir.int], [], []> : !mir.specialized_func
  mir.return %0 : !mir.specialized_func
}

//===----------------------------------------------------------------------===//
// Binary operations: verify that the interpreter correctly evaluates all 16
// binary ops on constant integer operands.

// CHECK: mir.evaluated_func @test_add [#mir.int<5> : !mir.int]
mir.func @test_add() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<3> : !mir.int
  %1 = mir.constant #mir.int<2> : !mir.int
  %2 = mir.add %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_sub [#mir.int<4> : !mir.int]
mir.func @test_sub() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<7> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.sub %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_mul [#mir.int<20> : !mir.int]
mir.func @test_mul() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<4> : !mir.int
  %1 = mir.constant #mir.int<5> : !mir.int
  %2 = mir.mul %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_div [#mir.int<3> : !mir.int]
mir.func @test_div() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<10> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.div %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_mod [#mir.int<1> : !mir.int]
mir.func @test_mod() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<10> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.mod %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_and [#mir.int<2> : !mir.int]
mir.func @test_and() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<6> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.and %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_or [#mir.int<7> : !mir.int]
mir.func @test_or() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<5> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.or %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_xor [#mir.int<5> : !mir.int]
mir.func @test_xor() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<6> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.xor %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_shl [#mir.int<8> : !mir.int]
mir.func @test_shl() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<1> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.shl %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_shr [#mir.int<4> : !mir.int]
mir.func @test_shr() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<16> : !mir.int
  %1 = mir.constant #mir.int<2> : !mir.int
  %2 = mir.shr %0, %1 : !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_eq [#mir.int<1> : !mir.int]
mir.func @test_eq() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<3> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.eq %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_neq [#mir.int<1> : !mir.int]
mir.func @test_neq() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<3> : !mir.int
  %1 = mir.constant #mir.int<4> : !mir.int
  %2 = mir.neq %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_lt [#mir.int<1> : !mir.int]
mir.func @test_lt() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<2> : !mir.int
  %1 = mir.constant #mir.int<5> : !mir.int
  %2 = mir.lt %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_gt [#mir.int<1> : !mir.int]
mir.func @test_gt() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<5> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.gt %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_geq [#mir.int<1> : !mir.int]
mir.func @test_geq() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<3> : !mir.int
  %1 = mir.constant #mir.int<3> : !mir.int
  %2 = mir.geq %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.evaluated_func @test_leq [#mir.int<1> : !mir.int]
mir.func @test_leq() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<2> : !mir.int
  %1 = mir.constant #mir.int<5> : !mir.int
  %2 = mir.leq %0, %1 : !mir.int -> !mir.int
  mir.return %2 : !mir.int
}
