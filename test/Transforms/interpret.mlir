// RUN: silicon-opt --interpret %s | FileCheck %s

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

//===----------------------------------------------------------------------===//
// Argument-passing calls: verify that the interpreter correctly maps call
// operands to callee block arguments and propagates the results back.

// Functions with arguments stay as mir.func after the pass; only zero-arg
// callers are converted to mir.evaluated_func.

// CHECK: mir.func @SingleArgHelper
mir.func @SingleArgHelper(%x: !mir.int) -> (result: !mir.int) {
  %0 = mir.constant #mir.int<3> : !mir.int
  %1 = mir.add %x, %0 : !mir.int
  mir.return %1 : !mir.int
}

// CHECK: mir.evaluated_func @SingleArgCaller [#mir.int<10> : !mir.int]
mir.func @SingleArgCaller() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<7> : !mir.int
  %1 = mir.call @SingleArgHelper(%0) : (!mir.int) -> !mir.int
  mir.return %1 : !mir.int
}

// CHECK: mir.func @MultiArg
mir.func @MultiArg(%x: !mir.int, %y: !mir.int) -> (result: !mir.int) {
  %0 = mir.add %x, %y : !mir.int
  mir.return %0 : !mir.int
}

// CHECK: mir.evaluated_func @MultiArgCaller [#mir.int<13> : !mir.int]
mir.func @MultiArgCaller() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<5> : !mir.int
  %1 = mir.constant #mir.int<8> : !mir.int
  %2 = mir.call @MultiArg(%0, %1) : (!mir.int, !mir.int) -> !mir.int
  mir.return %2 : !mir.int
}

// CHECK: mir.func @LeafSub
mir.func @LeafSub(%a: !mir.int, %b: !mir.int) -> (result: !mir.int) {
  %0 = mir.sub %b, %a : !mir.int
  mir.return %0 : !mir.int
}

// CHECK: mir.func @Intermediate
mir.func @Intermediate(%x: !mir.int) -> (result: !mir.int) {
  %0 = mir.constant #mir.int<10> : !mir.int
  %1 = mir.call @LeafSub(%x, %0) : (!mir.int, !mir.int) -> !mir.int
  mir.return %1 : !mir.int
}

// CHECK: mir.evaluated_func @OuterCaller [#mir.int<20> : !mir.int]
mir.func @OuterCaller() -> (result: !mir.int) {
  %0 = mir.constant #mir.int<5> : !mir.int
  %1 = mir.call @Intermediate(%0) : (!mir.int) -> !mir.int
  %2 = mir.constant #mir.int<4> : !mir.int
  %3 = mir.mul %1, %2 : !mir.int
  mir.return %3 : !mir.int
}
