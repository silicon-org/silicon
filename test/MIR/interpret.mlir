// RUN: silicon-opt --interpret %s | FileCheck %s

//===----------------------------------------------------------------------===//
// Binary operations: verify that the interpreter correctly evaluates all 16
// binary ops on constant integer operands.

// CHECK: mir.evaluated_func @test_add [#si.int<5> : !si.int]
mir.func @test_add() -> (result: !si.int) {
  %0 = mir.constant #si.int<3> : !si.int
  %1 = mir.constant #si.int<2> : !si.int
  %2 = mir.add %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_sub [#si.int<4> : !si.int]
mir.func @test_sub() -> (result: !si.int) {
  %0 = mir.constant #si.int<7> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.sub %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_mul [#si.int<20> : !si.int]
mir.func @test_mul() -> (result: !si.int) {
  %0 = mir.constant #si.int<4> : !si.int
  %1 = mir.constant #si.int<5> : !si.int
  %2 = mir.mul %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_div [#si.int<3> : !si.int]
mir.func @test_div() -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.div %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_mod [#si.int<1> : !si.int]
mir.func @test_mod() -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.mod %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_and [#si.int<2> : !si.int]
mir.func @test_and() -> (result: !si.int) {
  %0 = mir.constant #si.int<6> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.and %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_or [#si.int<7> : !si.int]
mir.func @test_or() -> (result: !si.int) {
  %0 = mir.constant #si.int<5> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.or %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_xor [#si.int<5> : !si.int]
mir.func @test_xor() -> (result: !si.int) {
  %0 = mir.constant #si.int<6> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.xor %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_shl [#si.int<8> : !si.int]
mir.func @test_shl() -> (result: !si.int) {
  %0 = mir.constant #si.int<1> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.shl %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_shr [#si.int<4> : !si.int]
mir.func @test_shr() -> (result: !si.int) {
  %0 = mir.constant #si.int<16> : !si.int
  %1 = mir.constant #si.int<2> : !si.int
  %2 = mir.shr %0, %1 : !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_eq [#si.int<1> : !si.int]
mir.func @test_eq() -> (result: !si.int) {
  %0 = mir.constant #si.int<3> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.eq %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_neq [#si.int<1> : !si.int]
mir.func @test_neq() -> (result: !si.int) {
  %0 = mir.constant #si.int<3> : !si.int
  %1 = mir.constant #si.int<4> : !si.int
  %2 = mir.neq %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_lt [#si.int<1> : !si.int]
mir.func @test_lt() -> (result: !si.int) {
  %0 = mir.constant #si.int<2> : !si.int
  %1 = mir.constant #si.int<5> : !si.int
  %2 = mir.lt %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_gt [#si.int<1> : !si.int]
mir.func @test_gt() -> (result: !si.int) {
  %0 = mir.constant #si.int<5> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.gt %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_geq [#si.int<1> : !si.int]
mir.func @test_geq() -> (result: !si.int) {
  %0 = mir.constant #si.int<3> : !si.int
  %1 = mir.constant #si.int<3> : !si.int
  %2 = mir.geq %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.evaluated_func @test_leq [#si.int<1> : !si.int]
mir.func @test_leq() -> (result: !si.int) {
  %0 = mir.constant #si.int<2> : !si.int
  %1 = mir.constant #si.int<5> : !si.int
  %2 = mir.leq %0, %1 : !si.int -> !si.int
  mir.return %2 : !si.int
}

//===----------------------------------------------------------------------===//
// Argument-passing calls: verify that the interpreter correctly maps call
// operands to callee block arguments and propagates the results back.

// Functions with arguments stay as mir.func after the pass; only zero-arg
// callers are converted to mir.evaluated_func.

// CHECK: mir.func @SingleArgHelper
mir.func @SingleArgHelper(%x: !si.int) -> (result: !si.int) {
  %0 = mir.constant #si.int<3> : !si.int
  %1 = mir.add %x, %0 : !si.int
  mir.return %1 : !si.int
}

// CHECK: mir.evaluated_func @SingleArgCaller [#si.int<10> : !si.int]
mir.func @SingleArgCaller() -> (result: !si.int) {
  %0 = mir.constant #si.int<7> : !si.int
  %1 = mir.call @SingleArgHelper(%0) : (!si.int) -> !si.int
  mir.return %1 : !si.int
}

// CHECK: mir.func @MultiArg
mir.func @MultiArg(%x: !si.int, %y: !si.int) -> (result: !si.int) {
  %0 = mir.add %x, %y : !si.int
  mir.return %0 : !si.int
}

// CHECK: mir.evaluated_func @MultiArgCaller [#si.int<13> : !si.int]
mir.func @MultiArgCaller() -> (result: !si.int) {
  %0 = mir.constant #si.int<5> : !si.int
  %1 = mir.constant #si.int<8> : !si.int
  %2 = mir.call @MultiArg(%0, %1) : (!si.int, !si.int) -> !si.int
  mir.return %2 : !si.int
}

// CHECK: mir.func @LeafSub
mir.func @LeafSub(%a: !si.int, %b: !si.int) -> (result: !si.int) {
  %0 = mir.sub %b, %a : !si.int
  mir.return %0 : !si.int
}

// CHECK: mir.func @Intermediate
mir.func @Intermediate(%x: !si.int) -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.call @LeafSub(%x, %0) : (!si.int, !si.int) -> !si.int
  mir.return %1 : !si.int
}

// CHECK: mir.evaluated_func @OuterCaller [#si.int<20> : !si.int]
mir.func @OuterCaller() -> (result: !si.int) {
  %0 = mir.constant #si.int<5> : !si.int
  %1 = mir.call @Intermediate(%0) : (!si.int) -> !si.int
  %2 = mir.constant #si.int<4> : !si.int
  %3 = mir.mul %1, %2 : !si.int
  mir.return %3 : !si.int
}

//===----------------------------------------------------------------------===//
// Opaque pack/unpack: verify that the interpreter packs values into an opaque
// bundle and unpacks them back, preserving the individual attributes.

// CHECK: mir.evaluated_func @test_opaque_pack_unpack [#si.int<30> : !si.int]
mir.func @test_opaque_pack_unpack() -> (result: !si.int) {
  %0 = mir.constant #si.int<10> : !si.int
  %1 = mir.constant #si.int<20> : !si.int
  %2 = mir.opaque_pack (%0, %1) : (!si.int, !si.int) -> !si.opaque
  %3, %4 = mir.opaque_unpack %2 : !si.opaque -> !si.int, !si.int
  %5 = mir.add %3, %4 : !si.int
  mir.return %5 : !si.int
}

// CHECK: mir.evaluated_func @test_opaque_single [#si.int<42> : !si.int]
mir.func @test_opaque_single() -> (result: !si.int) {
  %0 = mir.constant #si.int<42> : !si.int
  %1 = mir.opaque_pack (%0) : (!si.int) -> !si.opaque
  %2 = mir.opaque_unpack %1 : !si.opaque -> !si.int
  mir.return %2 : !si.int
}
