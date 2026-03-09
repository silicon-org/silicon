// RUN: silicon-opt --lower-mir-to-circt %s | FileCheck %s

// ---- Module functions lower to hw.module ----

// CHECK-LABEL: hw.module @simple_module(in %a : i64, out result : i64)
// CHECK:         hw.output %a : i64
mir.func @simple_module(%a: !si.int) -> (result: !si.int) attributes {isModule} {
  mir.return %a : !si.int
}

// ---- Type mapping: !si.uint<N> → iN, !si.int → i64 ----

// CHECK-LABEL: hw.module @uint_types(in %x : i8, out result : i8)
mir.func @uint_types(%x: !si.uint<8>) -> (result: !si.uint<8>) attributes {isModule} {
  // CHECK: hw.output %x : i8
  mir.return %x : !si.uint<8>
}

// ---- Arithmetic ops ----

// CHECK-LABEL: hw.module @arith_ops
mir.func @arith_ops(%a: !si.int, %b: !si.int) -> (result: !si.int) attributes {isModule} {
  // CHECK: comb.add %a, %b : i64
  %0 = mir.add %a, %b : !si.int
  // CHECK: comb.sub %a, %b : i64
  %1 = mir.sub %a, %b : !si.int
  // CHECK: comb.mul %a, %b : i64
  %2 = mir.mul %a, %b : !si.int
  // CHECK: comb.divs %a, %b : i64
  %3 = mir.div %a, %b : !si.int
  // CHECK: comb.mods %a, %b : i64
  %4 = mir.mod %a, %b : !si.int
  mir.return %0 : !si.int
}

// ---- Bitwise ops ----

// CHECK-LABEL: hw.module @bitwise_ops
mir.func @bitwise_ops(%a: !si.int, %b: !si.int) -> (result: !si.int) attributes {isModule} {
  // CHECK: comb.and %a, %b : i64
  %0 = mir.and %a, %b : !si.int
  // CHECK: comb.or %a, %b : i64
  %1 = mir.or %a, %b : !si.int
  // CHECK: comb.xor %a, %b : i64
  %2 = mir.xor %a, %b : !si.int
  // CHECK: comb.shl %a, %b : i64
  %3 = mir.shl %a, %b : !si.int
  // CHECK: comb.shru %a, %b : i64
  %4 = mir.shr %a, %b : !si.int
  mir.return %0 : !si.int
}

// ---- Comparison ops (unsigned) ----

// CHECK-LABEL: hw.module @cmp_ops_unsigned
mir.func @cmp_ops_unsigned(%a: !si.uint<8>, %b: !si.uint<8>) -> (result: !si.uint<8>) attributes {isModule} {
  // CHECK: comb.icmp eq %a, %b : i8
  %0 = mir.eq %a, %b : !si.uint<8>
  // CHECK: comb.icmp ne %a, %b : i8
  %1 = mir.neq %a, %b : !si.uint<8>
  // CHECK: comb.icmp ult %a, %b : i8
  %2 = mir.lt %a, %b : !si.uint<8>
  // CHECK: comb.icmp ugt %a, %b : i8
  %3 = mir.gt %a, %b : !si.uint<8>
  // CHECK: comb.icmp ule %a, %b : i8
  %4 = mir.leq %a, %b : !si.uint<8>
  // CHECK: comb.icmp uge %a, %b : i8
  %5 = mir.geq %a, %b : !si.uint<8>
  mir.return %a : !si.uint<8>
}

// ---- Comparison ops (signed) ----

// CHECK-LABEL: hw.module @cmp_ops_signed
mir.func @cmp_ops_signed(%a: !si.int, %b: !si.int) -> (result: !si.int) attributes {isModule} {
  // CHECK: comb.icmp eq %a, %b : i64
  %0 = mir.eq %a, %b : !si.int
  // CHECK: comb.icmp ne %a, %b : i64
  %1 = mir.neq %a, %b : !si.int
  // CHECK: comb.icmp slt %a, %b : i64
  %2 = mir.lt %a, %b : !si.int
  // CHECK: comb.icmp sgt %a, %b : i64
  %3 = mir.gt %a, %b : !si.int
  // CHECK: comb.icmp sle %a, %b : i64
  %4 = mir.leq %a, %b : !si.int
  // CHECK: comb.icmp sge %a, %b : i64
  %5 = mir.geq %a, %b : !si.int
  mir.return %a : !si.int
}

// ---- Constants ----

// CHECK-LABEL: hw.module @constants
mir.func @constants() -> (result: !si.int) attributes {isModule} {
  // CHECK: %c42_i64 = hw.constant 42 : i64
  %c = mir.constant #si.int<42> : !si.int
  mir.return %c : !si.int
}

// ---- Calls become instances ----

// CHECK-LABEL: hw.module @callee(in %x : i64, out result : i64)
mir.func @callee(%x: !si.int) -> (result: !si.int) {
  mir.return %x : !si.int
}

// CHECK-LABEL: hw.module @caller
mir.func @caller() -> (result: !si.int) attributes {isModule} {
  %c = mir.constant #si.int<7> : !si.int
  // CHECK: hw.instance "callee_inst0" @callee
  %0 = mir.call @callee(%c) : (!si.int) -> !si.int
  mir.return %0 : !si.int
}

// ---- Comparison result returned directly as !si.bool ----

// CHECK-LABEL: hw.module @cmp_as_return
mir.func @cmp_as_return(%a: !si.int, %b: !si.int) -> (result: !si.bool) attributes {isModule} {
  // CHECK: [[CMP:%.+]] = comb.icmp sgt %a, %b : i64
  // CHECK: hw.output [[CMP]] : i1
  %0 = mir.gt %a, %b : !si.int
  mir.return %0 : !si.bool
}

// ---- Bool-to-i1 is a no-op after type conversion ----

// CHECK-LABEL: hw.module @bool_to_i1
mir.func @bool_to_i1(%a: !si.bool) -> (result: i1) attributes {isModule} {
  // CHECK: hw.output %a : i1
  %0 = mir.bool_to_i1 %a
  mir.return %0 : i1
}

// ---- !si.bool type maps to i1 ----

// CHECK-LABEL: hw.module @bool_arg(in %x : i1, out result : i1)
mir.func @bool_arg(%x: !si.bool) -> (result: !si.bool) attributes {isModule} {
  // CHECK: hw.output %x : i1
  mir.return %x : !si.bool
}

// ---- Zero-result module (no outputs) ----

// CHECK-LABEL: hw.module @zero_result()
// CHECK-NEXT:    hw.output
mir.func @zero_result() -> () attributes {isModule} {
  mir.return
}

// ---- Non-module functions are erased, not lowered ----

// Non-module functions with convertible types should be erased by the
// catch-all pattern, not lowered to hw.module.
// CHECK-NOT: hw.module @non_module
mir.func @non_module(%a: !si.int) -> (result: !si.int) {
  mir.return %a : !si.int
}

// ---- Remaining Silicon ops are erased ----

// The evaluated_func and split_func ops should be erased by the conversion.
// CHECK-NOT: mir.evaluated_func
// CHECK-NOT: hir.split_func
mir.evaluated_func @evaluated_result [#si.int<42>]
hir.split_func private @dummy_split() -> () {
  hir.signature () -> ()
} []
