// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// Types
unrealized_conversion_cast to !si.type
unrealized_conversion_cast to !si.int
unrealized_conversion_cast to !si.uint<42>
unrealized_conversion_cast to !si.unit
unrealized_conversion_cast to !si.anyfunc

// Attributes
unrealized_conversion_cast to index {attr = #si.type<!si.int>}
unrealized_conversion_cast to index {attr = #si.int<98765432109876543210987654321>}
unrealized_conversion_cast to index {attr = #si.unit}
// CHECK: %c42_int = mir.constant
%c42_int = mir.constant #si.int<42>

// CHECK: %int_type = mir.constant
%int_type = mir.constant #si.type<!si.int>

func.func @Return0() {
  mir.return
}
func.func @Return1(%arg0: !si.type) {
  mir.return %arg0 : !si.type
}
func.func @Return2(%arg0: !si.type, %arg1: !si.int) {
  mir.return %arg0, %arg1 : !si.type, !si.int
}

mir.call @foo() : () -> ()
mir.call @foo(%int_type) : (!si.type) -> (!si.type)
mir.call @foo(%int_type, %c42_int) : (!si.type, !si.int) -> (!si.type, !si.int)

// Functions
// CHECK-LABEL: mir.func @NoArgs
mir.func @NoArgs() -> () {
  mir.return
}

// CHECK-LABEL: mir.func @OneArg
mir.func @OneArg(%x: !si.int) -> (result: !si.int) {
  mir.return %x : !si.int
}

// CHECK-LABEL: mir.func @TwoArgs
mir.func @TwoArgs(%x: !si.int, %y: !si.uint<8>) -> (sum: !si.int) {
  mir.return %x : !si.int
}

// CHECK-LABEL: mir.func @MultiResult
mir.func @MultiResult(%x: !si.int) -> (a: !si.int, b: !si.type) {
  %t = mir.constant #si.type<!si.int>
  mir.return %x, %t : !si.int, !si.type
}

// CHECK-LABEL: mir.func private @Private
mir.func private @Private() -> () {
  mir.return
}

// CHECK-LABEL: mir.func @Module
// CHECK-SAME: attributes {isModule}
mir.func @Module(%x: !si.int) -> (result: !si.int) attributes {isModule} {
  mir.return %x : !si.int
}

// UInt type construction
// CHECK-LABEL: mir.func @UIntType
mir.func @UIntType(%width: !si.int) -> (result: !si.type) {
  // CHECK: mir.uint_type %width
  %0 = mir.uint_type %width
  mir.return %0 : !si.type
}

// Opaque type and attribute
unrealized_conversion_cast to !si.opaque
unrealized_conversion_cast to index {attr = #si.opaque<[#si.int<42>, #si.type<!si.int>]>}

// Opaque pack/unpack
%opaque_ctx = mir.opaque_pack (%c42_int, %int_type) : (!si.int, !si.type)
%opaque_a, %opaque_b = mir.opaque_unpack %opaque_ctx : !si.int, !si.type

// Evaluated func
// CHECK: mir.evaluated_func @eval_test [#si.int<42> : !si.int]
mir.evaluated_func @eval_test [#si.int<42> : !si.int]

// CHECK: mir.evaluated_func @eval_multi [#si.int<1> : !si.int, #si.type<!si.int> : !si.type]
mir.evaluated_func @eval_multi [#si.int<1> : !si.int, #si.type<!si.int> : !si.type]

// CHECK: mir.evaluated_func private {{.*}}@eval_private [#si.unit : !si.unit]
mir.evaluated_func private @eval_private [#si.unit : !si.unit]

// Binary ops
// CHECK-LABEL: mir.func @BinaryOps
mir.func @BinaryOps(%x: !si.int, %y: !si.int) -> () {
  // CHECK: mir.add %x, %y : !si.int
  %0 = mir.add %x, %y : !si.int
  // CHECK: mir.sub %x, %y : !si.int
  %1 = mir.sub %x, %y : !si.int
  // CHECK: mir.mul %x, %y : !si.int
  %2 = mir.mul %x, %y : !si.int
  // CHECK: mir.div %x, %y : !si.int
  %3 = mir.div %x, %y : !si.int
  // CHECK: mir.mod %x, %y : !si.int
  %4 = mir.mod %x, %y : !si.int
  // CHECK: mir.and %x, %y : !si.int
  %5 = mir.and %x, %y : !si.int
  // CHECK: mir.or %x, %y : !si.int
  %6 = mir.or %x, %y : !si.int
  // CHECK: mir.xor %x, %y : !si.int
  %7 = mir.xor %x, %y : !si.int
  // CHECK: mir.shl %x, %y : !si.int
  %8 = mir.shl %x, %y : !si.int
  // CHECK: mir.shr %x, %y : !si.int
  %9 = mir.shr %x, %y : !si.int
  mir.return
}

// Comparison ops
// CHECK-LABEL: mir.func @CmpOps
mir.func @CmpOps(%x: !si.int, %y: !si.int) -> () {
  // CHECK: mir.eq %x, %y : !si.int
  %0 = mir.eq %x, %y : !si.int
  // CHECK: mir.neq %x, %y : !si.int
  %1 = mir.neq %x, %y : !si.int
  // CHECK: mir.lt %x, %y : !si.int
  %2 = mir.lt %x, %y : !si.int
  // CHECK: mir.gt %x, %y : !si.int
  %3 = mir.gt %x, %y : !si.int
  // CHECK: mir.leq %x, %y : !si.int
  %4 = mir.leq %x, %y : !si.int
  // CHECK: mir.geq %x, %y : !si.int
  %5 = mir.geq %x, %y : !si.int
  mir.return
}

// bool_to_i1
// CHECK-LABEL: mir.func @BoolToI1
mir.func @BoolToI1(%cond: !si.bool) -> () {
  // CHECK: mir.bool_to_i1 %cond
  %0 = mir.bool_to_i1 %cond
  mir.return
}

