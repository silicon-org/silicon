// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s | FileCheck %s

// Types
unrealized_conversion_cast to !mir.type
unrealized_conversion_cast to !mir.int
unrealized_conversion_cast to !mir.uint<42>
unrealized_conversion_cast to !mir.unit
unrealized_conversion_cast to !mir.anyfunc
unrealized_conversion_cast to !mir.specialized_func

// Attributes
unrealized_conversion_cast to index {attr = #mir.type<!mir.int>}
unrealized_conversion_cast to index {attr = #mir.int<98765432109876543210987654321>}
unrealized_conversion_cast to index {attr = #mir.unit}
unrealized_conversion_cast to index {attr = #mir.func<@foo> : (i42) -> (index)}
unrealized_conversion_cast to index {attr = #mir.specialized_func<@foo, [], [], []>}
unrealized_conversion_cast to index {attr = #mir.specialized_func<@foo, [!mir.int], [!mir.type], [#mir.int<42> : !mir.int]>}

// CHECK: %c42_int = mir.constant
%c42_int = mir.constant #mir.int<42>

// CHECK: %int_type = mir.constant
%int_type = mir.constant #mir.type<!mir.int>

mir.specialize_func @foo() -> ()
mir.specialize_func @foo(%int_type) -> (%int_type)
mir.specialize_func @foo(%int_type) -> (%int_type), %c42_int : !mir.int

func.func @Return0() {
  mir.return
}
func.func @Return1(%arg0: !mir.type) {
  mir.return %arg0 : !mir.type
}
func.func @Return2(%arg0: !mir.type, %arg1: !mir.int) {
  mir.return %arg0, %arg1 : !mir.type, !mir.int
}

mir.call @foo() : () -> ()
mir.call @foo(%int_type) : (!mir.type) -> (!mir.type)
mir.call @foo(%int_type, %c42_int) : (!mir.type, !mir.int) -> (!mir.type, !mir.int)

// Functions
// CHECK-LABEL: mir.func @NoArgs
mir.func @NoArgs() -> () {
  mir.return
}

// CHECK-LABEL: mir.func @OneArg
mir.func @OneArg(%x: !mir.int) -> (result: !mir.int) {
  mir.return %x : !mir.int
}

// CHECK-LABEL: mir.func @TwoArgs
mir.func @TwoArgs(%x: !mir.int, %y: !mir.uint<8>) -> (sum: !mir.int) {
  mir.return %x : !mir.int
}

// CHECK-LABEL: mir.func @MultiResult
mir.func @MultiResult(%x: !mir.int) -> (a: !mir.int, b: !mir.type) {
  %t = mir.constant #mir.type<!mir.int>
  mir.return %x, %t : !mir.int, !mir.type
}

// CHECK-LABEL: mir.func private @Private
mir.func private @Private() -> () {
  mir.return
}

// Opaque type and attribute
unrealized_conversion_cast to !mir.opaque
unrealized_conversion_cast to index {attr = #mir.opaque<[#mir.int<42>, #mir.type<!mir.int>]>}

// Opaque pack/unpack
%opaque_ctx = mir.opaque_pack (%c42_int, %int_type) : (!mir.int, !mir.type) -> !mir.opaque
%opaque_a, %opaque_b = mir.opaque_unpack %opaque_ctx : !mir.opaque -> !mir.int, !mir.type
