// RUN: silicon-opt %s | silicon-opt | FileCheck %s

// Roundtrip test for all `si.` types and attributes.

// CHECK-LABEL: func.func @types
func.func @types(
  // CHECK-SAME: %{{.*}}: !si.bool
  %z: !si.bool,
  // CHECK-SAME: %{{.*}}: !si.int
  %a: !si.int,
  // CHECK-SAME: %{{.*}}: !si.uint<8>
  %b: !si.uint<8>,
  // CHECK-SAME: %{{.*}}: !si.unit
  %c: !si.unit,
  // CHECK-SAME: %{{.*}}: !si.type
  %d: !si.type,
  // CHECK-SAME: %{{.*}}: !si.anyfunc
  %e: !si.anyfunc,
  // CHECK-SAME: %{{.*}}: !si.opaque,
  %f: !si.opaque,
  // CHECK-SAME: %{{.*}}: !si.opaque_list
  %g: !si.opaque_list
) {
  return
}

// CHECK-LABEL: func.func @attrs
func.func @attrs() {
  // CHECK: mir.constant #si.bool<true>
  %b0 = mir.constant #si.bool<true>
  // CHECK: mir.constant #si.bool<false>
  %b1 = mir.constant #si.bool<false>
  // CHECK: mir.constant #si.int<42>
  %0 = mir.constant #si.int<42>
  // CHECK: mir.constant #si.type<!si.int>
  %1 = mir.constant #si.type<!si.int>
  // CHECK: mir.constant #si.type<!si.uint<16>>
  %2 = mir.constant #si.type<!si.uint<16>>
  // CHECK: mir.constant #si.uint<8, 42>
  %u0 = mir.constant #si.uint<8, 42>
  // CHECK: mir.constant #si.uint<16, 0>
  %u1 = mir.constant #si.uint<16, 0>
  // CHECK: mir.constant #si.unit
  %3 = mir.constant #si.unit
  // CHECK: mir.constant #si.opaque<[]>
  %4 = mir.constant #si.opaque<[]> : !si.opaque
  // CHECK: mir.constant #si.opaque<[#si.int<1> : !si.int, #si.int<2> : !si.int]>
  %5 = mir.constant #si.opaque<[#si.int<1>, #si.int<2>]> : !si.opaque
  return
}
