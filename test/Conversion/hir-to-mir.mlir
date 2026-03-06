// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: mir.func @Types
hir.func @Types() -> () {
  // CHECK: mir.constant #si.type<!si.int>
  %int_type = hir.int_type

  // CHECK: mir.constant #si.type<!si.unit>
  %unit_type = hir.unit_type

  // CHECK: mir.constant #si.type<!si.type>
  %type_type = hir.type_type

  // CHECK: mir.constant #si.int<42>
  // CHECK: mir.constant #si.type<!si.uint<42>>
  %c42_int = hir.constant_int 42
  %uint42_type = hir.uint_type %c42_int

  // CHECK: mir.constant #si.type<!si.anyfunc>
  %anyfunc_type = hir.anyfunc_type

  // CHECK: mir.constant #si.type<() -> ()>
  hir.func_type () -> ()
  // CHECK: mir.constant #si.type<(!si.int) -> !si.uint<42>>
  hir.func_type (%int_type) -> (%uint42_type)

  hir.return
}

// CHECK-LABEL: mir.func @Constants
hir.func @Constants() -> () {
  // CHECK: mir.constant #si.int<42>
  hir.constant_int 42
  // CHECK: mir.constant #si.unit
  hir.constant_unit
  // CHECK: mir.constant #si.type
  // CHECK: mir.constant #mir.func<@foo> : () -> ()
  %0 = hir.func_type () -> ()
  hir.constant_func @foo : %0
  hir.return
}

// CHECK-LABEL: mir.func @Calls
hir.func @Calls() -> () {
  // CHECK: mir.call @foo() : () -> ()
  hir.call @foo() : () -> ()

  %int_type = hir.int_type
  // CHECK: [[C42:%.+]] = mir.constant #si.int<42>
  %c42 = hir.constant_int 42
  // CHECK: mir.constant #si.type<!si.uint<42>>
  %uint42_type = hir.uint_type %c42

  // CHECK: mir.call @foo([[C42]]) : (!si.int) -> !si.uint<42>
  hir.call @foo(%c42) : (%int_type) -> (%uint42_type)

  hir.return
}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @BinaryArithmetic(%a: !si.int, %b: !si.int)
hir.func @BinaryArithmetic(%a, %b) -> (r0, r1, r2, r3, r4) {
  %int = hir.int_type
  %a_typed = hir.coerce_type %a, %int
  %b_typed = hir.coerce_type %b, %int
  %r0 = hir.add %a_typed, %b_typed : %int
  %r1 = hir.sub %a_typed, %b_typed : %int
  %r2 = hir.mul %a_typed, %b_typed : %int
  %r3 = hir.div %a_typed, %b_typed : %int
  %r4 = hir.mod %a_typed, %b_typed : %int
  // CHECK: mir.add %a, %b : !si.int
  // CHECK: mir.sub %a, %b : !si.int
  // CHECK: mir.mul %a, %b : !si.int
  // CHECK: mir.div %a, %b : !si.int
  // CHECK: mir.mod %a, %b : !si.int
  hir.return (%r0, %r1, %r2, %r3, %r4) : (%int, %int, %int, %int, %int)
}

// CHECK-LABEL: mir.func @BinaryBitwise(%a: !si.int, %b: !si.int)
hir.func @BinaryBitwise(%a, %b) -> (r0, r1, r2, r3, r4) {
  %int = hir.int_type
  %a_typed = hir.coerce_type %a, %int
  %b_typed = hir.coerce_type %b, %int
  %r0 = hir.and %a_typed, %b_typed : %int
  %r1 = hir.or %a_typed, %b_typed : %int
  %r2 = hir.xor %a_typed, %b_typed : %int
  %r3 = hir.shl %a_typed, %b_typed : %int
  %r4 = hir.shr %a_typed, %b_typed : %int
  // CHECK: mir.and %a, %b : !si.int
  // CHECK: mir.or %a, %b : !si.int
  // CHECK: mir.xor %a, %b : !si.int
  // CHECK: mir.shl %a, %b : !si.int
  // CHECK: mir.shr %a, %b : !si.int
  hir.return (%r0, %r1, %r2, %r3, %r4) : (%int, %int, %int, %int, %int)
}

// CHECK-LABEL: mir.func @BinaryComparison(%a: !si.int, %b: !si.int)
hir.func @BinaryComparison(%a, %b) -> (r0, r1, r2, r3, r4, r5) {
  %int = hir.int_type
  %a_typed = hir.coerce_type %a, %int
  %b_typed = hir.coerce_type %b, %int
  %r0 = hir.eq %a_typed, %b_typed : %int
  %r1 = hir.neq %a_typed, %b_typed : %int
  %r2 = hir.lt %a_typed, %b_typed : %int
  %r3 = hir.gt %a_typed, %b_typed : %int
  %r4 = hir.geq %a_typed, %b_typed : %int
  %r5 = hir.leq %a_typed, %b_typed : %int
  // CHECK: mir.eq %a, %b : !si.int -> !si.int
  // CHECK: mir.neq %a, %b : !si.int -> !si.int
  // CHECK: mir.lt %a, %b : !si.int -> !si.int
  // CHECK: mir.gt %a, %b : !si.int -> !si.int
  // CHECK: mir.geq %a, %b : !si.int -> !si.int
  // CHECK: mir.leq %a, %b : !si.int -> !si.int
  hir.return (%r0, %r1, %r2, %r3, %r4, %r5) : (%int, %int, %int, %int, %int, %int)
}

// Verify that UnifyOp forwards its operand through opaque_pack, rather than
// replacing it with a dummy !hir.any constant.
//
// CHECK-LABEL: mir.func @UnifyInOpaquePack(%T: !si.type)
hir.func @UnifyInOpaquePack(%T) -> (ctx) {
  %type_type = hir.type_type
  %coerced = hir.coerce_type %T, %type_type
  %unified = hir.unify %coerced, %T
  // CHECK: mir.opaque_pack(%T) : (!si.type) -> !si.opaque
  %packed = hir.opaque_pack(%unified)
  %opaque = hir.opaque_type
  hir.return (%packed) : (%opaque)
}

// CHECK-LABEL: mir.func @OpaqueTypes
hir.func @OpaqueTypes() -> () {
  // CHECK: mir.constant #si.type<!si.opaque>
  %opaque_type = hir.opaque_type
  hir.return
}

// CHECK-LABEL: mir.func @OpaqueArg(%a: !si.opaque)
hir.func @OpaqueArg(%a) -> (result) {
  %opaque = hir.opaque_type
  %a0 = hir.coerce_type %a, %opaque
  // CHECK: mir.return %a
  hir.return (%a0) : (%opaque)
}

// Verify that a hir.func with unresolved typeOfArgs in a call is not lowered.
// The outer function has a block argument used as a type operand for the call,
// so shouldLower must return false for it and leave it as hir.func.
//
// CHECK-LABEL: hir.func @UnresolvedCallArgType
// CHECK-NOT: mir.func @UnresolvedCallArgType
hir.func @UnresolvedCallArgType(%T) -> () {
  %x = hir.constant_int 0
  hir.call @identity(%x) : (%T) -> ()
  hir.return
}

// CHECK-LABEL: mir.func @Casts
hir.func @Casts() -> (a, b) {
  // CHECK-NEXT: [[TMP1:%.+]] = mir.constant
  %a0 = mir.constant #si.int<42>
  %a1 = builtin.unrealized_conversion_cast %a0 : !si.int to !hir.any

  // CHECK-NEXT: [[TMP2:%.+]] = mir.constant
  %b0 = mir.constant #mir.func<@foo> : () -> ()
  %b1 = builtin.unrealized_conversion_cast %b0 : () -> () to !hir.any

  %ta = hir.int_type
  %tb = hir.anyfunc_type
  // CHECK: mir.return [[TMP1]], [[TMP2]]
  hir.return (%a1, %b1) : (%ta, %tb)
}
