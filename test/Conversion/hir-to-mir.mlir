// RUN: silicon-opt --lower-hir-to-mir %s | FileCheck %s

// CHECK-LABEL: mir.func @Types
hir.func @Types() -> () {
  hir.signature () -> ()
} {
  // CHECK: mir.constant #si.type<!si.bool>
  %bool_type = hir.bool_type

  // CHECK: mir.constant #si.type<!si.int>
  %int_type = hir.int_type

  // CHECK: mir.constant #si.type<!si.unit>
  %unit_type = hir.unit_type

  // CHECK: mir.constant #si.type<!si.type>
  %type_type = hir.type_type

  // CHECK: mir.constant #si.int<42>
  // CHECK: mir.constant #si.type<!si.uint<42>>
  %c42_int = hir.constant_int 42 : %int_type
  %uint42_type = hir.uint_type %c42_int

  // CHECK: mir.constant #si.type<!si.anyfunc>
  %anyfunc_type = hir.anyfunc_type

  // CHECK: mir.constant #si.type<() -> ()>
  hir.func_type () -> ()
  // CHECK: mir.constant #si.type<(!si.int) -> !si.uint<42>>
  hir.func_type (%int_type) -> (%uint42_type)

  hir.return : () -> ()
}

// CHECK-LABEL: mir.func @Constants
hir.func @Constants() -> () {
  hir.signature () -> ()
} {
  // CHECK: mir.constant #si.bool<true>
  hir.constant_bool <true>
  // CHECK: mir.constant #si.bool<false>
  hir.constant_bool <false>
  // CHECK: mir.constant #si.int<42>
  %int0 = hir.int_type
  hir.constant_int 42 : %int0
  // CHECK: mir.constant #si.unit
  hir.constant_unit
  hir.return : () -> ()
}

// CHECK-LABEL: mir.func @Calls
hir.func @Calls() -> () {
  hir.signature () -> ()
} {
  // CHECK: mir.call @foo() : () -> ()
  hir.call @foo() : () -> ()

  %int_type = hir.int_type
  // CHECK: [[C42:%.+]] = mir.constant #si.int<42>
  %c42 = hir.constant_int 42 : %int_type
  // CHECK: mir.constant #si.type<!si.uint<42>>
  %uint42_type = hir.uint_type %c42

  // CHECK: mir.call @foo([[C42]]) : (!si.int) -> !si.uint<42>
  hir.call @foo(%c42) : (%int_type) -> (%uint42_type)

  hir.return : () -> ()
}

//===----------------------------------------------------------------------===//
// Binary operations
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @BinaryArithmetic(%a: !si.int, %b: !si.int)
hir.func @BinaryArithmetic(%a, %b) -> (r0, r1, r2, r3, r4) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int, %int, %int, %int, %int)
} {
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
  hir.return %r0, %r1, %r2, %r3, %r4 : (%int, %int) -> (%int, %int, %int, %int, %int)
}

// CHECK-LABEL: mir.func @BinaryBitwise(%a: !si.int, %b: !si.int)
hir.func @BinaryBitwise(%a, %b) -> (r0, r1, r2, r3, r4) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int, %int, %int, %int, %int)
} {
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
  hir.return %r0, %r1, %r2, %r3, %r4 : (%int, %int) -> (%int, %int, %int, %int, %int)
}

// CHECK-LABEL: mir.func @BinaryComparison(%a: !si.int, %b: !si.int)
hir.func @BinaryComparison(%a, %b) -> (r0, r1, r2, r3, r4, r5) {
  %int = hir.int_type
  %bool = hir.bool_type
  hir.signature (%int, %int) -> (%bool, %bool, %bool, %bool, %bool, %bool)
} {
  %int = hir.int_type
  %bool = hir.bool_type
  %a_typed = hir.coerce_type %a, %int
  %b_typed = hir.coerce_type %b, %int
  %r0 = hir.eq %a_typed, %b_typed : %bool
  %r1 = hir.neq %a_typed, %b_typed : %bool
  %r2 = hir.lt %a_typed, %b_typed : %bool
  %r3 = hir.gt %a_typed, %b_typed : %bool
  %r4 = hir.geq %a_typed, %b_typed : %bool
  %r5 = hir.leq %a_typed, %b_typed : %bool
  // CHECK: mir.eq %a, %b : !si.int
  // CHECK: mir.neq %a, %b : !si.int
  // CHECK: mir.lt %a, %b : !si.int
  // CHECK: mir.gt %a, %b : !si.int
  // CHECK: mir.geq %a, %b : !si.int
  // CHECK: mir.leq %a, %b : !si.int
  hir.return %r0, %r1, %r2, %r3, %r4, %r5 : (%int, %int) -> (%bool, %bool, %bool, %bool, %bool, %bool)
}

// Verify that UnifyOp forwards its operand through opaque_pack, rather than
// replacing it with a dummy !hir.any constant.
//
// CHECK-LABEL: mir.func @UnifyInOpaquePack(%T: !si.type)
hir.func @UnifyInOpaquePack(%T) -> (ctx) {
  %type_type = hir.type_type
  %opaque = hir.opaque_type
  hir.signature (%type_type) -> (%opaque)
} {
  %type_type = hir.type_type
  %coerced = hir.coerce_type %T, %type_type
  %unified = hir.unify %coerced, %T
  // CHECK: mir.opaque_pack(%T) : (!si.type)
  %packed = hir.opaque_pack(%unified)
  %opaque = hir.opaque_type
  hir.return %packed : (%type_type) -> (%opaque)
}

// CHECK-LABEL: mir.func @OpaqueTypes
hir.func @OpaqueTypes() -> () {
  hir.signature () -> ()
} {
  // CHECK: mir.constant #si.type<!si.opaque>
  %opaque_type = hir.opaque_type
  hir.return : () -> ()
}

// Opaque args block lowering — the opaque context must be resolved through
// specialization first. The function stays as HIR until then.
// CHECK-LABEL: hir.func @OpaqueArg(%a) -> (result)
hir.func @OpaqueArg(%a) -> (result) {
  %opaque = hir.opaque_type
  hir.signature (%opaque) -> (%opaque)
} {
  %opaque = hir.opaque_type
  %a0 = hir.coerce_type %a, %opaque
  // CHECK: hir.return
  hir.return %a0 : (%opaque) -> (%opaque)
}

// Verify that hir.unify in return type position is handled correctly when both
// operands resolve to the same type.
//
// CHECK-LABEL: mir.func @UnifyInReturnType
// CHECK: mir.return
hir.func @UnifyInReturnType() -> (result) {
  %int = hir.int_type
  hir.signature () -> (%int)
} {
  %int = hir.int_type
  %int2 = hir.int_type
  %ty = hir.unify %int, %int2
  %c0 = hir.constant_int 0 : %ty
  hir.return %c0 : () -> (%ty)
}

// Verify that a hir.func with unresolved typeOfArgs in a call is not lowered.
// The outer function has a block argument used as a type operand for the call,
// so shouldLower must return false for it and leave it as hir.func.
//
// CHECK-LABEL: hir.func @UnresolvedCallArgType
// CHECK-NOT: mir.func @UnresolvedCallArgType
hir.func @UnresolvedCallArgType(%T) -> () {
  %T_type = hir.type_of %T
  hir.signature (%T_type) -> ()
} {
  %x = hir.constant_int 0 : %T
  %T_type = hir.type_of %T
  hir.call @identity(%x) : (%T) -> ()
  hir.return : (%T_type) -> ()
}

// Verify that a function with a non-constant type operand on coerce_type is
// skipped by shouldLower and left as hir.func.
//
// CHECK-LABEL: hir.func @CoerceTypeNonConstant
// CHECK-NOT: mir.func @CoerceTypeNonConstant
hir.func @CoerceTypeNonConstant(%a, %ty) -> (result) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int)
} {
  %int = hir.int_type
  // CHECK: hir.coerce_type
  %r = hir.coerce_type %a, %ty
  hir.return %r : (%int, %int) -> (%int)
}

// CHECK-LABEL: mir.func @Casts
hir.func @Casts() -> (a) {
  %ta = hir.int_type
  hir.signature () -> (%ta)
} {
  // CHECK-NEXT: [[TMP1:%.+]] = mir.constant
  %a0 = mir.constant #si.int<42>
  %a1 = builtin.unrealized_conversion_cast %a0 : !si.int to !hir.any

  %ta = hir.int_type
  // CHECK: mir.return [[TMP1]]
  hir.return %a1 : () -> (%ta)
}

//===----------------------------------------------------------------------===//
// Unused block arguments
//===----------------------------------------------------------------------===//

// Arg types are derived from the return op's typeOfArgs, even when the
// corresponding block arguments are unused in the function body.

// CHECK-LABEL: mir.func @UnusedSecondArg(%a: !si.int, %b: !si.int) -> (result: !si.int)
hir.func @UnusedSecondArg(%a, %b) -> (result) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int)
} {
  %int = hir.int_type
  %a_typed = hir.coerce_type %a, %int
  // CHECK: mir.return %a
  hir.return %a_typed : (%int, %int) -> (%int)
}

// CHECK-LABEL: mir.func @UnusedFirstArg(%a: !si.int, %b: !si.int) -> (result: !si.int)
hir.func @UnusedFirstArg(%a, %b) -> (result) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int)
} {
  %int = hir.int_type
  %b_typed = hir.coerce_type %b, %int
  // CHECK: mir.return %b
  hir.return %b_typed : (%int, %int) -> (%int)
}

// CHECK-LABEL: mir.func @UnusedBothArgs(%a: !si.int, %b: !si.int) -> (result: !si.int)
hir.func @UnusedBothArgs(%a, %b) -> (result) {
  %int = hir.int_type
  hir.signature (%int, %int) -> (%int)
} {
  %int = hir.int_type
  %c42 = hir.constant_int 42 : %int
  // CHECK: mir.return
  hir.return %c42 : (%int, %int) -> (%int)
}

//===----------------------------------------------------------------------===//
// hir.coerce_to_i1 → mir.bool_to_i1
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @CoerceToI1(%a: !si.bool)
hir.func @CoerceToI1(%a) -> () {
  %bool = hir.bool_type
  hir.signature (%bool) -> ()
} {
  %bool = hir.bool_type
  %a_typed = hir.coerce_type %a, %bool
  // CHECK: mir.bool_to_i1 %a
  %i1 = hir.coerce_to_i1 %a_typed
  hir.return : (%bool) -> ()
}

//===----------------------------------------------------------------------===//
// hir.type_of conversion (replaced with dummy type constant)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @TypeOf(%a: !si.int)
hir.func @TypeOf(%a) -> () {
  %int = hir.int_type
  hir.signature (%int) -> ()
} {
  %int = hir.int_type
  %a_typed = hir.coerce_type %a, %int
  // type_of is replaced with a dummy constant; it should not appear in MIR.
  // CHECK-NOT: hir.type_of
  %ty = hir.type_of %a_typed
  hir.return : (%int) -> ()
}

//===----------------------------------------------------------------------===//
// hir.mir_constant passthrough
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @MIRConstant
hir.func @MIRConstant() -> (result) {
  %int = hir.int_type
  hir.signature () -> (%int)
} {
  // CHECK: mir.constant #si.int<99>
  %c = hir.mir_constant #si.int<99>
  %int = hir.int_type
  hir.return %c : () -> (%int)
}

//===----------------------------------------------------------------------===//
// hir.opaque_pack → mir.opaque_pack (standalone, not via UnifyInOpaquePack)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: mir.func @OpaquePack(%a: !si.int, %b: !si.bool)
hir.func @OpaquePack(%a, %b) -> (ctx) {
  %int = hir.int_type
  %bool = hir.bool_type
  %opaque = hir.opaque_type
  hir.signature (%int, %bool) -> (%opaque)
} {
  %int = hir.int_type
  %bool = hir.bool_type
  %a_typed = hir.coerce_type %a, %int
  %b_typed = hir.coerce_type %b, %bool
  // CHECK: mir.opaque_pack(%a, %b) : (!si.int, !si.bool)
  %packed = hir.opaque_pack(%a_typed, %b_typed)
  %opaque = hir.opaque_type
  hir.return %packed : (%int, %bool) -> (%opaque)
}
