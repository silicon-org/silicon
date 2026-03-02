// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

func.func @TypeConstructors(%value: !hir.any, %type: !hir.any) {
  hir.unit_type
  hir.int_type
  hir.uint_type %value
  hir.anyfunc_type
  hir.ref_type %type
  return
}

func.func @ValueTyping() {
  %0 = hir.constant_int 42
  %1 = hir.type_of %0
  %2 = hir.int_type
  %3 = hir.coerce_type %0, %2
  return
}

func.func @Foo(%arg0: !hir.any, %arg1: !hir.any, %arg2: !hir.any, %arg3: i1) {
  hir.constant_int 42
  hir.constant_unit
  hir.inferrable
  hir.unify %arg0, %arg1
  hir.binary %arg0, %arg1 : %arg2
  hir.let "x" : %arg0
  hir.store %arg0, %arg1 : %arg2
  hir.const_br ^bb1(%arg0 : !hir.any)
^bb1(%0: !hir.any):
  hir.const_cond_br %arg3, ^bb1(%0 : !hir.any), ^bb2
^bb2:
  return
}

hir.int_type {x = #hir.int<42>}

%int_type = hir.int_type
%c42_int = hir.constant_int 42

hir.specialize_func @foo() -> ()
hir.specialize_func @foo(%int_type) -> (%int_type)
hir.specialize_func @foo(%int_type) -> (%int_type), %int_type, %c42_int

hir.call @foo() : () -> ()
hir.call @foo(%int_type) : (%int_type) -> (%int_type)
hir.call @foo(%int_type, %c42_int) : (%int_type, %int_type) -> (%int_type, %int_type)

// Test HIR function operations with symbol visibility
hir.func @public_visibility1() -> () {}
hir.func public @public_visibility2() -> () {}
hir.func private @private_visibility() -> () {}
hir.func nested @nested_visibility() -> () {}

// Test return with operands
hir.func @ReturnWithOperands() -> (result) {
  %t = hir.int_type
  %0 = hir.constant_int 42
  hir.return(%0) : (%t)
}

hir.unified_func @UnifiedSimple(%a: 0, %b: -1) -> (result: 0) {
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  %0 = hir.type_of %a
  hir.unified_return(%a) : (%0)
}
hir.unified_call @UnifiedSimple(%c42_int, %c42_int) : (%int_type, %int_type) -> (%int_type) (!hir.any, !hir.any) -> !hir.any [0, -1] -> [0]
hir.unified_call @UnifiedSimple(%c42_int, %c42_int) : (%int_type, %int_type) -> (%int_type) (!hir.any, !hir.any) -> !hir.any [0, -1] -> [0]

hir.expr {
  hir.yield
}
hir.expr : !hir.any, !hir.any {
  %0 = hir.int_type
  %1 = hir.anyfunc_type
  hir.yield %0, %1 : !hir.any, !hir.any
}

// split_func with args, results, signature, and phase map
hir.split_func @SplitExample(a: -3, b: -1) -> (c: 0) {
^bb0(%a_arg: !hir.any, %b_arg: !hir.any):
  %st0 = hir.int_type
  hir.signature (%st0, %st0) -> (%st0)
} [
  -3: @SplitExample.0,
  -1: @SplitExample.1,
  0: @SplitExample.2
]

// split_func with no args/results
hir.split_func @SplitNoArgs() -> () {
  hir.signature () -> ()
} [
  0: @SplitNoArgs.0
]

// multiphase_func with first/last args
hir.multiphase_func @Multi(last a, first b) -> (out) [
  @Multi.0,
  @Multi.1
]

// multiphase_func with no args
hir.multiphase_func @MultiNoArgs() -> (out) [
  @MultiNoArgs.0,
  @MultiNoArgs.1,
  @MultiNoArgs.2
]

// opaque ops
hir.opaque_type
%opaque_a = hir.int_type
%opaque_b = hir.constant_int 42
%packed = hir.opaque_pack (%opaque_a, %opaque_b)
%unp_x, %unp_y = hir.opaque_unpack %packed : !hir.any, !hir.any
