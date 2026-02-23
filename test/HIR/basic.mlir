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
hir.func @public_visibility1 {}
hir.func public @public_visibility2 {}
hir.func private @private_visibility {}
hir.func nested @nested_visibility {}

hir.unified_func @UnifiedSimple [0, -1] -> [0] attributes {argNames = ["a", "b"]} {
^bb0(%a: !hir.any, %b: !hir.any):
  %0 = hir.int_type
  hir.unified_signature (%0, %0) -> (%0)
} {
  hir.unified_return
}
hir.unified_call @UnifiedSimple(%c42_int, %c42_int) : (!hir.any, !hir.any) -> (!hir.any) [0, -1] -> [0]
hir.checked_call @UnifiedSimple(%c42_int, %c42_int) : (%int_type, %int_type) -> (%int_type) -> (!hir.any) [0, -1] [0]

hir.expr {
  hir.yield
}
hir.expr : !hir.any, !hir.any {
  %0 = hir.int_type
  %1 = hir.anyfunc_type
  hir.yield %0, %1 : !hir.any, !hir.any
}
