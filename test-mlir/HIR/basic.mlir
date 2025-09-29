// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

func.func @TypeConstructors(%value: !hir.value, %type: !hir.type) {
  hir.int_type
  hir.uint_type %value
  hir.anyfunc_type
  hir.ref_type %type
  hir.const_type %type
  return
}

func.func @ValueTyping() {
  %0 = hir.constant_int 42
  %1 = hir.type_of %0 : !hir.value
  %2 = hir.int_type
  %3 = hir.coerce_type %0, %2 : !hir.value
  return
}

func.func @Foo(%arg0: !hir.type, %arg1: !hir.type, %arg2: !hir.type, %arg3: i1, %arg4: !hir.const<!hir.type>) {
  hir.constant_int 42
  hir.constant_unit
  hir.inferrable : !hir.type
  hir.unify %arg0, %arg1 : !hir.type
  hir.let "x" : %arg0
  hir.store %arg0, %arg1 : %arg2
  hir.const_wrap %arg0 : !hir.type
  hir.const_unwrap %arg4 : <!hir.type>
  hir.const_br ^bb1(%arg0 : !hir.type)
^bb1(%0: !hir.type):
  hir.const_cond_br %arg3, ^bb1(%0 : !hir.type), ^bb2
^bb2:
  return
}

hir.int_type {x = #hir.int<42>}

%int_type = hir.int_type
%c42_int = hir.constant_int 42

hir.specialize_func @foo() -> ()
hir.specialize_func @foo(%int_type) -> (%int_type)
hir.specialize_func @foo(%int_type) -> (%int_type), %int_type, %c42_int : !hir.type, !hir.value

%foo_type = hir.func_type () -> ()
%foo = hir.constant_func @foo : %foo_type
hir.call %foo() : () -> ()
hir.call %foo(%int_type) : (!hir.type) -> (!hir.type)
hir.call %foo(%int_type, %c42_int) : (!hir.type, !hir.value) -> (!hir.type, !hir.value)

// Test HIR function operations with symbol visibility
hir.func @public_visibility1 {}
hir.func public @public_visibility2 {}
hir.func private @private_visibility {}
hir.func nested @nested_visibility {}

hir.unchecked_func @UncheckedSimple {
  %0 = hir.int_type
  %1 = hir.unchecked_arg "a", %0 : !hir.type
  %2 = hir.unchecked_arg "b", %0 : !hir.type
  hir.unchecked_signature (%1, %2 : !hir.value, !hir.value) -> (%0 : !hir.type)
} {
  hir.unchecked_return
}
hir.unchecked_call @UncheckedSimple(%c42_int, %c42_int) : (!hir.value, !hir.value) -> (!hir.value)

hir.expr {
  hir.yield
}
hir.expr : !hir.type, !hir.type {
  %0 = hir.int_type
  %1 = hir.anyfunc_type
  hir.yield %0, %1 : !hir.type, !hir.type
}
