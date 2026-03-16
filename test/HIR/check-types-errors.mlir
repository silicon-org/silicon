// RUN: silicon-opt --check-types --split-input-file --verify-diagnostics %s

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between unit_type and int_type.
func.func @UnitVsInt() {
  %0 = hir.unit_type
  %1 = hir.int_type
  // expected-error @below {{type mismatch: `()` is not compatible with `int`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Two identical concrete types should not produce an error.
func.func @SameConcrete(%arg0: !hir.any) {
  %0 = hir.int_type
  %1 = hir.unify %0, %0
  call @use_type(%1) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Inferrable operands should not produce an error.
func.func @InferrableVsConcrete() {
  %0 = hir.inferrable
  %1 = hir.int_type
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between opaque_type and int_type.
func.func @OpaqueVsInt() {
  %0 = hir.opaque_type
  %1 = hir.int_type
  // expected-error @below {{type mismatch: `opaque` is not compatible with `int`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Both inferrable should not produce an error.
func.func @BothInferrable() {
  %0 = hir.inferrable
  %1 = hir.inferrable
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between int_type and bool_type.
func.func @IntVsBool() {
  %0 = hir.int_type
  %1 = hir.bool_type
  // expected-error @below {{type mismatch: `int` is not compatible with `bool`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between bool_type and uint_type.
func.func @BoolVsUInt() {
  %0 = hir.bool_type
  %int = hir.int_type
  %c8 = hir.constant_int 8 : %int
  %1 = hir.uint_type %c8
  // expected-error @below {{type mismatch: `bool` is not compatible with `uint`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between ref_type and int_type.
func.func @RefVsInt() {
  %inner = hir.int_type
  %0 = hir.ref_type %inner
  %1 = hir.int_type
  // expected-error @below {{type mismatch: `ref` is not compatible with `int`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between type_type and bool_type.
func.func @TypeVsBool() {
  %0 = hir.type_type
  %1 = hir.bool_type
  // expected-error @below {{type mismatch: `type` is not compatible with `bool`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between anyfunc_type and unit_type.
func.func @AnyfuncVsUnit() {
  %0 = hir.anyfunc_type
  %1 = hir.unit_type
  // expected-error @below {{type mismatch: `anyfunc` is not compatible with `()`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Type mismatch between func_type and int_type.
func.func @FuncTypeVsInt() {
  %int = hir.int_type
  %0 = hir.func_type (%int) -> (%int)
  %1 = hir.int_type
  // expected-error @below {{type mismatch: a function type is not compatible with `int`}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_value(%arg0: !hir.any)

// Integer literal with a non-integer type (e.g., unit).
func.func @IntLiteralWithUnitType() {
  %0 = hir.unit_type
  // expected-error @below {{integer literal is not compatible with type `()`}}
  %1 = hir.constant_int 42 : %0
  call @use_value(%1) : (!hir.any) -> ()
  return
}
