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

// -----

// Type mismatch inside a structured control-flow arm.
func.func @MismatchInsideIfArm(%c: !hir.any) {
  %0 = hir.unit_type
  uir.if %c {
    %1 = hir.int_type
    // expected-error @below {{type mismatch: `()` is not compatible with `int`}}
    %2 = hir.unify %0, %1
    uir.yield
  }
  return
}

// -----

// Integer literal with a non-integer type inside a loop body.
func.func @IntLiteralWithUnitTypeInLoop() {
  %0 = hir.unit_type
  uir.loop {
    // expected-error @below {{integer literal is not compatible with type `()`}}
    %1 = hir.constant_int 42 : %0
    uir.break
  }
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Two same-kind type constructors are not a provable mismatch: their width
// operands may still unify. InferTypes decomposes such pairs; a genuine width
// mismatch surfaces as a constant-level unify instead.
func.func @SameKindUIntNoError(%w1: !hir.any, %w2: !hir.any) {
  %0 = hir.uint_type %w1
  %1 = hir.uint_type %w2
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// A unify of two same-kind constants with different values is a provable
// mismatch. This is the shape InferTypes produces when decomposing
// `unify(uint<8>, uint<4>)` into `uint<unify(8, 4)>`.
func.func @ConstWidthMismatch() {
  %int = hir.int_type
  %a = hir.constant_int 8 : %int
  %b = hir.constant_int 4 : %int
  // expected-error @below {{type mismatch: constant `8` is not compatible with constant `4`}}
  %u = hir.unify %a, %b
  call @use_type(%u) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// The same leaf rule applies to mir_constant values.
func.func @MirConstantMismatch() {
  %a = hir.mir_constant #si.int<8>
  %b = hir.mir_constant #si.int<4>
  // expected-error @below {{type mismatch: constant `8` is not compatible with constant `4`}}
  %u = hir.unify %a, %b
  call @use_type(%u) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Constants of different op kinds may denote the same value, so they are not
// a provable mismatch even with different spellings.
func.func @CrossKindConstNoError() {
  %int = hir.int_type
  %a = hir.mir_constant #si.int<8>
  %b = hir.constant_int 8 : %int
  %u = hir.unify %a, %b
  call @use_type(%u) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Mismatched constant booleans are provably unequal.
func.func @ConstantBoolMismatch() {
  %a = hir.constant_bool <true>
  %b = hir.constant_bool <false>
  // expected-error @below {{type mismatch: constant `true` is not compatible with constant `false`}}
  %u = hir.unify %a, %b
  call @use_type(%u) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// Two func_type ops with different arities are structurally incompatible:
// same kind, but differing operand-segment attributes.
func.func @FuncTypeArityMismatch() {
  %int = hir.int_type
  %0 = hir.func_type (%int) -> (%int)
  %1 = hir.func_type (%int, %int) -> (%int)
  // expected-error @below {{type mismatch: a function type is not compatible with a function type}}
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// -----

func.func private @use_type(%arg0: !hir.any)

// An unresolved unify with an inferrable operand inside an arm produces no
// error; later pipeline stages may still resolve it.
func.func @InferrableInsideArmNoError(%c: !hir.any) {
  %0 = hir.int_type
  uir.if %c {
    %1 = hir.inferrable
    %2 = hir.unify %1, %0
    uir.yield
  }
  return
}
