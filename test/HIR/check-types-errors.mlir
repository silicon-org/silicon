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

// Both inferrable should not produce an error.
func.func @BothInferrable() {
  %0 = hir.inferrable
  %1 = hir.inferrable
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}
