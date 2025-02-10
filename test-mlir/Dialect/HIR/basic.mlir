// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

func.func @Foo(%arg0: !hir.type, %arg1: !hir.type, %arg2: !hir.type) {
  hir.constant_int 42
  hir.inferrable : !hir.type
  hir.int_type
  hir.ref_type %arg0
  hir.unify %arg0, %arg1 : !hir.type
  hir.let "x" : %arg0
  hir.store %arg0, %arg1 : %arg2
  return
}

hir.int_type {x = #hir.int<42>}
