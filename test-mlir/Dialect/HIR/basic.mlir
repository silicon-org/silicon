// RUN: silicon-opt --verify-roundtrip --verify-diagnostics %s

func.func @Foo(%arg0: !hir.type, %arg1: !hir.type, %arg2: !hir.type, %arg3: i1, %arg4: !hir.const<!hir.type>) {
  hir.constant_int 42
  hir.constant_unit
  hir.inferrable : !hir.type
  hir.int_type
  hir.ref_type %arg0
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
