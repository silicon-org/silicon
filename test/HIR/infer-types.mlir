// RUN: silicon-opt --infer-types %s | FileCheck %s

func.func private @use_type(%arg0: !hir.any)
func.func private @use_value(%arg0: !hir.any)

// CHECK-LABEL: func @TwoInferrable
func.func @TwoInferrable() {
  // CHECK: [[T:%.+]] = hir.inferrable {a}
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.inferrable {a}
  %1 = hir.inferrable {b}
  %2 = hir.unify %0, %1
  call @use_type(%0) : (!hir.any) -> ()
  call @use_type(%1) : (!hir.any) -> ()
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @TwoInferrableReversed
func.func @TwoInferrableReversed() {
  // CHECK: [[T:%.+]] = hir.inferrable {a}
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.inferrable {a}
  %1 = hir.inferrable {b}
  %2 = hir.unify %1, %0  // reversed
  call @use_type(%0) : (!hir.any) -> ()
  call @use_type(%1) : (!hir.any) -> ()
  call @use_type(%2) : (!hir.any) -> ()
  return
}


// CHECK-LABEL: func @InferrableAndConcrete1
func.func @InferrableAndConcrete1() {
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  // CHECK-NEXT: call @use_type([[T]])
  %0 = hir.int_type
  %1 = hir.inferrable
  %2 = hir.unify %0, %1
  call @use_type(%0) : (!hir.any) -> ()
  call @use_type(%1) : (!hir.any) -> ()
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @InferrableAndConcrete2
func.func @InferrableAndConcrete2() {
  // When the concrete appears after the inferrable but is a pure op with no
  // operands, it can be cloned to the inferrable's location.
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: call @use_type([[T]])
  %0 = hir.inferrable
  %1 = hir.int_type
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOps1
func.func @UnifyConcreteOps1() {
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[REF:%.+]] = hir.ref_type [[INT]]
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  %0 = hir.int_type
  %1 = hir.inferrable
  %2 = hir.ref_type %0
  %3 = hir.ref_type %1
  %4 = hir.unify %2, %3
  call @use_type(%2) : (!hir.any) -> ()
  call @use_type(%3) : (!hir.any) -> ()
  call @use_type(%4) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOps2
func.func @UnifyConcreteOps2() {
  // Even when the erased op's operands don't dominate the kept op, the
  // decomposition creates operand-level unifications at the unify site.
  // The inferrable gets resolved to int_type, producing ref_type(int_type).
  // CHECK: [[T:%.+]] = hir.int_type
  // CHECK: hir.ref_type [[T]]
  // CHECK: hir.ref_type [[T]]
  // CHECK: [[R:%.+]] = hir.ref_type [[T]]
  // CHECK: call @use_type([[R]])
  %0 = hir.int_type
  %1 = hir.ref_type %0
  %2 = hir.inferrable
  %3 = hir.ref_type %2
  %4 = hir.unify %1, %3
  call @use_type(%4) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @DecomposeUintTypeUnify
func.func @DecomposeUintTypeUnify() {
  // When two uint_type ops have different width operands and the later one's
  // operand doesn't dominate the earlier one, decompose into a width-level
  // unification: `unify(uint<a>, uint<b>)` becomes `uint<unify(a, b)>`.
  // CHECK: [[A:%.+]] = hir.constant_int
  // CHECK: hir.uint_type [[A]]
  // CHECK: [[B:%.+]] = hir.add
  // CHECK: hir.uint_type [[B]]
  // CHECK: [[U:%.+]] = hir.unify [[A]], [[B]]
  // CHECK: [[R:%.+]] = hir.uint_type [[U]]
  // CHECK: call @use_type([[R]])
  %0 = hir.int_type
  %1 = hir.constant_int 8 : %0
  %2 = hir.uint_type %1
  %3 = hir.constant_int 4 : %0
  %4 = hir.constant_int 4 : %0
  %5 = hir.add %3, %4 : %0
  %6 = hir.uint_type %5
  %7 = hir.unify %2, %6
  call @use_type(%7) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @InferIfDominates
func.func @InferIfDominates(%arg0: i1) {
  // CHECK: [[TMP:%.+]] = hir.int_type
  %0 = hir.int_type
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  // CHECK: ^bb3:
  // CHECK-NEXT: call @use_type([[TMP]])
  %1 = hir.inferrable
  %2 = hir.unify %0, %1
  call @use_type(%2) : (!hir.any) -> ()
  // CHECK-NEXT: call @use_type([[TMP]])
  %3 = hir.int_type
  %4 = hir.unify %0, %3
  call @use_type(%4) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @UnifyConcreteOpsSwapped
func.func @UnifyConcreteOpsSwapped() {
  // The RHS (ref %1) dominates the LHS (ref %0), so the pass must swap
  // keep/erase in the concrete-concrete case.
  // CHECK-NEXT: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[REF:%.+]] = hir.ref_type [[INT]]
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  // CHECK-NEXT: call @use_type([[REF]])
  %0 = hir.int_type
  %1 = hir.inferrable
  %2 = hir.ref_type %1
  %3 = hir.ref_type %0
  %4 = hir.unify %3, %2
  call @use_type(%2) : (!hir.any) -> ()
  call @use_type(%3) : (!hir.any) -> ()
  call @use_type(%4) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @InferConstantInt
func.func @InferConstantInt() {
  // CHECK: [[TY:%.+]] = hir.int_type
  // CHECK: [[TMP:%.+]] = hir.constant_int 42 : [[TY]]
  // CHECK-NEXT: call @use_value([[TMP]])
  // CHECK-NEXT: call @use_value([[TMP]])
  // CHECK-NEXT: call @use_value([[TMP]])
  %t = hir.int_type
  %0 = hir.constant_int 42 : %t
  %1 = hir.inferrable
  %2 = hir.unify %0, %1
  call @use_value(%0) : (!hir.any) -> ()
  call @use_value(%1) : (!hir.any) -> ()
  call @use_value(%2) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @InferConstantIntFromContext
func.func @InferConstantIntFromContext() {
  // When a constant_int has an inferrable type operand and that inferrable is
  // unified with a uint_type, the inferrable resolves to uint_type.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK: [[W:%.+]] = hir.constant_int 8 : [[INT]]
  // CHECK: [[UINT:%.+]] = hir.uint_type [[W]]
  // CHECK: hir.constant_int 42 : [[UINT]]
  %int = hir.int_type
  %w = hir.constant_int 8 : %int
  %uint = hir.uint_type %w
  %inf = hir.inferrable
  %v = hir.constant_int 42 : %inf
  %u = hir.unify %inf, %uint
  call @use_value(%v) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @InferConstantIntDefault
func.func @InferConstantIntDefault() {
  // When a constant_int has an inferrable type operand with no constraints,
  // the fallback assigns int_type.
  // CHECK: [[TY:%.+]] = hir.int_type
  // CHECK: hir.constant_int 42 : [[TY]]
  %inf = hir.inferrable
  %v = hir.constant_int 42 : %inf
  call @use_value(%v) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @CloneCrossBlock
func.func @CloneCrossBlock(%arg0: i1) {
  // When a concrete op is in a branch but the inferrable is in the dominator,
  // the concrete op is cloned to the inferrable's location.
  // CHECK: [[T:%.+]] = hir.bool_type
  // CHECK-NEXT: cf.cond_br
  %0 = hir.inferrable
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %1 = hir.bool_type
  %2 = hir.unify %0, %1
  cf.br ^bb3
^bb2:
  cf.br ^bb3
^bb3:
  call @use_type(%0) : (!hir.any) -> ()
  return
}

//===----------------------------------------------------------------------===//
// Structured control-flow regions
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func @UnifyInsideIfArms
func.func @UnifyInsideIfArms(%c: !hir.any, %x: !hir.any) {
  // A shared inferrable constrained by unify ops inside both arms resolves to
  // the concrete type; the arm-local unify ops dissolve.
  // CHECK: [[T:%.+]] = hir.bool_type
  %st = hir.inferrable
  %t = hir.bool_type
  // CHECK: uir.if %arg0 : [[T]]
  %r = uir.if %c : %st {
    %u = hir.unify %st, %t
    // CHECK: uir.yield %arg1 : [[T]]
    uir.yield %x : %u
  } else {
    %u = hir.unify %st, %t
    uir.yield %x : %u
  }
  // CHECK: call @use_type([[T]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @HoistTreeFromIfArm
func.func @HoistTreeFromIfArm(%c: !hir.any) {
  // A type tree defined inside an arm is materialized at the inferrable's
  // location, including the tree's cloned leaves.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 8 : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W]]
  %st = hir.inferrable
  %int = hir.int_type
  uir.if %c {
    %w = hir.constant_int 8 : %int
    %u = hir.uint_type %w
    %uu = hir.unify %st, %u
    uir.yield
  }
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @HoistTreeFromLoop
func.func @HoistTreeFromLoop() {
  // A type tree defined inside a loop body is materialized at the inferrable.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 4 : [[INT]]
  // CHECK-NEXT: [[W2:%.+]] = hir.add [[W]], [[W]] : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W2]]
  %st = hir.inferrable
  %int = hir.int_type
  uir.loop {
    %w = hir.constant_int 4 : %int
    %w2 = hir.add %w, %w : %int
    %u = hir.uint_type %w2
    %uu = hir.unify %st, %u
    uir.break
  }
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @HoistTreeFromExpr
func.func @HoistTreeFromExpr(%x: !hir.any) {
  // A type tree defined inside a uir.expr body is materialized at the
  // inferrable; the expr's result type operand resolves to the hoisted tree.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 4 : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W]]
  %st = hir.inferrable
  %int = hir.int_type
  // CHECK: uir.expr : [[U]]
  %r = uir.expr : %st {
    %w = hir.constant_int 4 : %int
    %u = hir.uint_type %w
    %uu = hir.unify %st, %u
    // CHECK: uir.yield %arg0 : [[U]]
    uir.yield %x : %uu
  }
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @HoistTreeFromPickAlternative
func.func @HoistTreeFromPickAlternative(%c: !hir.any) {
  // A type tree defined inside a pick alternative is materialized at the
  // inferrable.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 8 : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W]]
  %st = hir.inferrable
  %int = hir.int_type
  hir.pick %c [
    0 (): {
      %w = hir.constant_int 8 : %int
      %u = hir.uint_type %w
      %uu = hir.unify %st, %u
      hir.yield
    }
  ]
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @TwoArmsConstrainSameInferrable
func.func @TwoArmsConstrainSameInferrable(%c: !hir.any) {
  // Both arms constrain the same inferrable with structurally equal trees;
  // the first resolves the inferrable, the second unifies structurally.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 8 : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W]]
  // CHECK-NOT: hir.unify
  %st = hir.inferrable
  %int = hir.int_type
  uir.if %c {
    %w = hir.constant_int 8 : %int
    %u = hir.uint_type %w
    %uu = hir.unify %st, %u
    uir.yield
  } else {
    %w = hir.constant_int 8 : %int
    %u = hir.uint_type %w
    %uu = hir.unify %st, %u
    uir.yield
  }
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @NonHoistableLoopArg
func.func @NonHoistableLoopArg(%a: !hir.any, %int: !hir.any) {
  // The type tree hangs off a loop iteration argument, which is genuinely
  // local to the loop body. The unification is left unresolved for later
  // passes to diagnose.
  // CHECK: hir.inferrable
  // CHECK: hir.unify
  %st = hir.inferrable
  uir.loop (%w = %a : %int) {
    %u = hir.uint_type %w
    %uu = hir.unify %st, %u
    uir.continue %w : %int
  }
  call @use_type(%st) : (!hir.any) -> ()
  return
}

// CHECK-LABEL: func @ConcreteDominatesIntoArm
func.func @ConcreteDominatesIntoArm(%c: !hir.any) {
  // An inferrable inside an arm resolves to a concrete type defined outside.
  // CHECK: [[T:%.+]] = hir.int_type
  %t = hir.int_type
  uir.if %c {
    %i = hir.inferrable
    // CHECK: hir.constant_int 42 : [[T]]
    %v = hir.constant_int 42 : %i
    %u = hir.unify %i, %t
    uir.yield
  }
  return
}

// CHECK-LABEL: func @ConstantIntFallbackInRegion
func.func @ConstantIntFallbackInRegion(%c: !hir.any) {
  // The int_type fallback for unconstrained literals applies inside regions.
  uir.if %c {
    // CHECK: [[T:%.+]] = hir.int_type
    // CHECK: hir.constant_int 42 : [[T]]
    %i = hir.inferrable
    %v = hir.constant_int 42 : %i
    uir.yield
  }
  return
}

// CHECK-LABEL: func @HoistTreeSameBlock
func.func @HoistTreeSameBlock() {
  // Tree materialization is position-based, not region-based: a multi-level
  // tree defined after the inferrable in the same block hoists as well.
  // CHECK: [[INT:%.+]] = hir.int_type
  // CHECK-NEXT: [[W:%.+]] = hir.constant_int 8 : [[INT]]
  // CHECK-NEXT: [[U:%.+]] = hir.uint_type [[W]]
  %st = hir.inferrable
  %int = hir.int_type
  %w = hir.constant_int 8 : %int
  %u = hir.uint_type %w
  %uu = hir.unify %st, %u
  // CHECK: call @use_type([[U]])
  call @use_type(%st) : (!hir.any) -> ()
  return
}
