// RUN: silicon-opt --split-phases2="sort-only=true" %s | FileCheck %s
// RUN: silicon-opt --split-phases2="sort-only=true annotate-phases=true" %s | FileCheck %s --check-prefix=ANNOT

//===----------------------------------------------------------------------===//
// Dissolution: exprs are inlined, pins replaced by their inputs, and the
// interior ops keep their analyzed phases in the maintained phase map.

// ANNOT-LABEL: uir.func @DissolveExprsAndPins
uir.func @DissolveExprsAndPins(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // ANNOT-NOT: uir.expr
  // ANNOT: hir.mul %a, %a {{.*}}pa.phase = "-1"
  %y = uir.expr pin -1 : %t {
    %prod = hir.mul %a, %a : %t
    uir.yield %prod : %t
  }
  // ANNOT-NOT: uir.pin
  %pinY = uir.pin %y, 0 : !hir.any
  // The add consumes the former expr result directly.
  // ANNOT: [[SUM:%.+]] = hir.add {{.*}}pa.operands = ["-1", "-1", "float"]{{.*}}pa.phase = "0"
  %sum = hir.add %pinY, %pinY : %t
  // ANNOT: uir.return [[SUM]]
  uir.return %sum : %t
}

// Nested exprs dissolve inner-first; floating exprs dissolve like pinned ones.
// ANNOT-LABEL: uir.func @DissolveNestedExprs
uir.func @DissolveNestedExprs(%a: -1) -> (result: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // ANNOT-NOT: uir.expr
  // ANNOT: [[PROD:%.+]] = hir.mul %a, %a {{.*}}pa.phase = "-1"
  // ANNOT: [[SUM:%.+]] = hir.add [[PROD]], %a {{.*}}pa.phase = "-1"
  %y = uir.expr pin -1 : %t {
    %inner = uir.expr : %t {
      %prod = hir.mul %a, %a : %t
      uir.yield %prod : %t
    }
    %sum = hir.add %inner, %a : %t
    uir.yield %sum : %t
  }
  // ANNOT: uir.return [[SUM]]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Call decomposition: a call to a split callee becomes one hir.call per
// witness entry, chained through an opaque context value, each at its
// absolute phase.

uir.split_func @half(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} [
  -1: @half.0,
  0: @half.1
]

// ANNOT-LABEL: uir.func @DecomposeCall
uir.func @DecomposeCall(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // ANNOT-NOT: uir.call
  // ANNOT: [[CTX:%.+]] = hir.call @half.0(%x) : (%{{.*}}) -> (%{{.*}}) {pa.operands = {{.*}}pa.phase = "-1", pa.results = ["-1"]}
  // ANNOT: [[RES:%.+]] = hir.call @half.1(%y, [[CTX]]) : (%{{.*}}, %{{.*}}) -> (%{{.*}}) {pa.operands = {{.*}}pa.phase = "0", pa.results = ["0"]}
  %r = uir.call @half(%x, %y) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
  // ANNOT: uir.return [[RES]]
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Sum-over-index: a const loop whose body contains dyn content using the
// loop's iteration argument. One record per full activation; the break path
// pushes nothing since it precedes all later-phase content.

// CHECK-LABEL: uir.func @SumOverIndex
uir.func @SumOverIndex(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  // CHECK: uir.loop (
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      // CHECK: uir.if
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      uir.if %done {
        uir.break
      }
      uir.expr pin 1 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[REC:%.+]] = hir.opaque_pack(%arg0)
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[REC]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg0
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Branch extraction: a const if with dyn content in both arms and a const
// result. The if is rebuilt with a choice result fed per arm, the dyn
// content lands in a pick after the if, and arm-local const values cross as
// choice payloads.

// CHECK-LABEL: uir.func @IfConstDynEffects
uir.func @IfConstDynEffects(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[IF:%.+]]:2 = uir.if %c : %{{.*}}, [[CT]] {
  // CHECK:   [[A:%.+]] = hir.add %c, %c
  // CHECK:   [[C0:%.+]] = hir.choice 0([[A]])
  // CHECK:   uir.yield [[A]], [[C0]] : %{{.*}}, [[CT]]
  // CHECK: } else {
  // CHECK:   [[C1:%.+]] = hir.choice 1()
  // CHECK:   uir.yield %c, [[C1]] : %{{.*}}, [[CT]]
  // CHECK: }
  %r = uir.expr pin -1 : %t {
    %v = uir.if %c : %t {
      %a = hir.add %c, %c : %t
      uir.expr pin 1 {
        %va = hir.var : %t
        hir.write_var %va, %a : %t
        uir.yield
      }
      uir.yield %a : %t
    } else {
      uir.expr pin 1 {
        %vb = hir.var : %t
        hir.write_var %vb, %x : %t
        uir.yield
      }
      uir.yield %c : %t
    }
    uir.yield %v : %t
  }
  // CHECK: hir.pick [[IF]]#1 [0 (%arg0): {
  // CHECK:   hir.write_var %{{.*}}, %arg0
  // CHECK:   hir.yield
  // CHECK: } 1 (): {
  // CHECK:   hir.write_var %{{.*}}, %x
  // CHECK:   hir.yield
  // CHECK: }]
  // CHECK: hir.add %x, [[IF]]#0
  %out = hir.add %x, %r : %t
  uir.return %out : %t
}

//===----------------------------------------------------------------------===//
// Later-phase if result: the dyn result is removed from the if and
// reconstructed as a result of the receiving pick, whose alternatives yield
// the per-arm values.

// CHECK-LABEL: uir.func @IfDynResult
uir.func @IfDynResult(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // The if keeps only the choice result.
  // CHECK: [[IF:%.+]] = uir.if %c : [[CT:%.+]] {
  // CHECK:   [[C0:%.+]] = hir.choice 0()
  // CHECK:   uir.yield [[C0]] : [[CT]]
  // CHECK: } else {
  // CHECK:   [[C1:%.+]] = hir.choice 1()
  // CHECK:   uir.yield [[C1]] : [[CT]]
  // CHECK: }
  %r = uir.expr pin -1 : %t {
    %v = uir.if %c : %t {
      %a = uir.expr pin 1 : %t {
        %s = hir.add %x, %x : %t
        uir.yield %s : %t
      }
      uir.yield %a : %t
    } else {
      %b = uir.expr pin 1 : %t {
        %m = hir.mul %x, %x : %t
        uir.yield %m : %t
      }
      uir.yield %b : %t
    }
    uir.yield %v : %t
  }
  // CHECK: [[V:%.+]] = hir.pick [[IF]] : !hir.any [0 (): {
  // CHECK:   [[S:%.+]] = hir.add %x, %x
  // CHECK:   hir.yield [[S]]
  // CHECK: } 1 (): {
  // CHECK:   [[M:%.+]] = hir.mul %x, %x
  // CHECK:   hir.yield [[M]]
  // CHECK: }]
  // CHECK: uir.return [[V]]
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Filtering: an if-const inside a for-const with dyn content in one arm.
// The inner if extracts into a choice + pick; the pick is later-phase
// content of the loop, so the choice crosses through the record and the
// pick lands in the replicate.

// CHECK-LABEL: uir.func @Filtering
uir.func @Filtering(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      %odd = hir.eq %i, %one : %t
      // CHECK: [[CH:%.+]] = uir.if %{{.*}} : %{{.*}} {
      // CHECK:   hir.choice 0()
      // CHECK: } else {
      // CHECK:   hir.choice 1()
      // CHECK: }
      uir.if %odd {
        uir.expr pin 1 {
          %cur = hir.read_var %acc : %t
          %new = hir.add %cur, %i : %t
          hir.write_var %acc, %new : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[REC:%.+]] = hir.opaque_pack(%arg0, [[CH]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[REC]]
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0, %arg1) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg1 [0 (): {
  // CHECK:   hir.add %{{.*}}, %arg0
  // CHECK: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Nested loops: nested hit lists arise with no special casing — the inner
// loop's list is created per outer activation and crosses through the outer
// record like any snapshot.

// CHECK-LABEL: uir.func @NestedLoops
uir.func @NestedLoops(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  // CHECK: [[OUTER:%.+]] = hir.opaque_list_create
  // CHECK: uir.loop (
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      // CHECK: [[INNER:%.+]] = hir.opaque_list_create
      // CHECK: uir.loop (
      uir.loop (%j = %zero : %t) {
        %done2 = hir.geq %j, %i : %t
        uir.if %done2 {
          uir.break
        }
        uir.expr pin 1 {
          %cur = hir.read_var %acc : %t
          %sum = hir.add %i, %j : %t
          %new = hir.add %cur, %sum : %t
          hir.write_var %acc, %new : %t
          uir.yield
        }
        // The pure i+j migrates to the const phase; its snapshot crosses
        // through the inner record.
        // CHECK: [[SUM:%.+]] = hir.add %arg0, %arg1
        // CHECK: [[IREC:%.+]] = hir.opaque_pack([[SUM]])
        // CHECK-NEXT: hir.opaque_list_push [[INNER]], [[IREC]]
        %next2 = hir.add %j, %one : %t
        uir.continue %next2 : %t
      }
      // CHECK: [[OREC:%.+]] = hir.opaque_pack([[INNER]])
      // CHECK-NEXT: hir.opaque_list_push [[OUTER]], [[OREC]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[OUTER]] {
  // CHECK-NEXT: hir.replicate (%arg1) in %arg0 {
  // CHECK:   hir.read_var
  // CHECK:   hir.add %{{.*}}, %arg1
  // CHECK:   hir.write_var
  // CHECK: }
  // CHECK: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Distance 2: dyn-dyn content under a for-const rides through two
// boundaries; the intermediate replicate builds the next boundary's hit
// list, exactly like a loop would.

// CHECK-LABEL: uir.func @DistanceTwo
uir.func @DistanceTwo(%n: -1, %x: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = uir.expr pin 1 : %t {
    %v = hir.var : %t
    uir.yield %v : %t
  }
  // CHECK: [[HITS0:%.+]] = hir.opaque_list_create
  // CHECK: uir.loop (
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      uir.expr pin 2 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      // CHECK: hir.opaque_pack(%arg0)
      // CHECK-NEXT: hir.opaque_list_push [[HITS0]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: [[HITS1:%.+]] = hir.opaque_list_create
  // CHECK-NEXT: hir.replicate (%arg0) in [[HITS0]] {
  // CHECK-NEXT: [[REC1:%.+]] = hir.opaque_pack(%arg0)
  // CHECK-NEXT: hir.opaque_list_push [[HITS1]], [[REC1]]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  // CHECK: hir.replicate (%arg0) in [[HITS1]] {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg0
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = uir.expr pin 1 : %t {
    %out = hir.read_var %acc : %t
    uir.yield %out : %t
  }
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Sibling dyn blocks interleaved with const content: one record and one
// replicate per boundary, content in program order.

// CHECK-LABEL: uir.func @SiblingInterleaved
uir.func @SiblingInterleaved(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %a = hir.var : %t
  %b = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      uir.expr pin 1 {
        hir.write_var %a, %i : %t
        uir.yield
      }
      %isq = hir.mul %i, %i : %t
      uir.expr pin 1 {
        hir.write_var %b, %isq : %t
        uir.yield
      }
      // Both dyn blocks share one record: (i, i*i).
      // CHECK: hir.opaque_pack(%arg0, %{{.*}})
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0, %arg1) in [[LIST]] {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg1
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %a : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Later-phase return under earlier CF: the return rides the record inside a
// return-terminated expr carrier and becomes the replicate body's
// terminator. If it fires during expansion, it terminates the receiving
// phase's function.

// CHECK-LABEL: uir.func @ReturnInReplicate
uir.func @ReturnInReplicate(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      uir.expr pin 1 {
        %s = hir.add %x, %i : %t
        uir.return %s : %t
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: [[S:%.+]] = hir.add %x, %arg0
  // CHECK-NEXT: uir.return [[S]]
  // CHECK-NEXT: }
  uir.return %x : %t
}

// Later-phase return under an earlier if: the return lands in a pick
// alternative; the other alternative yields and execution falls through to
// the trailing return.
// CHECK-LABEL: uir.func @ReturnInPick
uir.func @ReturnInPick(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: [[IF:%.+]] = uir.if %c : [[CT:%.+]] {
  // CHECK:   hir.choice 0()
  // CHECK: } else {
  // CHECK:   hir.choice 1()
  // CHECK: }
  uir.expr pin -1 {
    uir.if %c {
      uir.expr pin 1 {
        %s = hir.mul %x, %x : %t
        uir.return %s : %t
      }
      uir.yield
    }
    uir.yield
  }
  // CHECK: hir.pick [[IF]] [0 (): {
  // CHECK-NEXT: [[S:%.+]] = hir.mul %x, %x
  // CHECK-NEXT: uir.return [[S]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK: uir.return %x
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// Hoisting: a const write under a dyn if executes unconditionally per the
// D-rule and hoists to before the if.

// CHECK-LABEL: uir.func @HoistConstWrite
uir.func @HoistConstWrite(%c: 1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %v = hir.var : %t
  // CHECK: hir.write_var
  // CHECK-NEXT: uir.if %c {
  uir.expr pin 1 {
    uir.if %c {
      uir.expr pin -1 {
        hir.write_var %v, %x : %t
        uir.yield
      }
      uir.yield
    }
    uir.yield
  }
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// Hoisting a callee's earlier half: after call decomposition, the phase-0
// half of a call under a phase-1 if hoists out; the phase-1 half stays
// conditional and consumes the hoisted context.

uir.split_func @f(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} [
  -1: @f.0,
  0: @f.1
]

// CHECK-LABEL: uir.func @HoistCallHalf
uir.func @HoistCallHalf(%c: 1, %x: 0, %k: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: [[CTX:%.+]] = hir.call @f.0(%k)
  // CHECK-NEXT: uir.if %c {
  // CHECK: hir.call @f.1(%x, [[CTX]])
  // CHECK: }
  uir.expr pin 1 {
    uir.if %c {
      %r = uir.call @f(%k, %x) : (%t, %t) -> (%t) (!hir.any, !hir.any) -> !hir.any [-1, 0] -> [0]
      uir.yield
    }
    uir.yield
  }
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// Hoist-before-extract: a const increment under a dyn if inside a for-const
// runs once per iteration, before the if. It hoists into the loop body at
// its boundary, while the dyn if extracts into the replicate.

// CHECK-LABEL: uir.func @HoistThenExtract
uir.func @HoistThenExtract(%n: -1, %c: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %k = uir.expr pin -1 : %t {
    %v = hir.var : %t
    uir.yield %v : %t
  }
  %d = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  // CHECK: uir.loop (
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        uir.break
      }
      // The const increment sits in the loop body, unconditionally.
      // CHECK: hir.read_var
      // CHECK-NEXT: hir.add
      // CHECK-NEXT: hir.write_var
      // CHECK: hir.opaque_pack()
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      uir.expr pin 1 {
        uir.if %c {
          uir.expr pin -1 {
            %cur = hir.read_var %k : %t
            %inc = hir.add %cur, %one : %t
            hir.write_var %k, %inc : %t
            uir.yield
          }
          %kv = hir.read_var %k : %t
          hir.write_var %d, %kv : %t
          uir.yield
        }
        uir.yield
      }
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate () in [[LIST]] {
  // CHECK-NEXT: uir.if %c {
  // CHECK:   hir.read_var
  // CHECK:   hir.write_var
  // CHECK: }
  // CHECK: hir.yield
  // CHECK: }
  %r = hir.read_var %d : %t
  uir.return %r : %t
}
