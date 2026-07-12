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

// Dead pure content — an unused if result, an unused loop break value, and
// an undemanded floating expr — must sort like live content: undemanded
// region results get their provider-derived phase, dead pure ops float,
// and everything stays in place.
// CHECK-LABEL: uir.func @DeadPureContent
uir.func @DeadPureContent(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  // CHECK: uir.if %c
  // CHECK: hir.add %x, %x
  %r = uir.if %c : %t {
    %a = hir.add %x, %x : %t
    uir.yield %a : %t
  } else {
    uir.yield %x : %t
  }
  // CHECK: uir.loop
  // CHECK: hir.sub %x, %x
  %s = uir.loop () : %t {
    %b = hir.sub %x, %x : %t
    uir.break %b : %t
  }
  // CHECK: hir.mul %x, %x
  %dead = uir.expr : %t {
    %m = hir.mul %x, %x : %t
    uir.yield %m : %t
  }
  // CHECK: uir.return %x
  uir.return %x : %t
}

// Dead CF results must also compose with later-phase content: extraction
// partitions results by their provider-derived phase, so the undemanded
// result — provided here at the op's own phase — stays on the op while the
// later-phase effects extract.
// CHECK-LABEL: uir.func @DeadIfResultLaterContent
uir.func @DeadIfResultLaterContent(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %va = hir.var : %t
  // The if keeps the dead result alongside the new choice result.
  // CHECK: [[IF:%.+]]:2 = uir.if %c :
  // CHECK: hir.pick [[IF]]#1 [0 (%arg0): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  %r = uir.if %c : %t {
    %a = hir.add %x, %x : %t
    uir.expr pin 1 {
      hir.write_var %va, %a : %t
      uir.yield
    }
    uir.yield %a : %t
  } else {
    uir.yield %x : %t
  }
  uir.return %x : %t
}

// CHECK-LABEL: uir.func @DeadLoopResultLaterContent
uir.func @DeadLoopResultLaterContent(%n: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %va = hir.var : %t
  // The loop keeps the dead break-value result; the record chain extracts
  // as usual.
  // CHECK: hir.opaque_list_create
  // CHECK: uir.loop (%arg0 = %{{.*}}) : %{{.*}} {
  // CHECK: uir.break %arg0
  %s = uir.loop (%i = %zero : %t) : %t {
    %done = hir.geq %i, %n : %t
    uir.if %done {
      uir.break %i : %t
    }
    uir.expr pin 1 {
      hir.write_var %va, %x : %t
      uir.yield
    }
    %next = hir.add %i, %one : %t
    uir.continue %next : %t
  }
  // CHECK: hir.replicate
  // CHECK: hir.write_var
  uir.return %x : %t
}

// A dead CF result whose provider is itself later-phase content gets the
// provider-derived phase from the coverage stamp and choice-ifies exactly
// like a demanded result: the value reconstructs in the receiving pick
// (unused there), and no stale operand stays behind on the rebuilt op.
// CHECK-LABEL: uir.func @DeadIfResultLaterValue
uir.func @DeadIfResultLaterValue(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %va = hir.var : %t
  // CHECK: [[IF:%.+]] = uir.if %c : {{.*}} {
  // CHECK: hir.choice 0()
  // CHECK: hir.choice 1()
  // CHECK: hir.pick [[IF]] : !hir.any [0 (): {
  // CHECK-NEXT: hir.read_var
  %r = uir.if %c : %t {
    %a = uir.expr pin 1 : %t {
      %v = hir.read_var %va : %t
      uir.yield %v : %t
    }
    uir.yield %a : %t
  } else {
    uir.yield %x : %t
  }
  uir.return %x : %t
}

// CHECK-LABEL: uir.func @DeadLoopResultLaterValue
uir.func @DeadLoopResultLaterValue(%n: 0, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %va = hir.var : %t
  // CHECK: [[LOOP:%.+]] = uir.loop (%arg0 = %{{.*}}) : {{.*}} {
  // CHECK: hir.choice 0()
  // CHECK: uir.break
  // CHECK: hir.pick [[LOOP]] : !hir.any [0 (): {
  // CHECK-NEXT: hir.read_var
  %s = uir.loop (%i = %zero : %t) : %t {
    %done = hir.geq %i, %n : %t
    uir.if %done {
      %b = uir.expr pin 1 : %t {
        %v = hir.read_var %va : %t
        uir.yield %v : %t
      }
      uir.break %b : %t
    }
    %next = hir.add %i, %one : %t
    uir.continue %next : %t
  }
  uir.return %x : %t
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
// loop's iteration argument. Every activation pushes exactly one record:
// the break path an empty exited() link, full activations a continued link
// carrying the iteration argument.

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
      // CHECK-NEXT: [[XC:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[XREC:%.+]] = hir.opaque_pack([[XC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[XREC]]
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
      // CHECK: [[CC:%.+]] = hir.choice 1(%arg0)
      // CHECK-NEXT: [[REC:%.+]] = hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[REC]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg1
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
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
      // CHECK: uir.if
      // CHECK-NEXT: hir.choice 0()
      // CHECK-NEXT: hir.opaque_pack
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
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
      // CHECK: [[CC:%.+]] = hir.choice 1(%arg0, [[CH]])
      // CHECK-NEXT: [[REC:%.+]] = hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[REC]]
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1, %arg2): {
  // CHECK-NEXT: hir.pick %arg2 [0 (): {
  // CHECK:   hir.add %{{.*}}, %arg1
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
        // through the inner record's continued link.
        // CHECK: [[SUM:%.+]] = hir.add %arg0, %arg1
        // CHECK: [[ICC:%.+]] = hir.choice 1([[SUM]])
        // CHECK-NEXT: [[IREC:%.+]] = hir.opaque_pack([[ICC]])
        // CHECK-NEXT: hir.opaque_list_push [[INNER]], [[IREC]]
        %next2 = hir.add %j, %one : %t
        uir.continue %next2 : %t
      }
      // CHECK: [[OCC:%.+]] = hir.choice 1([[INNER]])
      // CHECK-NEXT: [[OREC:%.+]] = hir.opaque_pack([[OCC]])
      // CHECK-NEXT: hir.opaque_list_push [[OUTER]], [[OREC]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[OUTER]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: hir.replicate (%arg2) in %arg1 {
  // CHECK-NEXT: hir.pick %arg2 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg3): {
  // CHECK:   hir.read_var
  // CHECK:   hir.add %{{.*}}, %arg3
  // CHECK:   hir.write_var
  // CHECK: }]
  // CHECK: }
  // CHECK: }]
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
      // CHECK: uir.if
      // CHECK-NEXT: hir.choice 0()
      // CHECK-NEXT: hir.opaque_pack
      // CHECK-NEXT: hir.opaque_list_push [[HITS0]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      uir.if %done {
        uir.break
      }
      uir.expr pin 2 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      // CHECK: [[CC:%.+]] = hir.choice 1(%arg0)
      // CHECK-NEXT: hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[HITS0]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // The intermediate replicate re-chains: its pick re-tags the exit choice
  // for the next boundary's records.
  // CHECK: [[HITS1:%.+]] = hir.opaque_list_create
  // CHECK-NEXT: hir.replicate (%arg0) in [[HITS0]] {
  // CHECK-NEXT: [[CH1:%.+]] = hir.pick %arg0 : !hir.any [0 (): {
  // CHECK-NEXT: hir.choice 0()
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: hir.choice 1(%arg1)
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: [[REC1:%.+]] = hir.opaque_pack([[CH1]])
  // CHECK-NEXT: hir.opaque_list_push [[HITS1]], [[REC1]]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  // CHECK: hir.replicate (%arg0) in [[HITS1]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg1
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
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
      // Both dyn blocks share one continued link: (i, i*i).
      // CHECK: [[CC:%.+]] = hir.choice 1(%arg0, %{{.*}})
      // CHECK-NEXT: hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      %next = hir.add %i, %one : %t
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1, %arg2): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg1
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg2
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %a : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Break-value choice: a const loop whose break carries a dyn value yields a
// choice instead; the pick after the loop reconstructs the value from the
// break site's samples.

// CHECK-LABEL: uir.func @BreakValue
uir.func @BreakValue(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[CH:%.+]] = uir.loop (%arg0 = %{{.*}}) : [[CT]] {
  // CHECK:   uir.if
  // CHECK:     [[C0:%.+]] = hir.choice 0(%arg0)
  // CHECK:     uir.break [[C0]] : [[CT]]
  // CHECK:   }
  // CHECK: }
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %ksq = hir.mul %k, %k : %t
      %done = hir.geq %ksq, %n : %t
      uir.if %done {
        %v = uir.expr pin 1 : %t {
          %s = hir.add %x, %k : %t
          uir.yield %s : %t
        }
        uir.break %v : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK: [[Y:%.+]] = hir.pick [[CH]] : !hir.any [0 (%arg0): {
  // CHECK-NEXT: [[S:%.+]] = hir.add %x, %arg0
  // CHECK-NEXT: hir.yield [[S]]
  // CHECK-NEXT: }]
  // CHECK: uir.return [[Y]]
  uir.return %y : %t
}

// Two break sites providing distinct dyn values: one choice, two tags, two
// pick alternatives.
// CHECK-LABEL: uir.func @TwoBreakValues
uir.func @TwoBreakValues(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  // CHECK: [[CH:%.+]] = uir.loop (
  // CHECK:   hir.choice 0(%arg0)
  // CHECK:   hir.choice 1(%arg0)
  // CHECK: }
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %v = uir.expr pin 1 : %t {
          %s = hir.add %x, %k : %t
          uir.yield %s : %t
        }
        uir.break %v : %t
      }
      %big = hir.geq %k, %one : %t
      uir.if %big {
        %w = uir.expr pin 1 : %t {
          %m = hir.mul %x, %k : %t
          uir.yield %m : %t
        }
        uir.break %w : %t
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK: hir.pick [[CH]] : !hir.any [0 (%arg0): {
  // CHECK:   hir.add %x, %arg0
  // CHECK: } 1 (%arg0): {
  // CHECK:   hir.mul %x, %arg0
  // CHECK: }]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Partial activation: a loop exit after later-phase content. Both the exit
// and the fall-through push a record ending in an exit-path choice, and the
// receiving replicate dispatches on it with a pick (empty on both paths
// here, since no content follows the exit).

// CHECK-LABEL: uir.func @ExitAfterContent
uir.func @ExitAfterContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %v, %i : %t
        uir.yield
      }
      %done = hir.geq %i, %n : %t
      // The break path ran the write: it pushes a record tagged exited.
      // CHECK: uir.if
      // CHECK-NEXT: [[C0:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[R0:%.+]] = hir.opaque_pack(%arg0, [[C0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R0]]
      // CHECK-NEXT: uir.break
      uir.if %done {
        uir.break
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[C1:%.+]] = hir.choice 1()
      // CHECK-NEXT: [[R1:%.+]] = hir.opaque_pack(%arg0, [[C1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R1]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0, %arg1) in [[LIST]] {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  // CHECK-NEXT: hir.pick %arg1 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Multi-exit chain: content A, exit 1, content B, exit 2, content C. Each
// exit pushes its own record: the shared prefix, one continued link per
// divergence it fell past, ending in its exit tag. The replicate mirrors
// the chain with nested picks.

// CHECK-LABEL: uir.func @MultiExitChain
uir.func @MultiExitChain(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %a = hir.var : %t
  %b = hir.var : %t
  %c = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      uir.expr pin 1 {
        hir.write_var %a, %i : %t
        uir.yield
      }
      %d1 = hir.geq %i, %n : %t
      // Exit 1 ran only A.
      // CHECK: uir.if
      // CHECK-NEXT: [[E1:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[R1:%.+]] = hir.opaque_pack(%arg0, [[E1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R1]]
      // CHECK-NEXT: uir.break
      uir.if %d1 {
        uir.break
      }
      // CHECK: [[ISQ:%.+]] = hir.mul %arg0, %arg0
      %isq = hir.mul %i, %i : %t
      uir.expr pin 1 {
        hir.write_var %b, %isq : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      %d2 = hir.eq %i, %one : %t
      // Exit 2 ran A and B: its chain continues past divergence 1 with B's
      // samples and exits at divergence 2.
      // CHECK: uir.if
      // CHECK-NEXT: [[E2:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[K2:%.+]] = hir.choice 1([[ISQ]], [[E2]])
      // CHECK-NEXT: [[R2:%.+]] = hir.opaque_pack(%arg0, [[K2]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R2]]
      // CHECK-NEXT: uir.continue
      uir.if %d2 {
        uir.continue %next : %t
      }
      uir.expr pin 1 {
        hir.write_var %c, %i : %t
        uir.yield
      }
      // The full activation continues past both divergences.
      // CHECK: [[E3:%.+]] = hir.choice 1()
      // CHECK-NEXT: [[K3:%.+]] = hir.choice 1([[ISQ]], [[E3]])
      // CHECK-NEXT: [[R3:%.+]] = hir.opaque_pack(%arg0, [[K3]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R3]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // C samples the deduplicated prefix field %arg0 from the outermost level.
  // CHECK: hir.replicate (%arg0, %arg1) in [[LIST]] {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  // CHECK-NEXT: hir.pick %arg1 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg2, %arg3): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg2
  // CHECK-NEXT: hir.pick %arg3 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %c : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// The canonical partial activation: a for-const with a leading contentless
// bounds-check break (an empty exited() link), dyn content A, a const
// continue divergence, and dyn content B. Bounds-break activations stamp
// nothing, continue activations stamp only A, full activations stamp A
// then B.
// for const i in 0..n { x += 1; const { if i == 2 { continue } } x *= 2; }

// CHECK-LABEL: uir.func @CanonicalContinue
uir.func @CanonicalContinue(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %two = hir.constant_int 2 : %t
  %v = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      // The bounds-check exit runs no content: an empty exited() record.
      // CHECK: uir.if
      // CHECK-NEXT: [[B0:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[BR:%.+]] = hir.opaque_pack([[B0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[BR]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      uir.if %done {
        uir.break
      }
      uir.expr pin 1 {
        %c = hir.read_var %v : %t
        %c1 = hir.add %c, %one : %t
        hir.write_var %v, %c1 : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      %skip = hir.eq %i, %two : %t
      // The skip exit fell past the bounds divergence: its exited() link
      // rides in that divergence's continued link.
      // CHECK: uir.if
      // CHECK-NEXT: [[C0:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[K0:%.+]] = hir.choice 1([[C0]])
      // CHECK-NEXT: [[R0:%.+]] = hir.opaque_pack([[K0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R0]]
      // CHECK-NEXT: uir.continue
      uir.if %skip {
        uir.continue %next : %t
      }
      uir.expr pin 1 {
        %d = hir.read_var %v : %t
        %d2 = hir.mul %d, %two : %t
        hir.write_var %v, %d2 : %t
        uir.yield
      }
      // CHECK: [[C1:%.+]] = hir.choice 1()
      // CHECK-NEXT: [[K1:%.+]] = hir.choice 1([[C1]])
      // CHECK-NEXT: [[R1:%.+]] = hir.opaque_pack([[K1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R1]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.pick %arg1 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.mul
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %v : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A divergence with no later-phase content inside accepts arbitrarily
// nested exits: each exit terminator pushes an empty exited() link in
// place, and the receiving pick gets one empty alternative per exit.

// CHECK-LABEL: uir.func @NestedExitsNoContent
uir.func @NestedExitsNoContent(%n: -1, %x: 0) -> (result: 0) {
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
      %big = hir.geq %i, %one : %t
      // CHECK: uir.if
      // CHECK-NEXT: uir.if
      // CHECK-NEXT: [[X0:%.+]] = hir.choice 0()
      // CHECK-NEXT: hir.opaque_pack([[X0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      // CHECK-NEXT: [[X1:%.+]] = hir.choice 1()
      // CHECK-NEXT: hir.opaque_pack([[X1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      uir.if %done {
        uir.if %big {
          uir.break
        }
        uir.break
      }
      uir.expr pin 1 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[CC:%.+]] = hir.choice 2(%arg0)
      // CHECK-NEXT: hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 2 (%arg1): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg1
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// A nested exit on a rejoining arm of a contentless divergence: the exit
// pushes its empty link where it fires, and the rejoining path falls
// through to the continue like any full activation.

// CHECK-LABEL: uir.func @NestedExitInRejoiningArm
uir.func @NestedExitInRejoiningArm(%n: -1, %x: 0) -> (result: 0) {
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
      %big = hir.geq %i, %one : %t
      // CHECK: uir.if
      // CHECK-NEXT: [[X0:%.+]] = hir.choice 0()
      // CHECK-NEXT: hir.opaque_pack([[X0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: } else {
      // CHECK-NEXT: uir.if
      // CHECK-NEXT: [[X1:%.+]] = hir.choice 1()
      // CHECK-NEXT: hir.opaque_pack([[X1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.break
      // CHECK-NEXT: }
      uir.if %done {
        uir.break
      } else {
        uir.if %big {
          uir.break
        }
        uir.yield
      }
      uir.expr pin 1 {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[CC:%.+]] = hir.choice 2(%arg0)
      // CHECK-NEXT: hir.opaque_pack([[CC]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 2 (%arg1): {
  // CHECK-NEXT: hir.read_var
  // CHECK-NEXT: hir.add %{{.*}}, %arg1
  // CHECK-NEXT: hir.write_var
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %acc : %t
  uir.return %r : %t
}

//===----------------------------------------------------------------------===//
// Break-value choice combined with partial activation: the break site
// pushes both its activation record (ending in its exit tag) and its
// break-value choice; the replicate consumes all records before the pick
// reconstructs the exit value.

// CHECK-LABEL: uir.func @BreakValueWithPartialActivation
uir.func @BreakValueWithPartialActivation(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  // CHECK: [[CH:%.+]] = uir.loop (%arg0 = %{{.*}}) :
  %y = uir.expr pin -1 : %t {
    %r = uir.loop (%k = %zero : %t) : %t {
      uir.expr pin 1 {
        hir.write_var %v, %k : %t
        uir.yield
      }
      %done = hir.geq %k, %n : %t
      // CHECK: uir.if
      // CHECK-NEXT: [[BV:%.+]] = hir.choice 0(%arg0)
      // CHECK-NEXT: [[E0:%.+]] = hir.choice 0()
      // CHECK-NEXT: [[R0:%.+]] = hir.opaque_pack(%arg0, [[E0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R0]]
      // CHECK-NEXT: uir.break [[BV]]
      uir.if %done {
        %w = uir.expr pin 1 : %t {
          %s = hir.add %x, %k : %t
          uir.yield %s : %t
        }
        uir.break %w : %t
      }
      %next = hir.add %k, %one : %t
      // CHECK: [[E1:%.+]] = hir.choice 1()
      // CHECK-NEXT: [[R1:%.+]] = hir.opaque_pack(%arg0, [[E1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R1]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield %r : %t
  }
  // CHECK: hir.replicate (%arg0, %arg1) in [[LIST]] {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg0
  // CHECK-NEXT: hir.pick %arg1 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  // CHECK: [[Y:%.+]] = hir.pick [[CH]] : !hir.any [0 (%arg0): {
  // CHECK-NEXT: [[S:%.+]] = hir.add %x, %arg0
  // CHECK-NEXT: hir.yield [[S]]
  // CHECK-NEXT: }]
  // CHECK: uir.return [[Y]]
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Content inside an exit-carrying arm: the arm's samples ride the exit
// alternative's payload; the fall-through content lands in the continued
// alternative.

// CHECK-LABEL: uir.func @ContentInExitArm
uir.func @ContentInExitArm(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %v = hir.var : %t
  // CHECK: [[LIST:%.+]] = hir.opaque_list_create
  uir.expr pin -1 {
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      // CHECK: uir.if
      // CHECK-NEXT: [[C0:%.+]] = hir.choice 0(%arg0)
      // CHECK-NEXT: [[R0:%.+]] = hir.opaque_pack([[C0]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R0]]
      // CHECK-NEXT: uir.break
      uir.if %done {
        uir.expr pin 1 {
          hir.write_var %v, %i : %t
          uir.yield
        }
        uir.break
      }
      uir.expr pin 1 {
        hir.write_var %v, %x : %t
        uir.yield
      }
      %next = hir.add %i, %one : %t
      // CHECK: [[C1:%.+]] = hir.choice 1()
      // CHECK-NEXT: [[R1:%.+]] = hir.opaque_pack([[C1]])
      // CHECK-NEXT: hir.opaque_list_push [[LIST]], [[R1]]
      // CHECK-NEXT: uir.continue
      uir.continue %next : %t
    }
    uir.yield
  }
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (%arg1): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %arg1
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: hir.write_var %{{.*}}, %x
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: }
  %r = hir.read_var %v : %t
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
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (%arg1): {
  // CHECK-NEXT: [[S:%.+]] = hir.add %x, %arg1
  // CHECK-NEXT: uir.return [[S]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: hir.yield
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
      // The const increment sits in the loop body, unconditionally. The
      // continued link carries no samples.
      // CHECK: hir.read_var
      // CHECK-NEXT: hir.add
      // CHECK-NEXT: hir.write_var
      // CHECK: [[CC:%.+]] = hir.choice 1()
      // CHECK-NEXT: hir.opaque_pack([[CC]])
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
  // CHECK: hir.replicate (%arg0) in [[LIST]] {
  // CHECK-NEXT: hir.pick %arg0 [0 (): {
  // CHECK-NEXT: hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT: uir.if %c {
  // CHECK:   hir.read_var
  // CHECK:   hir.write_var
  // CHECK: }
  // CHECK: hir.yield
  // CHECK: }]
  // CHECK: hir.yield
  // CHECK: }
  %r = hir.read_var %d : %t
  uir.return %r : %t
}
