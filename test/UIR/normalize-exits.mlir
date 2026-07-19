// RUN: silicon-opt %s --normalize-exits | FileCheck %s

//===----------------------------------------------------------------------===//
// No early exit: a function whose only return is the body terminator is
// untouched, even with control flow present.

// CHECK-LABEL: uir.func @NoEarlyExit
uir.func @NoEarlyExit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK-NOT: hir.choice
  // CHECK-NOT: hir.pick
  %t = hir.int_type
  %v = uir.if %c : %t {
    uir.yield %x : %t
  } else {
    %y = hir.add %x, %x : %t
    uir.yield %y : %t
  }
  uir.return %v : %t
  // CHECK: uir.return
  // CHECK-NOT: hir.choice
  // CHECK-NOT: hir.pick
}

//===----------------------------------------------------------------------===//
// Basic early exit under an else-less if: the if gains a choice result (the
// else arm is materialized for the tag-1 path), the continuation moves into
// the dispatch pick's tag-1 alternative, and the threaded value and type
// operand feed the single trailing return.

// CHECK-LABEL: uir.func @ExitUnderIf
uir.func @ExitUnderIf(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:   uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[IF]] : !hir.any, !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V]], [[TV]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   hir.yield [[Y]], [[T]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Early exit under an if with a result: the exit arm fills the result slot
// with uir.unreachable_value, the yielding arm gains the tag-1 choice, and
// the continuation keeps using the if's value result.

// CHECK-LABEL: uir.func @ExitUnderIfWithResult
uir.func @ExitUnderIfWithResult(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]]:2 = uir.if %c : [[T]], [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:   [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[UV]], [[C0]] : [[T]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[Y]], [[C1]] : [[T]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[IF]]#1 : !hir.any, !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V]], [[TV]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Z:%.+]] = hir.add [[IF]]#0, %x : [[T]]
  // CHECK-NEXT:   hir.yield [[Z]], [[T]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  %v = uir.if %c : %t {
    uir.return %x : %t
  } else {
    %y = hir.add %x, %x : %t
    uir.yield %y : %t
  }
  %z = hir.add %v, %x : %t
  uir.return %z : %t
}

//===----------------------------------------------------------------------===//
// All paths exit: both arms build tag 0 with their own values, the tag-1
// alternative holds the dead unreachable tail, and the body still ends in a
// single return on the picked value.

// CHECK-LABEL: uir.func @AllArmsExit
uir.func @AllArmsExit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK-NEXT:   [[C0A:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:   uir.yield [[C0A]] : [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   [[C0B:%.+]] = hir.choice 0([[Y]], [[T]])
  // CHECK-NEXT:   uir.yield [[C0B]] : [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[IF]] : !hir.any, !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V]], [[TV]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  } else {
    %y = hir.add %x, %x : %t
    uir.return %y : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Two sequential exit-carrying ifs right-nest: the second dispatch lives in
// the first dispatch's tag-1 alternative and threads its results up, so no
// continuation is duplicated.

// CHECK-LABEL: uir.func @TwoSequentialExits
uir.func @TwoSequentialExits(%c: 0, %d: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[P1:%.+]]:2 = hir.pick {{%.+}} : !hir.any, !hir.any [0 ([[V1:%.+]], [[TV1:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V1]], [[TV1]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[IF2:%.+]] = uir.if %d
  // CHECK:        [[P2:%.+]]:2 = hir.pick [[IF2]] : !hir.any, !hir.any [0 ([[V2:%.+]], [[TV2:%.+]]): {
  // CHECK-NEXT:     hir.yield [[V2]], [[TV2]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[Z:%.+]] = hir.add %x, %x
  // CHECK-NEXT:     hir.yield [[Z]], {{%.+}}
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   hir.yield [[P2]]#0, [[P2]]#1
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P1]]#0 : [[P1]]#1
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  uir.if %d {
    %y = hir.add %x, %x : %t
    uir.return %y : %t
  }
  %z = hir.add %x, %x : %t
  uir.return %z : %t
}

//===----------------------------------------------------------------------===//
// A nested exit re-wraps level by level: the inner if's dispatch runs inside
// the outer arm and yields the outer arm's own tag-0/tag-1 encoding, forming
// a linear chain of binary picks.

// CHECK-LABEL: uir.func @NestedIfExit
uir.func @NestedIfExit(%a: 0, %b: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[OIF:%.+]] = uir.if %a : [[CT]] {
  // CHECK-NEXT:   [[IIF:%.+]] = uir.if %b : [[CT]] {
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:     uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[IP:%.+]] = hir.pick [[IIF]] : !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:     [[RC0:%.+]] = hir.choice 0([[V]], [[TV]])
  // CHECK-NEXT:     hir.yield [[RC0]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[RC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.yield [[RC1]]
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.yield [[IP]] : [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[EC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[EC1]] : [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[OIF]] : !hir.any, !hir.any [0 (
  // CHECK: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  uir.if %a {
    uir.if %b {
      uir.return %x : %t
    }
    uir.yield
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Return under a loop becomes a break-carried choice: the exit path breaks
// out with tag 0 in the loop's trailing result slot (its value slot filled
// with uir.unreachable_value), the existing break gains the payload-free
// tag 1, and the post-loop pick dispatches between exit and completion.

// CHECK-LABEL: uir.func @ReturnUnderLoop
uir.func @ReturnUnderLoop(%c: 0, %d: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[L:%.+]]:2 = uir.loop : [[T]], [[CT]] {
  // CHECK-NEXT:   [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:     uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hir.pick [[IF]] [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:     [[BC0:%.+]] = hir.choice 0([[V]], [[TV]])
  // CHECK-NEXT:     [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     uir.break [[UV]], [[BC0]] : [[T]], [[CT]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     uir.if %d {
  // CHECK-NEXT:       [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:       [[BC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.break [[Y]], [[BC1]] : [[T]], [[CT]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[L]]#1 : !hir.any, !hir.any [0 ([[PV:%.+]], [[PT:%.+]]): {
  // CHECK-NEXT:   hir.yield [[PV]], [[PT]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.yield [[L]]#0, [[T]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  %l = uir.loop : %t {
    uir.if %c {
      uir.return %x : %t
    }
    uir.if %d {
      %y = hir.add %x, %x : %t
      uir.break %y : %t
    }
    uir.continue
  }
  uir.return %l : %t
}

//===----------------------------------------------------------------------===//
// A breakless loop whose only exit is the return: the rebuilt loop's sole
// result is the choice, and the dead tail after it stays in the tag-1
// alternative behind an unreachable.

// CHECK-LABEL: uir.func @ReturnUnderBreaklessLoop
uir.func @ReturnUnderBreaklessLoop(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[L:%.+]] = uir.loop : [[CT]] {
  // CHECK:        hir.pick {{%.+}} [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:     [[BC0:%.+]] = hir.choice 0([[V]], [[TV]])
  // CHECK-NEXT:     uir.break [[BC0]] : [[CT]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[L]] : !hir.any, !hir.any [0 ([[PV:%.+]], [[PT:%.+]]): {
  // CHECK-NEXT:   hir.yield [[PV]], [[PT]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  uir.loop {
    uir.if %c {
      uir.return %x : %t
    }
    uir.continue
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// A return terminating a pinned expr body: the expr is rebuilt with the
// choice as an extra result, crossing the pin like any other result, and its
// original result slot is filled with uir.unreachable_value.

// CHECK-LABEL: uir.func @ExitInExpr
uir.func @ExitInExpr(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[E:%.+]]:2 = uir.expr pin -1 : [[T]], [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0(%x, [[T]])
  // CHECK-NEXT:   [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   uir.yield [[UV]], [[C0]] : [[T]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:2 = hir.pick [[E]]#1 : !hir.any, !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V]], [[TV]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.yield [[E]]#0, [[T]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  %e = uir.expr pin -1 : %t {
    uir.return %x : %t
  }
  uir.return %e : %t
}

//===----------------------------------------------------------------------===//
// A divergent signature region (terminated by uir.unreachable) does not
// affect body normalization: the result arity comes from the function's
// declared result phases, not from the signature terminator.

// CHECK-LABEL: uir.func @DivergentSignature
uir.func @DivergentSignature(%c: 0, %x: 0) -> (result: 0) {
  uir.unreachable
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[P:%.+]]:2 = hir.pick {{%.+}} : !hir.any, !hir.any [0 ([[V:%.+]], [[TV:%.+]]): {
  // CHECK-NEXT:   hir.yield [[V]], [[TV]]
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x
  // CHECK-NEXT:   hir.yield [[Y]], {{%.+}}
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]]#0 : [[P]]#1
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// A void function: the exit payloads and the threading are empty, and the
// effects still move into the dispatch alternatives.

// CHECK-LABEL: uir.func @VoidExit
uir.func @VoidExit(%c: 0, %x: 0) -> () {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> ()
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK: [[IF:%.+]] = uir.if %c : [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[C0]] : [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[C1]] : [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: hir.pick [[IF]] [0 (): {
  // CHECK-NEXT:   hir.yield
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.write_var {{%.+}}, %x
  // CHECK-NEXT:   hir.yield
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return
  %t = hir.int_type
  %va = hir.var : %t
  uir.if %c {
    uir.return
  }
  hir.write_var %va, %x : %t
  uir.return
}
