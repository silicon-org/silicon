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
// Basic early exit under an else-less if: the if gains an exit-value slot
// typed by the shared result-type inferrable (site types unify into it) and
// a payload-free tag; the else arm is materialized for the tag-1 path with
// an unreachable_value slot. The dispatch pick consumes the slot directly,
// the continuation moves into its tag-1 alternative, and the single trailing
// return is typed by the shared inferrable.

// CHECK-LABEL: uir.func @ExitUnderIf
uir.func @ExitUnderIf(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]]:2 = uir.if %c : [[ST]], [[CT]] {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield %x, [[C0]] : [[T]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[UV]], [[C1]] : [[ST]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[IF]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[IF]]#0
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   hir.yield [[Y]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// Early exit under an if with a result: the original result slot keeps its
// own type operand, the exit arm fills it with unreachable_value, and the
// continuation keeps using the if's value result.

// CHECK-LABEL: uir.func @ExitUnderIfWithResult
uir.func @ExitUnderIfWithResult(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]]:3 = uir.if %c : [[T]], [[ST]], [[CT]] {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[UV]], %x, [[C0]] : [[T]], [[T]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   [[UVE:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[Y]], [[UVE]], [[C1]] : [[T]], [[ST]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[IF]]#2 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[IF]]#1
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Z:%.+]] = hir.add [[IF]]#0, %x : [[T]]
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   hir.yield [[Z]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
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
// All paths exit: both arms provide the exit slot under tag 0, the tag-1
// alternative holds the dead unreachable tail, and the body still ends in a
// single return on the picked value.

// CHECK-LABEL: uir.func @AllArmsExit
uir.func @AllArmsExit(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]]:2 = uir.if %c : [[ST]], [[CT:%.+]] {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   [[C0A:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield %x, [[C0A]] : [[T]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   [[C0B:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[Y]], [[C0B]] : [[T]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[IF]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[IF]]#0
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
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
// the first dispatch's tag-1 alternative and threads its result up, so no
// continuation is duplicated, and both ifs share the one result-type
// inferrable.

// CHECK-LABEL: uir.func @TwoSequentialExits
uir.func @TwoSequentialExits(%c: 0, %d: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK: [[IF1:%.+]]:2 = uir.if %c : [[ST]], {{%.+}} {
  // CHECK: [[P1:%.+]] = hir.pick [[IF1]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[IF1]]#0
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[IF2:%.+]]:2 = uir.if %d : [[ST]], {{%.+}} {
  // CHECK:        [[P2:%.+]] = hir.pick [[IF2]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:     hir.yield [[IF2]]#0
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[Z:%.+]] = hir.add %x, %x
  // CHECK-NEXT:     hir.unify [[ST]], {{%.+}}
  // CHECK-NEXT:     hir.yield [[Z]]
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   hir.yield [[P2]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P1]] : [[ST]]
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
// the outer arm and yields the outer arm's slots with a fresh tag; the
// shared inferrable types every level's exit slot, so no unify chains are
// needed at the re-wrap sites.

// CHECK-LABEL: uir.func @NestedIfExit
uir.func @NestedIfExit(%a: 0, %b: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[OIF:%.+]]:2 = uir.if %a : [[ST]], [[CT:%.+]] {
  // CHECK-NEXT:   [[IIF:%.+]]:2 = uir.if %b : [[ST]], [[CT]] {
  // CHECK-NEXT:     hir.unify [[ST]], [[T]]
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield %x, [[C0]] : [[T]], [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK-NEXT:     [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     uir.yield [[UV]], [[C1]] : [[ST]], [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   [[IP:%.+]]:2 = hir.pick [[IIF]]#1 : !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:     [[RC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     hir.yield [[IIF]]#0, [[RC0]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     [[RUV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     [[RC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:     hir.yield [[RUV]], [[RC1]]
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.yield [[IP]]#0, [[IP]]#1 : [[ST]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[EUV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[EC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[EUV]], [[EC1]] : [[ST]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[OIF]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[OIF]]#0
  // CHECK: uir.return [[P]] : [[ST]]
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
// Return under a loop becomes a break-carried tag: the exit path breaks out
// with tag 0 and the exit value in the loop's trailing slots (its original
// result slot filled with unreachable_value), the existing break gains the
// unreachable exit slot and tag 1, and the post-loop pick dispatches between
// exit and completion.

// CHECK-LABEL: uir.func @ReturnUnderLoop
uir.func @ReturnUnderLoop(%c: 0, %d: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[L:%.+]]:3 = uir.loop : [[T]], [[ST]], [[CT:%.+]] {
  // CHECK-NEXT:   [[IF:%.+]]:2 = uir.if %c : [[ST]], [[CT]] {
  // CHECK:        hir.pick [[IF]]#1 [0 (): {
  // CHECK-NEXT:     [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:     [[BC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.break [[UV]], [[IF]]#0, [[BC0]] : [[T]], [[ST]], [[CT]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     uir.if %d {
  // CHECK-NEXT:       [[Y:%.+]] = hir.add %x, %x : [[T]]
  // CHECK-NEXT:       [[BUV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:       [[BC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.break [[Y]], [[BUV]], [[BC1]] : [[T]], [[ST]], [[CT]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[L]]#2 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[L]]#1
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   hir.yield [[L]]#0
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
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
// A breakless loop whose only exit is the return: the rebuilt loop's results
// are the exit slot and the tag, and the dead tail after it stays in the
// tag-1 alternative behind an unreachable.

// CHECK-LABEL: uir.func @ReturnUnderBreaklessLoop
uir.func @ReturnUnderBreaklessLoop(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[L:%.+]]:2 = uir.loop : [[ST]], [[CT:%.+]] {
  // CHECK:        hir.pick {{%.+}}#1 [0 (): {
  // CHECK-NEXT:     [[BC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.break {{%.+}}#0, [[BC0]] : [[ST]], [[CT]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[L]]#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[L]]#0
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
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
// A return terminating a pinned expr body: the expr is rebuilt with the exit
// slot and tag as extra results, crossing the pin like any other result, and
// its original result slot is filled with unreachable_value.

// CHECK-LABEL: uir.func @ExitInExpr
uir.func @ExitInExpr(%c: 0, %x: 0) -> (result: 0) {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> (%1)
} {
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[E:%.+]]:3 = uir.expr pin -1 : [[T]], [[ST]], [[CT:%.+]] {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   [[UV:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[UV]], %x, [[C0]] : [[T]], [[T]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]] = hir.pick [[E]]#2 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[E]]#1
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   hir.unify [[ST]], [[T]]
  // CHECK-NEXT:   hir.yield [[E]]#0
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
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
  // CHECK: [[ST:%.+]] = hir.inferrable
  // CHECK: [[P:%.+]] = hir.pick {{%.+}}#1 : !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield {{%.+}}#0
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[Y:%.+]] = hir.add %x, %x
  // CHECK-NEXT:   hir.unify [[ST]], {{%.+}}
  // CHECK-NEXT:   hir.yield [[Y]]
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.return [[P]] : [[ST]]
  %t = hir.int_type
  uir.if %c {
    uir.return %x : %t
  }
  %y = hir.add %x, %x : %t
  uir.return %y : %t
}

//===----------------------------------------------------------------------===//
// A void function: there are no exit slots and no result-type inferrables,
// only the tag, and the effects still move into the dispatch alternatives.

// CHECK-LABEL: uir.func @VoidExit
uir.func @VoidExit(%c: 0, %x: 0) -> () {
  %0 = hir.bool_type
  %1 = hir.int_type
  uir.signature (%0, %1) -> ()
} {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NOT: hir.inferrable
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

//===----------------------------------------------------------------------===//
// The signature region follows the same rules as the body, with
// uir.signature as its return-like exit: an early signature under an if is
// rebuilt into exit slots (one per argument and result type value) plus a
// payload-free tag, and the dispatch pick threads the slots into the single
// trailing uir.signature. A uir.signature site carries no type operands of
// its own, so the slots are typed by the shared inferrables directly and no
// unify is emitted.

// CHECK-LABEL: uir.func @SignatureExitUnderIf
uir.func @SignatureExitUnderIf(%c: 0, %x: 0) -> (result: 0) {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[S0:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S1:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S2:%.+]] = hir.inferrable
  // CHECK-NEXT: [[B:%.+]] = hir.bool_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[IF:%.+]]:4 = uir.if %c : [[S0]], [[S1]], [[S2]], [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[B]], [[T]], [[T]], [[C0]] : [[S0]], [[S1]], [[S2]], [[CT]]
  // CHECK-NEXT: } else {
  // CHECK-NEXT:   [[U0:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[U1:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[U2:%.+]] = uir.unreachable_value
  // CHECK-NEXT:   [[C1:%.+]] = hir.choice 1()
  // CHECK-NEXT:   uir.yield [[U0]], [[U1]], [[U2]], [[C1]] : [[S0]], [[S1]], [[S2]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:3 = hir.pick [[IF]]#3 : !hir.any, !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[IF]]#0, [[IF]]#1, [[IF]]#2 : !hir.any, !hir.any, !hir.any
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[W:%.+]] = hir.int_type
  // CHECK-NEXT:   hir.yield [[B]], [[W]], [[W]] : !hir.any, !hir.any, !hir.any
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.signature ([[P]]#0, [[P]]#1) -> ([[P]]#2)
  %b = hir.bool_type
  %int = hir.int_type
  uir.if %c {
    uir.signature (%b, %int) -> (%int)
  }
  %w = hir.int_type
  uir.signature (%b, %w) -> (%w)
} {
  %t = hir.int_type
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// A signature exit under a loop becomes a break-carried tag, exactly like a
// return under a loop in the body: exit paths break out with the slots and
// tag 0, existing breaks gain unreachable slots and tag 1, and the post-loop
// pick dispatches into the trailing uir.signature.

// CHECK-LABEL: uir.func @SignatureExitUnderLoop
uir.func @SignatureExitUnderLoop(%c: 0, %d: 0, %x: 0) -> (result: 0) {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[S0:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S1:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S2:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S3:%.+]] = hir.inferrable
  // CHECK-NEXT: [[B:%.+]] = hir.bool_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[L:%.+]]:5 = uir.loop : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]] {
  // CHECK-NEXT:   [[IF:%.+]]:5 = uir.if %c : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]] {
  // CHECK-NEXT:     [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.yield [[B]], [[B]], [[T]], [[T]], [[C0]] : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]]
  // CHECK-NEXT:   } else {
  // CHECK:          uir.yield {{.*}} : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]]
  // CHECK-NEXT:   }
  // CHECK-NEXT:   hir.pick [[IF]]#4 [0 (): {
  // CHECK-NEXT:     [[BC0:%.+]] = hir.choice 0()
  // CHECK-NEXT:     uir.break [[IF]]#0, [[IF]]#1, [[IF]]#2, [[IF]]#3, [[BC0]] : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]]
  // CHECK-NEXT:   } 1 (): {
  // CHECK-NEXT:     uir.if %d {
  // CHECK:            [[BC1:%.+]] = hir.choice 1()
  // CHECK-NEXT:       uir.break {{.*}}, [[BC1]] : [[S0]], [[S1]], [[S2]], [[S3]], [[CT]]
  // CHECK-NEXT:     }
  // CHECK-NEXT:     uir.continue
  // CHECK-NEXT:   }]
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:4 = hir.pick [[L]]#4 : !hir.any, !hir.any, !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[L]]#0, [[L]]#1, [[L]]#2, [[L]]#3 : !hir.any, !hir.any, !hir.any, !hir.any
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   [[W:%.+]] = hir.int_type
  // CHECK-NEXT:   hir.yield [[B]], [[B]], [[W]], [[W]] : !hir.any, !hir.any, !hir.any, !hir.any
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.signature ([[P]]#0, [[P]]#1, [[P]]#2) -> ([[P]]#3)
  %b = hir.bool_type
  %int = hir.int_type
  uir.loop {
    uir.if %c {
      uir.signature (%b, %b, %int) -> (%int)
    }
    uir.if %d {
      uir.break
    }
    uir.continue
  }
  %w = hir.int_type
  uir.signature (%b, %b, %w) -> (%w)
} {
  %t = hir.int_type
  uir.return %x : %t
}

//===----------------------------------------------------------------------===//
// A signature exit terminating a pinned expr in the signature region: the
// expr is rebuilt with the slots and tag as extra results, and the dead tail
// after the exhaustive exit stays in the tag-1 alternative behind an
// unreachable.

// CHECK-LABEL: uir.func @SignatureExitInExpr
uir.func @SignatureExitInExpr(%c: 0, %x: 0) -> (result: 0) {
  // CHECK: [[CT:%.+]] = hir.choice_type
  // CHECK-NEXT: [[S0:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S1:%.+]] = hir.inferrable
  // CHECK-NEXT: [[S2:%.+]] = hir.inferrable
  // CHECK-NEXT: [[B:%.+]] = hir.bool_type
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[E:%.+]]:4 = uir.expr pin -1 : [[S0]], [[S1]], [[S2]], [[CT]] {
  // CHECK-NEXT:   [[C0:%.+]] = hir.choice 0()
  // CHECK-NEXT:   uir.yield [[B]], [[T]], [[T]], [[C0]] : [[S0]], [[S1]], [[S2]], [[CT]]
  // CHECK-NEXT: }
  // CHECK-NEXT: [[P:%.+]]:3 = hir.pick [[E]]#3 : !hir.any, !hir.any, !hir.any [0 (): {
  // CHECK-NEXT:   hir.yield [[E]]#0, [[E]]#1, [[E]]#2 : !hir.any, !hir.any, !hir.any
  // CHECK-NEXT: } 1 (): {
  // CHECK-NEXT:   uir.unreachable
  // CHECK-NEXT: }]
  // CHECK-NEXT: uir.signature ([[P]]#0, [[P]]#1) -> ([[P]]#2)
  %b = hir.bool_type
  %int = hir.int_type
  uir.expr pin -1 {
    uir.signature (%b, %int) -> (%int)
  }
  uir.unreachable
} {
  %t = hir.int_type
  uir.return %x : %t
}
