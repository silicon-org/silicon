// RUN: silicon-opt --check-calls --split-input-file --verify-diagnostics %s

// A callee that does not resolve, that is not a function, or whose phases or
// arity disagree with the call is rejected by the `uir.call` verifier before
// any pass runs; see test/UIR/errors.mlir.

//===----------------------------------------------------------------------===//
// A signature that transitively calls itself has no order in which the callee's
// declared types are known before the caller inlines them.

// expected-error @below {{signature of `foo` cannot call itself}}
uir.func @foo() -> () {
  // expected-remark @below {{calling `foo` itself here}}
  uir.call @foo() : () -> () [] -> []
  uir.signature () -> ()
} {
  uir.return
}

// -----

// expected-error @below {{signature of `foo` cannot call itself}}
uir.func @foo() -> () {
  // expected-remark @below {{called through `bar`}}
  uir.call @bar() : () -> () [] -> []
  uir.signature () -> ()
} {
  uir.return
}
uir.func @bar() -> () {
  // expected-remark @below {{called through `gux`}}
  uir.call @gux() : () -> () [] -> []
  uir.signature () -> ()
} {
  uir.return
}
uir.func @gux() -> () {
  // expected-remark @below {{calling `foo` itself here}}
  uir.call @foo() : () -> () [] -> []
  uir.signature () -> ()
} {
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// The signature of a `uir.split_func` takes part in the same traversal, so a
// cycle running through a declaration is reported like any other.

// expected-error @below {{signature of `foo` cannot call itself}}
uir.split_func @foo() -> () {
  // expected-remark @below {{called through `bar`}}
  uir.call @bar() : () -> () [] -> []
  uir.signature () -> ()
} []
uir.func @bar() -> () {
  // expected-remark @below {{calling `foo` itself here}}
  uir.call @foo() : () -> () [] -> []
  uir.signature () -> ()
} {
  uir.return
}

// -----

//===----------------------------------------------------------------------===//
// A signature spread across multiple blocks has no single computation to copy
// into a use. Call checking runs before control flow is flattened.

// expected-error @below {{signature region must be a single block exiting through one 'uir.signature'}}
uir.func @MultiBlockSig(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
^dead:
  uir.unreachable
} {
  %t = hir.type_of %a
  uir.return %a : %t
}

// -----

//===----------------------------------------------------------------------===//
// A signature exiting through more than one `uir.signature` declares its types
// on several paths; exit normalization brings those together into one exit
// before call checking runs.

// expected-error @below {{signature region must be a single block exiting through one 'uir.signature'}}
uir.func @MultiExitSig(%c: 0, %x: 0) -> (result: 0) {
  uir.if %c {
    uir.signature (%c, %x) -> (%x)
  }
  uir.signature (%c, %x) -> (%c)
} {
  uir.return %x : %x
}

// -----

//===----------------------------------------------------------------------===//
// A signature that never exits declares nothing at all.

// expected-error @below {{signature region must be a single block exiting through one 'uir.signature'}}
uir.func @NoSigExit(%x: 0) -> (result: 0) {
  uir.unreachable
} {
  uir.return %x : %x
}

// -----

//===----------------------------------------------------------------------===//
// The callee's signature is read the same way, so an unusable one is reported
// where it is declared, at the call site's expense.

uir.func @Caller(%x: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %r = uir.call @UnusableCallee(%x) : (%i0) -> (%i1) [0] -> [0]
  uir.return
}

// expected-error @below {{signature region must be a single block exiting through one 'uir.signature'}}
uir.split_func @UnusableCallee(%a: 0) -> (result: 0) {
  uir.unreachable
} []

// -----

//===----------------------------------------------------------------------===//
// Call checking rewrites the regions of declared functions. A call anywhere
// else is never walked, and would silently keep the unconstrained types the
// caller wrote.

uir.func private @Callee(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.type_of %a
  uir.return %a : %t
}

func.func @Stray(%v: !hir.any) {
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  // expected-error @below {{compiler bug: call is not inside a 'uir.func' or 'uir.split_func', so nothing checks it against the function it calls}}
  // expected-note @below {{on line}}
  %r = uir.call @Callee(%v) : (%i0) -> (%i1) [0] -> [0]
  return
}

// -----

//===----------------------------------------------------------------------===//
// Checking runs before sorting, and produces the types sorting consumes.
// Sorted input has nothing left to check.

uir.func @Sorted(%a: -1) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  // expected-error @below {{compiler bug: phases have already been sorted; calls are checked before sorting, whose input the checked types are part of}}
  // expected-note @below {{on line}}
  %p:2 = uir.phase -1 {
    %prod = hir.mul %a, %a : %t
    uir.phase_return yield %prod, %t
  }
  uir.phase 0 {
    %sum = hir.add %p#0, %p#0 : %p#1
    uir.phase_return %sum : %p#1
  }
  uir.unreachable
}
