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
