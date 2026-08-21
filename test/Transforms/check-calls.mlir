// RUN: silicon-opt --check-calls %s | FileCheck %s
// RUN: silicon-opt --check-calls %s | silicon-opt --verify-roundtrip

// Call checking copies the type computation a function declares into every
// place that consumes it: into each caller ahead of the `uir.call`, and into
// the function's own body. The copies are tied to the local types with
// `hir.unify` and `hir.coerce_type`.

//===----------------------------------------------------------------------===//
// A plain call. The callee's declared types are copied in front of the call and
// unified with the type operands the caller carries, and the unify results
// become the call's type operands.

// CHECK-LABEL: uir.func private @Callee
uir.func private @Callee(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  // The body receives its own copy of the signature, coerces the argument to
  // its declared type, and unifies the returned type with the declared result.
  // CHECK:      } {
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[CA:%.+]] = hir.coerce_type %a, [[T]]
  // CHECK-NEXT: [[TY:%.+]] = hir.type_of [[CA]]
  // CHECK-NEXT: [[U:%.+]] = hir.unify [[TY]], [[T]]
  // CHECK-NEXT: uir.return [[CA]] : [[U]]
  %t = hir.type_of %a
  uir.return %a : %t
}

// CHECK-LABEL: uir.func @Caller
uir.func @Caller() -> () {
  uir.signature () -> ()
} {
  // CHECK:      } {
  // CHECK-NEXT: [[V:%.+]] = builtin.unrealized_conversion_cast
  // CHECK-NEXT: [[I0:%.+]] = hir.inferrable
  // CHECK-NEXT: [[I1:%.+]] = hir.inferrable
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[UA:%.+]] = hir.unify [[T]], [[I0]]
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[T]], [[I1]]
  // CHECK-NEXT: uir.call @Callee([[V]]) : ([[UA]]) -> ([[UR]]) [0] -> [0]
  %v = builtin.unrealized_conversion_cast to !hir.any
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %r = uir.call @Callee(%v) : (%i0) -> (%i1) [0] -> [0]
  uir.return
}

//===----------------------------------------------------------------------===//
// A dependent type: the callee's argument and result types are computed from
// its own const argument. The caller recomputes them from the argument it
// passes, in its own context, rather than reading anything from the callee.

// CHECK-LABEL: uir.func private @DepUInt
uir.func private @DepUInt(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %2 = hir.uint_type %N
  uir.signature (%0, %1) -> (%2)
} {
  // The copied signature computes the dependent types from the raw arguments;
  // only the original body ops see the coerced ones.
  // CHECK:      } {
  // CHECK-NEXT: [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[U1:%.+]] = hir.uint_type %N
  // CHECK-NEXT: [[U2:%.+]] = hir.uint_type %N
  // CHECK-NEXT: hir.coerce_type %N, [[T]]
  // CHECK-NEXT: [[CX:%.+]] = hir.coerce_type %x, [[U1]]
  // CHECK-NEXT: [[TY:%.+]] = hir.type_of [[CX]]
  // CHECK-NEXT: [[UN:%.+]] = hir.unify [[TY]], [[U2]]
  // CHECK-NEXT: uir.return [[CX]] : [[UN]]
  %0 = hir.type_of %x
  uir.return %x : %0
}

// CHECK-LABEL: uir.func @DepCaller
uir.func @DepCaller(%n: -1, %v: 0) -> () {
  %0 = hir.int_type
  %1 = hir.uint_type %n
  uir.signature (%0, %1) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CN:%.+]] = hir.coerce_type %n
  // CHECK-NEXT: [[CV:%.+]] = hir.coerce_type %v
  // The callee's dependent types, recomputed here from the caller's own value.
  // CHECK:      [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[A1:%.+]] = hir.uint_type [[CN]]
  // CHECK-NEXT: [[A2:%.+]] = hir.uint_type [[CN]]
  // CHECK-NEXT: [[UN:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UX:%.+]] = hir.unify [[A1]], {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[A2]], {{%.+}}
  // CHECK-NEXT: uir.call @DepUInt([[CN]], [[CV]]) : ([[UN]], [[UX]]) -> ([[UR]]) [-1, 0] -> [0]
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r = uir.call @DepUInt(%n, %v) : (%i0, %i1) -> (%i2) [-1, 0] -> [0]
  uir.return
}

//===----------------------------------------------------------------------===//
// A call nested in a `uir.if` arm. The copied signature lands in the arm's
// block, right in front of the call.

// CHECK-LABEL: uir.func @CallInsideIfArm
uir.func @CallInsideIfArm(%c: 0, %v: 0) -> () {
  %b = hir.bool_type
  %i = hir.int_type
  uir.signature (%b, %i) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CV:%.+]] = hir.coerce_type %v
  // CHECK-NEXT: uir.if
  uir.if %c {
    // CHECK-NEXT: [[I0:%.+]] = hir.inferrable
    // CHECK-NEXT: [[I1:%.+]] = hir.inferrable
    // CHECK-NEXT: [[T:%.+]] = hir.int_type
    // CHECK-NEXT: [[UA:%.+]] = hir.unify [[T]], [[I0]]
    // CHECK-NEXT: [[UR:%.+]] = hir.unify [[T]], [[I1]]
    // CHECK-NEXT: uir.call @Callee([[CV]]) : ([[UA]]) -> ([[UR]]) [0] -> [0]
    %i0 = hir.inferrable
    %i1 = hir.inferrable
    %r = uir.call @Callee(%v) : (%i0) -> (%i1) [0] -> [0]
    uir.yield
  }
  uir.return
}

//===----------------------------------------------------------------------===//
// A call in a signature region. It is checked like any other call, and the
// copy the body receives brings the checked call along.

// CHECK-LABEL: uir.func @SigCaller
uir.func @SigCaller(%x: 0) -> (result: 0) {
  // CHECK:      [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[UA:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[R:%.+]] = uir.call @Callee({{%.+}}) : ([[UA]]) -> ([[UR]])
  // CHECK-NEXT: uir.signature ([[R]]) -> ([[R]])
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %u = hir.constant_unit
  %t = uir.call @Callee(%u) : (%i0) -> (%i1) [0] -> [0]
  uir.signature (%t) -> (%t)
} {
  // CHECK:      } {
  // CHECK:      [[BT:%.+]] = hir.int_type
  // CHECK-NEXT: hir.unify [[BT]], {{%.+}}
  // CHECK-NEXT: hir.unify [[BT]], {{%.+}}
  // CHECK-NEXT: [[BR:%.+]] = uir.call @Callee
  // CHECK-NEXT: [[CX:%.+]] = hir.coerce_type %x, [[BR]]
  // CHECK-NEXT: [[TY:%.+]] = hir.type_of [[CX]]
  // CHECK-NEXT: [[U:%.+]] = hir.unify [[TY]], [[BR]]
  // CHECK-NEXT: uir.return [[CX]] : [[U]]
  %0 = hir.type_of %x
  uir.return %x : %0
}

//===----------------------------------------------------------------------===//
// A `uir.split_func` callee. Its signature region is the whole interface, in
// the declared state of a precompiled or not-yet-compiled function as much as
// in the split state; nothing but that region is read.

// CHECK-LABEL: uir.split_func @DeclaredOnly
// CHECK-NOT:   hir.unify
uir.split_func @DeclaredOnly(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %a
  uir.signature (%0, %1) -> (%1)
} []

// CHECK-LABEL: uir.func @CallDeclaredOnly
uir.func @CallDeclaredOnly(%n: -1, %v: 0) -> () {
  %0 = hir.int_type
  %1 = hir.uint_type %n
  uir.signature (%0, %1) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CN:%.+]] = hir.coerce_type %n
  // CHECK-NEXT: [[CV:%.+]] = hir.coerce_type %v
  // CHECK:      [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[A1:%.+]] = hir.uint_type [[CN]]
  // CHECK-NEXT: [[UN:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UX:%.+]] = hir.unify [[A1]], {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[A1]], {{%.+}}
  // CHECK-NEXT: uir.call @DeclaredOnly([[CN]], [[CV]]) : ([[UN]], [[UX]]) -> ([[UR]]) [-1, 0] -> [0]
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r = uir.call @DeclaredOnly(%n, %v) : (%i0, %i1) -> (%i2) [-1, 0] -> [0]
  uir.return
}

// The per-phase functions of the split callee below, which call checking never
// looks at.
hir.func private @Split.0(%a) -> (ctx) {
  %int = hir.int_type
  %opaque = hir.opaque_type
  hir.signature (%int) -> (%opaque)
} {
  %opaque = hir.opaque_type
  %packed = hir.opaque_pack()
  hir.return %packed : %opaque
}
hir.func private @Split.1(%b, %ctx) -> (result) {
  %int = hir.int_type
  %opaque = hir.opaque_type
  hir.signature (%int, %opaque) -> (%int)
} {
  %int = hir.int_type
  hir.return %b : %int
}

// CHECK-LABEL: uir.split_func @Split
uir.split_func @Split(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} [
  @Split.0,
  @Split.1
]

// CHECK-LABEL: uir.func @CallSplit
uir.func @CallSplit(%n: -1, %v: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0, %0) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CN:%.+]] = hir.coerce_type %n
  // CHECK-NEXT: [[CV:%.+]] = hir.coerce_type %v
  // CHECK:      [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[UN:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UX:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: uir.call @Split([[CN]], [[CV]]) : ([[UN]], [[UX]]) -> ([[UR]]) [-1, 0] -> [0]
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r = uir.call @Split(%n, %v) : (%i0, %i1) -> (%i2) [-1, 0] -> [0]
  uir.return
}

//===----------------------------------------------------------------------===//
// A `uir.split_func` signature is already-checked output, so nothing in it is
// rewritten: phase sorting snapshots the declaration downstream of this pass,
// and a precompiled library ships an interface checked when the library was
// compiled. The call below is in checked form -- its type operands are the
// unify results a check produced -- and comes out of this run unchanged. A
// second check would stack another unify on each of them, since `hir.unify`
// folds only two identical values.

// CHECK-LABEL: uir.split_func @CheckedSigCall
uir.split_func @CheckedSigCall(%x: 0) -> (result: 0) {
  // CHECK:      [[T:%.+]] = hir.int_type
  // CHECK-NEXT: [[UA:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify [[T]], {{%.+}}
  // CHECK-NEXT: [[R:%.+]] = uir.call @Callee({{%.+}}) : ([[UA]]) -> ([[UR]])
  // CHECK-NEXT: uir.signature ([[R]]) -> ([[R]])
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %u = hir.constant_unit
  %t0 = hir.int_type
  %ua = hir.unify %t0, %i0
  %ur = hir.unify %t0, %i1
  %t = uir.call @Callee(%u) : (%ua) -> (%ur) [0] -> [0]
  uir.signature (%t) -> (%t)
} []

// The caller's copy of that signature carries the checked call along, and the
// call to the split function itself is checked against the copy.
// CHECK-LABEL: uir.func @CallCheckedSigCall
uir.func @CallCheckedSigCall(%v: 0) -> () {
  %0 = hir.int_type
  uir.signature (%0) -> ()
} {
  // CHECK:      } {
  // CHECK:      [[CV:%.+]] = hir.coerce_type %v
  // CHECK:      [[UA:%.+]] = hir.unify {{%.+}}, {{%.+}}
  // CHECK-NEXT: [[UR:%.+]] = hir.unify {{%.+}}, {{%.+}}
  // CHECK-NEXT: [[R:%.+]] = uir.call @Callee({{%.+}}) : ([[UA]]) -> ([[UR]])
  // CHECK-NEXT: [[UX:%.+]] = hir.unify [[R]], {{%.+}}
  // CHECK-NEXT: [[URR:%.+]] = hir.unify [[R]], {{%.+}}
  // CHECK-NEXT: uir.call @CheckedSigCall([[CV]]) : ([[UX]]) -> ([[URR]]) [0] -> [0]
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %r = uir.call @CheckedSigCall(%v) : (%i0) -> (%i1) [0] -> [0]
  uir.return
}

//===----------------------------------------------------------------------===//
// A split function's signature calling back into the module is a leaf of the
// checking order, since its own types are resolved before this run begins. The
// caller inlines it like any other declaration, self-call included; there is no
// order to violate and nothing is reported.

// CHECK-LABEL: uir.split_func @CallsBack
// CHECK-NEXT:  uir.call @CallsIntoDeclaration()
// CHECK-NEXT:  uir.signature () -> ()
uir.split_func @CallsBack() -> () {
  uir.call @CallsIntoDeclaration() : () -> () [] -> []
  uir.signature () -> ()
} []

// CHECK-LABEL: uir.func @CallsIntoDeclaration
uir.func @CallsIntoDeclaration() -> () {
  // CHECK:      uir.call @CallsIntoDeclaration()
  // CHECK-NEXT: uir.call @CallsBack()
  // CHECK-NEXT: uir.signature () -> ()
  uir.call @CallsBack() : () -> () [] -> []
  uir.signature () -> ()
} {
  // CHECK:      uir.call @CallsIntoDeclaration()
  // CHECK-NEXT: uir.call @CallsBack()
  // CHECK-NEXT: uir.return
  uir.return
}

//===----------------------------------------------------------------------===//
// A function with neither arguments nor results still gets the (empty) copies,
// and a return nested in control flow is unified like any other.

// CHECK-LABEL: uir.func @Empty
uir.func @Empty() -> () {
  // CHECK: uir.signature () -> ()
  uir.signature () -> ()
} {
  // CHECK: uir.return{{$}}
  uir.return
}

// CHECK-LABEL: uir.func @NestedReturn
uir.func @NestedReturn(%c: 0, %x: 0) -> (result: 0) {
  %b = hir.bool_type
  %i = hir.int_type
  uir.signature (%b, %i) -> (%i)
} {
  // CHECK:      } {
  // CHECK-NEXT: [[BT:%.+]] = hir.bool_type
  // CHECK-NEXT: [[IT:%.+]] = hir.int_type
  // CHECK:      [[CX:%.+]] = hir.coerce_type %x, [[IT]]
  uir.if %c {
    // CHECK:      [[TY:%.+]] = hir.type_of [[CX]]
    // CHECK-NEXT: [[U:%.+]] = hir.unify [[TY]], [[IT]]
    // CHECK-NEXT: uir.return [[CX]] : [[U]]
    %t = hir.type_of %x
    uir.return %x : %t
  }
  // CHECK:      [[T2:%.+]] = hir.type_of [[CX]]
  // CHECK-NEXT: [[U2:%.+]] = hir.unify [[T2]], [[IT]]
  // CHECK-NEXT: uir.return [[CX]] : [[U2]]
  %t = hir.type_of %x
  uir.return %x : %t
}
