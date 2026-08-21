// RUN: silicon-opt --check-calls --sort-phases --thread-phases --split-funcs %s | FileCheck %s
// RUN: silicon-opt --check-calls --sort-phases --thread-phases --split-funcs %s | silicon-opt --verify-roundtrip

// Call checking composes with the phase split: `@Id` declares a dependent
// `uint<N>` type, and after splitting the caller computes that type in its own
// phase -1 function, from the argument it passes, and threads it through its
// own context. Nothing reads the callee's context for types.

uir.func private @Id(%N: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %N
  %2 = hir.uint_type %N
  uir.signature (%0, %1) -> (%2)
} {
  %t = hir.uint_type %N
  uir.return %x : %t
}

uir.func @Main(%n: -1, %v: 0) -> (result: 0) {
  %0 = hir.int_type
  %1 = hir.uint_type %n
  %2 = hir.uint_type %n
  uir.signature (%0, %1) -> (%2)
} {
  %i0 = hir.inferrable
  %i1 = hir.inferrable
  %i2 = hir.inferrable
  %r = uir.call @Id(%n, %v) : (%i0, %i1) -> (%i2) [-1, 0] -> [0]
  %t = hir.uint_type %n
  uir.return %r : %t
}

// The callee's own phase -1 computes its declared types and packs them for its
// phase 0, which types its argument and result from that context alone.
// CHECK-LABEL: hir.func private @"Id.-1.tail"(%N, %ctx)
// CHECK:         hir.uint_type %N
// CHECK:         hir.opaque_pack

// CHECK-LABEL: hir.func private @Id.0(%x, %ctx)
// CHECK:         } {
// CHECK-NEXT:    [[U:%.+]]:4 = hir.opaque_unpack %ctx
// CHECK-NEXT:    [[CX:%.+]] = hir.coerce_type %x, [[U]]#2
// CHECK-NEXT:    hir.return [[CX]] : [[U]]#3

// The caller recomputes the callee's dependent type from its own copy of the
// callee's signature, in its own phase -1, and hands it to the phase -1 call.
// CHECK-LABEL: hir.func private @"Main.-1.tail"(%n, %ctx)
// CHECK:         } {
// CHECK:         [[CN:%.+]] = hir.coerce_type %n
// CHECK-NEXT:    [[AT:%.+]] = hir.uint_type [[CN]]
// CHECK-NEXT:    [[RT:%.+]] = hir.uint_type [[CN]]
// CHECK-NEXT:    [[UA:%.+]] = hir.unify [[AT]], {{%.+}}
// CHECK-NEXT:    [[UR:%.+]] = hir.unify [[RT]], {{%.+}}
// CHECK-NEXT:    [[CTX:%.+]] = hir.call @"Id.-1"([[CN]]) : ({{%.+}}) -> ({{%.+}})
// The caller's own phase 0 receives the callee's argument and result type, and
// the callee's context, through the caller's own pack.
// CHECK:         hir.opaque_pack({{.*}}[[UA]], [[UR]], [[CTX]]{{.*}})

// The phase 0 call types its argument and result from the caller's own context
// slots; the callee's context rides in a slot of its own.
// CHECK-LABEL: hir.func private @Main.0(%v, %ctx)
// CHECK:         [[U:%.+]]:7 = hir.opaque_unpack %ctx
// CHECK:         hir.call @Id.0({{%.+}}, [[U]]#5) : ([[U]]#3, {{%.+}}) -> ([[U]]#4)

// CHECK-LABEL: uir.split_func @Main
// CHECK:       } [
// CHECK-NEXT:    @"Main.-1",
// CHECK-NEXT:    @Main.0
// CHECK-NEXT:  ]
