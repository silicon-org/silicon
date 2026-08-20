// RUN: silicon-opt --verify-roundtrip %s

// # The Split-Form Contract
//
// Hand-written HIR for the canonical function-splitting examples. This file is
// the expected *output* of the phase-blind function splitter: given a threaded
// `uir.func` (the output contract of threading normalization, see
// thread-phases.mlir), splitting emits one per-phase `hir.func` per `uir.phase`
// op, wires the phase-op result flow into opaque context pack/unpack, rewrites
// `uir.phase_call` into `hir.call` to the callee's per-phase entries,
// reconstructs each per-phase signature, and fills the phase-to-func link list
// of the `uir.split_func` (with `hir.eager_func` heading any pre-span chain).
// This file doubles as the lit-test contract for the splitter's output shape.
//
// The contract, per split function:
//
// - One `hir.func` per phase present in *either* region, in phase order. Its
//   body is the signature region's phase-P content followed by the body
//   region's; splitting merges, it never relocates or clones computation.
//   The first phase overall takes no incoming context; every later func takes
//   an extra trailing opaque context argument; every func but the last
//   produces an extra trailing opaque context result. Each func also takes its
//   phase's declared arguments. Function results ride the func of the phase
//   they belong to; the last func returns no context.
// - Values crossing a phase boundary are packed into the outgoing context by
//   the producing func (`hir.opaque_pack`) and unpacked by the consuming func
//   (`hir.opaque_unpack`). The boundary carries the union of what both regions'
//   phase ops yield, deduplicated structurally, so a type the signature and the
//   body both compute occupies one slot.
// - A per-phase signature region contains nothing but `hir.opaque_type` and one
//   `hir.opaque_unpack` of the incoming context. A type ascribing a phase-P
//   argument or result is phase P-1 content, so it always arrives through the
//   context and never from the func's own arguments; that is what makes the
//   per-phase MIR signature constructible after monomorphization and folding
//   without evaluating anything. The consequence is visible throughout this
//   file: even a function whose types are plain constants grows a phase P-1
//   func that computes them, below its declared span.
// - `uir.split_func` maps each span phase number to its entry func and holds
//   the original, unsorted signature (in terms of the declared block
//   arguments) as the stable late-binding interface. It is the callee symbol a
//   phase-agnostic `uir.call` targets. The signature region is preserved
//   verbatim from before sorting (sorting creates the split_func and moves the
//   implementation func aside), never reconstructed from the sorted form
//   (settled 2026-08-01).
// - A pre-span phase (content earlier than the declared span, argument-free by
//   construction) is folded into an `hir.eager_func` whose head link is the
//   pre-span func and whose tail is the earliest span entry; `uir.split_func`
//   maps the span entry to the eager_func symbol, whose external signature is
//   the earliest span entry's declared arguments and results — i.e. the tail
//   link's arguments minus the trailing pre-span context argument.
//
// Per-phase funcs are named `@<f>.<phase>` (quoted for negative phases). The
// names are cosmetic and may be uniquified freely; link lists and the
// split_func's phase list are the authoritative phase-to-func mapping, and
// nothing resolves by name convention (settled 2026-08-01).

//===----------------------------------------------------------------------===//
// Case 1: three-phase split of a dependent-type identity function.
//
// Threaded input `@Id(%N: -1, %x: 0) -> (result: 0)`, `x: uint[N]`:
//   signature:
//     %s2 = uir.phase -2 { %int = hir.int_type;    uir.phase_return yield %int }
//     %s1 = uir.phase -1 { %ty = hir.uint_type %N; uir.phase_return yield %ty }
//           uir.signature (%s2, %s1) -> (%s1)
//   body:
//     %p = uir.phase -1 { %ty = hir.uint_type %N; uir.phase_return yield %ty }
//          uir.phase  0 { %r = hir.add %x, %x : %p; uir.phase_return %r : %p }
//
// %N is declared at phase -1, so its own (plain int) type is phase -2 content
// and gets a func of its own, below the declared span. The dependent type
// uint[N] is phase -1 content and rides the opaque context into phase 0, where
// it is unpacked in the signature region (as the type of %x and the result) and
// again in the body (as the add type).

hir.func @"Id.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %ctx = hir.opaque_pack(%int)
  hir.return %ctx : %o
}

// The phase -1 func merges the signature's phase -1 content with the body's.
// Both compute uint[N]; the two coincide in one context slot, so the splitter
// packs one of them and leaves the other dead for the canonicalizer (elided
// here). %N's type is unpacked from the incoming context, never recomputed.
hir.func @"Id.-1.tail"(%N, %ctx_in) -> (ctx) {
  %o = hir.opaque_type
  %nty = hir.opaque_unpack %ctx_in : !hir.any
  hir.signature (%nty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %ty = hir.uint_type %N
  %ctx = hir.opaque_pack(%ty)
  hir.return %ctx : %o
}

hir.eager_func @"Id.-1"(N) -> (ctx) [
  @"Id.-2",
  @"Id.-1.tail"
]

hir.func @"Id.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  // The dependent type of %x arrives by unpacking the incoming context. A split
  // caller specializing phase 0 never reads this context for types: call
  // checking inlines the callee's signature computation into the caller, which
  // recomputes the type in its own earlier phase (settled 2026-08-01;
  // exercised in case 5).
  %ty = hir.opaque_unpack %ctx : !hir.any
  hir.signature (%ty, %o) -> (%ty)
} {
  %ty = hir.opaque_unpack %ctx : !hir.any
  %r = hir.add %x, %x : %ty
  hir.return %r : %ty
}

// The split_func keeps the unsorted signature flat (no phase ops), with the
// original dependent type computation inlined in terms of the block arguments.
// This flat region is load-bearing: it is what call checking inlines at
// late-bound call sites, identically to a `uir.func` signature region (settled
// 2026-08-01).
uir.split_func @Id(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  %ty = hir.uint_type %N
  uir.signature (%int, %ty) -> (%ty)
} [
  -1: @"Id.-1",
  0: @"Id.0"
]

//===----------------------------------------------------------------------===//
// Case 2: pre-span eager chain.
//
// Threaded input `@Pre(%N: -1, %x: 0) -> (result: 0)` with a pre-span body
// phase -2 (declared span [-1, 0]):
//   body:
//     %pre = uir.phase -2 { %c = hir.constant_int 5; uir.phase_return yield %c }
//     %v   = uir.phase -1 { %w = hir.add %N, %pre;   uir.phase_return yield %w }
//            uir.phase  0 { %r = hir.add %x, %v;     uir.phase_return %r }
//
// Phase -2 is argument-free (no declared value exists that early) and is where
// the signature also computes %N's type, so the phase -2 func merges the two
// and heads an `hir.eager_func`; the earliest span entry (phase -1) is the tail.
// The eager_func's external signature is the phase -1 entry's declared arguments
// (%N) and results — the tail link `@"Pre.-1.tail"` minus its trailing pre-span
// context argument — so the split_func and any caller target it like a plain
// func.

hir.func @"Pre.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %nty = hir.int_type
  %t = hir.int_type
  %c = hir.constant_int 5 : %t
  %ctx = hir.opaque_pack(%nty, %c)
  hir.return %ctx : %o
}

hir.func @"Pre.-1.tail"(%N, %ctx_pre) -> (ctx) {
  %o = hir.opaque_type
  %nty, %pre = hir.opaque_unpack %ctx_pre : !hir.any, !hir.any
  hir.signature (%nty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %nty, %pre = hir.opaque_unpack %ctx_pre : !hir.any, !hir.any
  %xty = hir.int_type
  %int = hir.int_type
  %w = hir.add %N, %pre : %int
  %ctx = hir.opaque_pack(%xty, %w)
  hir.return %ctx : %o
}

// The pre-span func keeps its phase-number name (@"Pre.-2") and the span-entry
// name goes to the eager_func, so the phase -1 body func is disambiguated as
// @"Pre.-1.tail"; these names are cosmetic, the eager_func's chain links are
// authoritative (settled 2026-08-01). Because the signature's type computation
// is ordinary phase content, it is already distributed across these links and
// needs no separate relocation.
hir.eager_func @"Pre.-1"(N) -> (ctx) [
  @"Pre.-2",
  @"Pre.-1.tail"
]

hir.func @"Pre.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  %xty, %v = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%xty, %o) -> (%xty)
} {
  %xty, %v = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %int = hir.int_type
  %r = hir.add %x, %v : %int
  hir.return %r : %int
}

uir.split_func @Pre(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} [
  -1: @"Pre.-1",
  0: @"Pre.0"
]

//===----------------------------------------------------------------------===//
// Case 3: pass-through gap phase.
//
// Threaded input `@Gap(%a: -2, %x: 0) -> (result: 0)`, declared span [-2, 0]
// with no body content at phase -1:
//   body:
//     %c0 = uir.phase -2 { %sq = hir.mul %a, %a; uir.phase_return yield %sq }
//     %c1 = uir.phase -1 { uir.phase_return yield %c0 }   // materialized gap
//           uir.phase  0 { %r = hir.add %c1, %x; uir.phase_return %r }
//
// The gap phase carries no body content, so its func only forwards: it unpacks
// the crossings entering the phase and repacks them, alongside the signature's
// own phase -1 content (the type of %x and the result). Collapsing a degenerate
// transcoder is a later peephole, not the splitter's job. %a is declared at
// phase -2, so its type is phase -3 content and the chain reaches one phase
// below the declared span, heading an eager chain.

hir.func @"Gap.-3"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %aty = hir.int_type
  %ctx = hir.opaque_pack(%aty)
  hir.return %ctx : %o
}

hir.func @"Gap.-2.tail"(%a, %ctx_in) -> (ctx) {
  %o = hir.opaque_type
  %aty = hir.opaque_unpack %ctx_in : !hir.any
  hir.signature (%aty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %sq = hir.mul %a, %a : %int
  %ctx = hir.opaque_pack(%sq)
  hir.return %ctx : %o
}

hir.eager_func @"Gap.-2"(a) -> (ctx) [
  @"Gap.-3",
  @"Gap.-2.tail"
]

hir.func @"Gap.-1"(%ctx_in) -> (ctx_out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %sq = hir.opaque_unpack %ctx_in : !hir.any
  %xty = hir.int_type
  %ctx_out = hir.opaque_pack(%xty, %sq)
  hir.return %ctx_out : %o
}

hir.func @"Gap.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  %xty, %sq = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%xty, %o) -> (%xty)
} {
  %xty, %sq = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %int = hir.int_type
  %r = hir.add %sq, %x : %int
  hir.return %r : %int
}

uir.split_func @Gap(%a: -2, %x: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} [
  -2: @"Gap.-2",
  -1: @"Gap.-1",
  0: @"Gap.0"
]

//===----------------------------------------------------------------------===//
// Case 4: caller side.
//
// Threaded input `@Caller(%x: -1, %y: 0) -> (result: 0)` calling the two-phase
// `@MultiPhase` (declared span [-1, 0]):
//   body:
//     %ctx = uir.phase -1 {
//       %c = uir.phase_call @MultiPhase[-1] (%x) : (%t) -> () context
//       uir.phase_return yield %c
//     }
//     uir.phase 0 {
//       %r = uir.phase_call @MultiPhase[0] (%y) context %ctx : (%t) -> (%t)
//       uir.phase_return %r : %t
//     }
//
// Each `uir.phase_call` becomes an `hir.call` to the callee's per-phase entry,
// living in the caller's own per-phase func. The callee context returned by the
// phase -1 entry crosses the caller's phase boundary like any other value: it is
// packed into the caller's own outgoing context and unpacked in the caller's
// phase 0 func before the second call. Both functions carry the phase -2 type
// funcs their signatures require, so the callee's phase -1 entry — the symbol
// the caller targets — is an `hir.eager_func`, called like any plain func.

// The callee, in its split form (the entries the caller's calls target).
hir.func @"MultiPhase.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %aty = hir.int_type
  %ctx = hir.opaque_pack(%aty)
  hir.return %ctx : %o
}

hir.func @"MultiPhase.-1.tail"(%a, %ctx_in) -> (ctx) {
  %o = hir.opaque_type
  %aty = hir.opaque_unpack %ctx_in : !hir.any
  hir.signature (%aty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %bty = hir.int_type
  %int = hir.int_type
  %sq = hir.mul %a, %a : %int
  %ctx = hir.opaque_pack(%bty, %sq)
  hir.return %ctx : %o
}

hir.eager_func @"MultiPhase.-1"(a) -> (ctx) [
  @"MultiPhase.-2",
  @"MultiPhase.-1.tail"
]

hir.func @"MultiPhase.0"(%b, %ctx) -> (result) {
  %o = hir.opaque_type
  %bty, %sq = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%bty, %o) -> (%bty)
} {
  %bty, %sq = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %int = hir.int_type
  %r = hir.add %sq, %b : %int
  hir.return %r : %int
}

uir.split_func @MultiPhase(%a: -1, %b: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} [
  -1: @"MultiPhase.-1",
  0: @"MultiPhase.0"
]

// The caller, split into per-phase funcs whose bodies hold the decomposed calls.
hir.func @"Caller.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %xty = hir.int_type
  %ctx = hir.opaque_pack(%xty)
  hir.return %ctx : %o
}

hir.func @"Caller.-1.tail"(%x, %ctx_in) -> (ctx) {
  %o = hir.opaque_type
  %xty = hir.opaque_unpack %ctx_in : !hir.any
  hir.signature (%xty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %yty = hir.int_type
  %int = hir.int_type
  // A decomposed uir.phase_call targets the callee's concrete per-phase entry
  // (@"MultiPhase.-1", here an eager_func), resolved through the @MultiPhase
  // split_func's link list; the split_func itself is the late-binding target
  // only for a phase-agnostic uir.call (settled 2026-08-01).
  %cc = hir.call @"MultiPhase.-1"(%x) : (%int) -> (%o)
  // The single callee context crossing is re-wrapped in the caller's own
  // context bundle rather than threaded directly: packing must stay symmetric
  // with the receiving phase's unpacking, so there is no special case for
  // crossing count or already-opaque crossings (settled 2026-08-01).
  %ctx = hir.opaque_pack(%yty, %cc)
  hir.return %ctx : %o
}

hir.eager_func @"Caller.-1"(x) -> (ctx) [
  @"Caller.-2",
  @"Caller.-1.tail"
]

hir.func @"Caller.0"(%y, %ctx) -> (result) {
  %o = hir.opaque_type
  %yty, %cc = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%yty, %o) -> (%yty)
} {
  %o = hir.opaque_type
  %yty, %cc = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %int = hir.int_type
  %r = hir.call @"MultiPhase.0"(%y, %cc) : (%int, %o) -> (%int)
  hir.return %r : %int
}

uir.split_func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %int = hir.int_type
  uir.signature (%int, %int) -> (%int)
} [
  -1: @"Caller.-1",
  0: @"Caller.0"
]

//===----------------------------------------------------------------------===//
// Case 5: split caller of a dependent-signature callee (the load-bearing
// interaction of cases 1 and 4).
//
// `@Use(%N: -1, %x: 0) -> (r: 0)`, `x: uint[N]`, calling the dependent `@Id`
// from case 1: `id(N, x)`. This is where the "how does the phase-0 call site
// obtain %x's dependent type?" question bites, since `@Id.0`'s argument type
// uint[N] is computed inside `@Id`'s own phase -1 and rides `@Id`'s *opaque*
// context, which the caller cannot see into.
//
// Resolution (matching the old-model rule that CheckCalls inlines the callee's
// signature into the caller): the caller recomputes uint[N] itself from the
// const argument %N it already holds, in its own phase -1, and threads it
// forward through the caller's *own* context. The callee's opaque context
// (%idctx) stays private callee plumbing and is never consulted for types; the
// caller supplies `typeOfArgs` for the `hir.call @"Id.0"` independently. Thus
// one uniform opaque-context mechanism carries both the caller's own type
// crossing and the callee's context, side by side.
//
// Settled 2026-08-01: "caller recomputes and threads the type; callee context
// is never read for types" is the design. Call checking inlines the callee's
// signature computation into the caller before sorting (from the signature
// region of the `uir.func` or `uir.split_func` the callee symbol resolves to,
// identically) and unifies it with the caller's types, so agreement across the
// split boundary is structural: checking either rejects the program with a user
// diagnostic or guarantees both sides pick the same types. A disagreement
// observed when the evaluator specializes a per-phase func is an internal
// compiler error, not a user diagnostic.

hir.func @"Use.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %nty = hir.int_type
  %ctx = hir.opaque_pack(%nty)
  hir.return %ctx : %o
}

hir.func @"Use.-1.tail"(%N, %ctx_in) -> (ctx) {
  %o = hir.opaque_type
  %nty = hir.opaque_unpack %ctx_in : !hir.any
  hir.signature (%nty, %o) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  // The dependent argument type the caller computes in its own phase -1, for
  // the phase-0 call site's typeOfArgs. It is signature content here, not a
  // relocated copy: the signature region carries this computation at phase -1
  // and splitting merges it into this func.
  %xty = hir.uint_type %N
  // The callee's phase -1 entry; its result is the callee's private context.
  %idctx = hir.call @"Id.-1"(%N) : (%int) -> (%o)
  // Both cross the caller's phase boundary in one opaque context.
  %ctx = hir.opaque_pack(%xty, %idctx)
  hir.return %ctx : %o
}

hir.eager_func @"Use.-1"(N) -> (ctx) [
  @"Use.-2",
  @"Use.-1.tail"
]

hir.func @"Use.0"(%x, %ctx) -> (r) {
  %o = hir.opaque_type
  %xty, %idctx = hir.opaque_unpack %ctx : !hir.any, !hir.any
  hir.signature (%xty, %o) -> (%xty)
} {
  %o = hir.opaque_type
  %xty, %idctx = hir.opaque_unpack %ctx : !hir.any, !hir.any
  %r = hir.call @"Id.0"(%x, %idctx) : (%xty, %o) -> (%xty)
  hir.return %r : %xty
}

uir.split_func @Use(%N: -1, %x: 0) -> (r: 0) {
  %int = hir.int_type
  %xty = hir.uint_type %N
  uir.signature (%int, %xty) -> (%xty)
} [
  -1: @"Use.-1",
  0: @"Use.0"
]
