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
// - One `hir.func` per `uir.phase` op, in phase order. The first phase op
//   overall (a pre-span phase when there is one, otherwise the first span phase)
//   takes no incoming context; every later func — including the first span phase
//   when preceded by pre-span content — takes an extra trailing opaque context
//   argument; every func but the last produces an extra trailing opaque context
//   result. Each func also takes its phase's declared arguments. Function
//   results ride the func of the phase they belong to; the last func returns no
//   context.
// - Values crossing a phase boundary are packed into the outgoing context by
//   the producing func (`hir.opaque_pack`) and unpacked by the consuming func
//   (`hir.opaque_unpack`). A dependent argument type needed by a later phase's
//   signature rides the same context and is unpacked in that func's signature
//   region.
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
// Case 1: simple two-phase split of a dependent-type identity function.
//
// Threaded input `@Id(%N: -1, %x: 0) -> (result: 0)`, `x: uint[N]`:
//   body:
//     %p = uir.phase -1 { %ty = hir.uint_type %N; uir.phase_return yield %ty }
//          uir.phase  0 { %r = hir.add %x, %x : %p; uir.phase_return %r : %p }
//   signature threads the uint type to phase 0 for %x and the result.
//
// The dependent type uint[N] is computed once, in the phase -1 func, and rides
// the opaque context into phase 0, where it is unpacked both in the signature
// region (as the type of %x and the result) and in the body (as the add type).

hir.func @"Id.-1"(%N) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int) -> (%o)
} {
  %o = hir.opaque_type
  %ty = hir.uint_type %N
  %ctx = hir.opaque_pack(%ty)
  hir.return %ctx : %o
}

hir.func @"Id.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  // The dependent type of %x is delivered by unpacking the incoming context in
  // the signature region (sig-crossing relocation). A split caller specializing
  // phase 0 never reads this context for types: call checking inlines the
  // callee's signature computation into the caller, which recomputes the type
  // in its own earlier phase (settled 2026-08-01; exercised in case 5).
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
// Threaded input `@Pre(%N: -1, %x: 0) -> (result: 0)` with a pre-span phase -2
// (declared span [-1, 0]):
//   body:
//     %pre = uir.phase -2 { %c = hir.constant_int 5; uir.phase_return yield %c }
//     %v   = uir.phase -1 { %w = hir.add %N, %pre;   uir.phase_return yield %w }
//            uir.phase  0 { %r = hir.add %x, %v;     uir.phase_return %r }
//
// Phase -2 is argument-free (no declared value exists that early), so its func
// heads an `hir.eager_func`; the earliest span entry (phase -1) is the tail.
// The eager_func's external signature is the phase -1 entry's declared arguments
// (%N) and results — the tail link `@"Pre.-1.tail"` minus its trailing pre-span
// context argument — so the split_func and any caller target it like a plain
// func.

hir.func @"Pre.-2"() -> (ctx) {
  %o = hir.opaque_type
  hir.signature () -> (%o)
} {
  %o = hir.opaque_type
  %t = hir.int_type
  %c = hir.constant_int 5 : %t
  %ctx = hir.opaque_pack(%c)
  hir.return %ctx : %o
}

hir.func @"Pre.-1.tail"(%N, %ctx_pre) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int, %o) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %pre = hir.opaque_unpack %ctx_pre : !hir.any
  %w = hir.add %N, %pre : %int
  %ctx = hir.opaque_pack(%w)
  hir.return %ctx : %o
}

// The pre-span func keeps its phase-number name (@"Pre.-2") and the span-entry
// name goes to the eager_func, so the phase -1 body func is disambiguated as
// @"Pre.-1.tail"; these names are cosmetic, the eager_func's chain links are
// authoritative (settled 2026-08-01). Relocating dependent pre-span signature
// computation into the eager links (settled 2026-07-29) is not exercised here,
// since @Pre has a non-dependent signature.
hir.eager_func @"Pre.-1"(N) -> (ctx) [
  @"Pre.-2",
  @"Pre.-1.tail"
]

hir.func @"Pre.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int, %o) -> (%int)
} {
  %int = hir.int_type
  %v = hir.opaque_unpack %ctx : !hir.any
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
// with no content at phase -1:
//   body:
//     %c0 = uir.phase -2 { %sq = hir.mul %a, %a; uir.phase_return yield %sq }
//     %c1 = uir.phase -1 { uir.phase_return yield %c0 }   // materialized gap
//           uir.phase  0 { %r = hir.add %c1, %x; uir.phase_return %r }
//
// The materialized empty gap phase becomes a context-forwarding func: it takes
// the incoming context and returns it unchanged. It is still a full span entry
// callers thread through lock-step.

hir.func @"Gap.-2"(%a) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %sq = hir.mul %a, %a : %int
  %ctx = hir.opaque_pack(%sq)
  hir.return %ctx : %o
}

// The gap func is emitted by the same uniform rule as every other per-phase
// func: unpack the crossings entering the phase, run the (empty) phase body,
// repack exactly the yielded crossings. Here the single crossing %sq is
// unpacked and repacked unchanged. Collapsing this degenerate transcoder to a
// bare `hir.return %ctx_in` is a later peephole, not the splitter's job: the
// splitter stays phase-blind and content-agnostic.
hir.func @"Gap.-1"(%ctx_in) -> (ctx_out) {
  %o = hir.opaque_type
  hir.signature (%o) -> (%o)
} {
  %o = hir.opaque_type
  %sq = hir.opaque_unpack %ctx_in : !hir.any
  %ctx_out = hir.opaque_pack(%sq)
  hir.return %ctx_out : %o
}

hir.func @"Gap.0"(%x, %ctx) -> (result) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int, %o) -> (%int)
} {
  %int = hir.int_type
  %sq = hir.opaque_unpack %ctx : !hir.any
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
// phase 0 func before the second call.

// The callee, in its split form (the entries the caller's calls target).
hir.func @"MultiPhase.-1"(%a) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %sq = hir.mul %a, %a : %int
  %ctx = hir.opaque_pack(%sq)
  hir.return %ctx : %o
}

hir.func @"MultiPhase.0"(%b, %ctx) -> (result) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int, %o) -> (%int)
} {
  %int = hir.int_type
  %sq = hir.opaque_unpack %ctx : !hir.any
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
hir.func @"Caller.-1"(%x) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  // A decomposed uir.phase_call targets the callee's concrete per-phase func
  // (@"MultiPhase.-1"), resolved through the @MultiPhase split_func's link
  // list; the split_func itself is the late-binding target only for a
  // phase-agnostic uir.call (settled 2026-08-01).
  %cc = hir.call @"MultiPhase.-1"(%x) : (%int) -> (%o)
  // The single callee context crossing is re-wrapped in the caller's own
  // context bundle (opaque_pack of one opaque) rather than threaded directly:
  // packing must stay symmetric with the receiving phase's unpacking, so there
  // is no special case for crossing count or already-opaque crossings (settled
  // 2026-08-01).
  %ctx = hir.opaque_pack(%cc)
  hir.return %ctx : %o
}

hir.func @"Caller.0"(%y, %ctx) -> (result) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int, %o) -> (%int)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  %cc = hir.opaque_unpack %ctx : !hir.any
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
// const argument %N it already holds, in its own phase -1 body, and threads it
// forward through the caller's *own* context as an ordinary sig-crossing type
// value. The callee's opaque context (%idctx) stays private callee plumbing and
// is never consulted for types; the caller supplies `typeOfArgs` for the
// `hir.call @"Id.0"` independently. Thus one uniform opaque-context mechanism
// carries both the caller's recomputed type crossing and the callee's context,
// side by side.
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

hir.func @"Use.-1"(%N) -> (ctx) {
  %o = hir.opaque_type
  %int = hir.int_type
  hir.signature (%int) -> (%o)
} {
  %o = hir.opaque_type
  %int = hir.int_type
  // The dependent argument type recomputed in the caller's own body (relocated
  // sig-crossing), for the phase-0 call site's typeOfArgs.
  %xty = hir.uint_type %N
  // The callee's phase -1 entry; its result is the callee's private context.
  %idctx = hir.call @"Id.-1"(%N) : (%int) -> (%o)
  // Both cross the caller's phase boundary in one opaque context.
  %ctx = hir.opaque_pack(%xty, %idctx)
  hir.return %ctx : %o
}

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
