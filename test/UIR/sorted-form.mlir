// RUN: silicon-opt --verify-roundtrip %s

// # The Sorted-Form Contract
//
// Hand-written phase-op IR for the canonical phase-sorting examples. This
// file is the output contract of phase sorting: the sorting pass must
// produce these shapes, and the function splitter's normalization pre-pass
// starts from them.
//
// The contract, per uir.func region (body and signature follow the same
// rules):
//
// - The region top level holds only uir.phase ops (at most one per phase, in
//   increasing phase order), floating ops, and the region terminator
//   (uir.unreachable for the body once returns have moved into phases,
//   uir.signature for the signature).
// - Values computed inside a phase and consumed in a later one cross as
//   phase-op results: yielded by uir.phase_return, consumed by any later
//   phase, at any distance. Values already visible at the region top level
//   (function arguments and floating ops) are consumed directly, without
//   routing through a phase op. A floating op that uses values moved into
//   phase regions sinks into the latest such phase; its operands from
//   earlier phases cross as phase results like any other use.
// - Function results ride on the `values` of uir.phase_return in the phase
//   they belong to; a phase_return carrying values or yields is content, so
//   a result-only phase still gets its phase op, while contentless phases
//   get none.
// - Calls are decomposed into per-callee-phase uir.phase_call ops chained
//   through an opaque context value; all entries of one decomposed call
//   share a single frame shift, and the context crosses phases like any
//   other value.
//
// Normalization (in or immediately before the function splitter) later
// materializes phase ops for externally visible phases without content and
// for intermediate hops, threads all crossings hop-by-hop through adjacent
// phase ops, and sinks floating ops into the phases that use them; none of
// that is sorting's concern, and none of it appears here.

//===----------------------------------------------------------------------===//
// Straight-line, single phase: all content wraps into one uir.phase op whose
// phase return carries the function result; the body terminator becomes
// uir.unreachable. The signature has no anchored content, so it gets no
// phase op at all: floating constants stay at the region top level.

uir.func @SinglePhase(%a: 0, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.phase 0 {
    %sum = hir.add %a, %b : %t
    uir.phase_return %sum : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Straight-line, two phases: phase -1 computes from the const argument and
// yields the crossing value; phase 0 consumes it as an ordinary SSA value
// and returns the function result.

uir.func @MultiPhase(%a: -1, %b: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %p = uir.phase -1 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return yield %sq
  }
  uir.phase 0 {
    %r = hir.add %p, %b : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Skip crossing: a value is yielded once, from the phase computing it, and
// consumed directly at any later phase; here %p is consumed at phase -1 and
// again at phase 0, skipping over the -1 op. Normalization later re-threads
// the phase 0 use hop-by-hop.

uir.func @SkipCrossing(%a: -2, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %p = uir.phase -2 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return yield %sq
  }
  uir.phase -1 {
    %v = hir.var : %t
    hir.write_var %v, %p : %t
    uir.phase_return
  }
  uir.phase 0 {
    %r = hir.add %p, %x : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Function results at different phases: each phase's phase return carries
// exactly the function results belonging to that phase. A value may double
// as a function result and a crossing (returned in `values`, yielded in
// `yields`).

uir.func @ResultsPerPhase(%a: -1, %b: 0) -> (c: -1, d: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0, %0)
} {
  %t = hir.int_type
  %p = uir.phase -1 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return %sq : %t yield %sq
  }
  uir.phase 0 {
    %r = hir.add %p, %b : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Result-only phase: a uir.phase_return carrying function results is
// content, so a phase whose only job is returning a value still gets its
// phase op.

uir.func @ResultOnlyPhase(%a: -1) -> (c: -1) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  uir.phase -1 {
    uir.phase_return %a : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Internal phases: content past the last declared phase (here effect-only
// phase 1 content in a function whose declaration only mentions phase 0)
// wraps into a uir.phase op like any other phase. Callers cannot target it;
// how it executes is a function-splitting concern.

uir.func @TrailingInternalPhase(%a: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0) -> (%0)
} {
  %t = hir.int_type
  %p = uir.phase 0 {
    %sq = hir.mul %a, %a : %t
    uir.phase_return %sq : %t yield %sq
  }
  uir.phase 1 {
    %v = hir.var : %t
    hir.write_var %v, %p : %t
    uir.phase_return
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// No empty phases: an externally visible phase without content (the unused
// const argument's phase -1) gets no uir.phase op. Normalization later
// materializes the empty phase op for it.

uir.func @NoEmptyPhases(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.phase 0 {
    %r = hir.add %x, %x : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Direct argument crossing: a function argument consumed past its declared
// phase is used directly; sorting does not route it through a phase op at
// its home phase, which stays absent here since it has no content.

uir.func @ArgCrossing(%a: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  uir.phase 0 {
    %r = hir.add %a, %x : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Call decomposition: a call to a two-phase callee becomes one
// uir.phase_call per externally visible callee phase, each inside the
// caller's phase where that entry executes. The context value chaining the
// entries crosses phases like any other value.

uir.func @Caller(%x: -1, %y: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ctx = uir.phase -1 {
    %c = uir.phase_call @MultiPhase[-1] (%x) : (%t) -> () context
    uir.phase_return yield %c
  }
  uir.phase 0 {
    %r = uir.phase_call @MultiPhase[0] (%y) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Frame-shifted call: the calleePhase is in the callee's own frame; the
// enclosing uir.phase is in the caller's frame. A call executing one phase
// later than the callee's frame (a dyn-shifted call) places the callee's
// [-1] entry in the caller's phase 0 and the [0] entry in phase 1.

uir.func @ShiftedCall(%k: 0, %x: 1) -> (result: 1) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ctx = uir.phase 0 {
    %c = uir.phase_call @MultiPhase[-1] (%k) : (%t) -> () context
    uir.phase_return yield %c
  }
  uir.phase 1 {
    %r = uir.phase_call @MultiPhase[0] (%x) context %ctx : (%t) -> (%t)
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Signature-region phases: a dependent signature computes the phase-0
// argument and result type from the const argument in its own phase -1 op;
// uir.signature consumes the phase result. The body cannot reference values
// from the signature region and recomputes the type in its own phase -1 op.

uir.func @DependentSignature(%N: -1, %x: 0) -> (result: 0) {
  %int = hir.int_type
  %p = uir.phase -1 {
    %ty = hir.uint_type %N
    uir.phase_return yield %ty
  }
  uir.signature (%int, %p) -> (%p)
} {
  %q = uir.phase -1 {
    %ty = hir.uint_type %N
    uir.phase_return yield %ty
  }
  uir.phase 0 {
    %r = hir.add %x, %x : %q
    uir.phase_return %r : %q
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Branch extraction, effects: a const if with dyn content in both arms. The
// rebuilt if yields the const result and a per-arm choice; both cross to
// phase 0, where a pick dispatches on the choice and arm-local const values
// arrive as the choice payload.

uir.func @IfChoiceEffects(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ct = hir.choice_type
  %p:2 = uir.phase -1 {
    %v:2 = uir.if %c : %t, %ct {
      %a = hir.add %c, %c : %t
      %c0 = hir.choice 0(%a)
      uir.yield %a, %c0 : %t, %ct
    } else {
      %c1 = hir.choice 1()
      uir.yield %c, %c1 : %t, %ct
    }
    uir.phase_return yield %v#0, %v#1
  }
  uir.phase 0 {
    %va = hir.var : %t
    hir.pick %p#1 [0 (%a): {
      hir.write_var %va, %a : %t
      hir.yield
    } 1 (): {
      hir.write_var %va, %x : %t
      hir.yield
    }]
    %out = hir.add %x, %p#0 : %t
    uir.phase_return %out : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Branch extraction, dyn result: the if's dyn-phase result is removed from
// the if and reconstructed as a result of the receiving pick, whose
// alternatives recompute the per-arm values.

uir.func @IfChoiceResult(%c: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ct = hir.choice_type
  %p = uir.phase -1 {
    %ch = uir.if %c : %ct {
      %c0 = hir.choice 0()
      uir.yield %c0 : %ct
    } else {
      %c1 = hir.choice 1()
      uir.yield %c1 : %ct
    }
    uir.phase_return yield %ch
  }
  uir.phase 0 {
    %v = hir.pick %p : !hir.any [0 (): {
      %s = hir.add %x, %x : %t
      hir.yield %s : !hir.any
    } 1 (): {
      %m = hir.mul %x, %x : %t
      hir.yield %m : !hir.any
    }]
    uir.phase_return %v : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Loop records: a const loop with dyn content pushes one record per
// activation onto a hit list (an exited link for the break path, a continued
// link carrying the iteration's snapshots otherwise). The list crosses to
// phase 0, where a replicate replays the records and a pick dispatches each.

uir.func @LoopRecords(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %p = uir.phase -1 {
    %list = hir.opaque_list_create
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      uir.if %done {
        %c0 = hir.choice 0()
        %r0 = hir.opaque_pack(%c0)
        hir.opaque_list_push %list, %r0
        uir.break
      }
      %next = hir.add %i, %one : %t
      %c1 = hir.choice 1(%i)
      %r1 = hir.opaque_pack(%c1)
      hir.opaque_list_push %list, %r1
      uir.continue %next : %t
    }
    uir.phase_return yield %list
  }
  uir.phase 0 {
    %acc = hir.var : %t
    hir.replicate (%rec) in %p {
      hir.pick %rec [0 (): {
        hir.yield
      } 1 (%i): {
        %cur = hir.read_var %acc : %t
        %new = hir.add %cur, %i : %t
        hir.write_var %acc, %new : %t
        hir.yield
      }]
      hir.yield
    }
    %r = hir.read_var %acc : %t
    uir.phase_return %r : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Break-value choice: a const loop whose break carries a dyn value breaks
// with a choice instead; the choice crosses to phase 0, where a pick
// reconstructs the value from the break site's snapshots.

uir.func @BreakValueChoice(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ct = hir.choice_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %p = uir.phase -1 {
    %ch = uir.loop (%k = %zero : %t) : %ct {
      %done = hir.geq %k, %n : %t
      uir.if %done {
        %c0 = hir.choice 0(%k)
        uir.break %c0 : %ct
      }
      %next = hir.add %k, %one : %t
      uir.continue %next : %t
    }
    uir.phase_return yield %ch
  }
  uir.phase 0 {
    %y = hir.pick %p : !hir.any [0 (%k): {
      %s = hir.add %x, %k : %t
      hir.yield %s : !hir.any
    }]
    uir.phase_return %y : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Rejoining-arm content: a divergence with dyn content in its arms carries two
// record slots. The cf choice (one tag per arm) dispatches the arm content's
// replay and carries its samples; it is built at every arm-ending point — at
// an exit site for a path that ends the activation inside the arm, at the arm
// yield otherwise — and rides the rebuilt if's result on rejoining paths. The
// reach choice gates the following segment: REACHED with the segment's samples
// at the back edge, empty NOT_REACHED at exit sites. The receiving replicate
// replays by data alone — a cf pick (arm dispatch), then a reach pick (segment
// gate) — with no exit ops; a record's tags fully determine which content runs.

uir.func @RejoiningArmContent(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ct = hir.choice_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  %p = uir.phase -1 {
    %list = hir.opaque_list_create
    uir.loop (%i = %zero : %t) {
      %done = hir.geq %i, %n : %t
      %cf = uir.if %done : %ct {
        %c0 = hir.choice 0(%i)
        %l0 = hir.choice 0()
        %r0 = hir.opaque_pack(%c0, %l0)
        hir.opaque_list_push %list, %r0
        uir.break
      } else {
        %m = hir.add %i, %one : %t
        %c1 = hir.choice 1(%m)
        uir.yield %c1 : %ct
      }
      %next = hir.add %i, %one : %t
      %l1 = hir.choice 1(%i)
      %r1 = hir.opaque_pack(%cf, %l1)
      hir.opaque_list_push %list, %r1
      uir.continue %next : %t
    }
    uir.phase_return yield %list
  }
  uir.phase 0 {
    %vacc = hir.var : %t
    hir.replicate (%cfr, %lr) in %p {
      hir.pick %cfr [0 (%si): {
        hir.write_var %vacc, %si : %t
        hir.yield
      } 1 (%sm): {
        hir.write_var %vacc, %sm : %t
        hir.yield
      }]
      hir.pick %lr [0 (): {
        hir.yield
      } 1 (%se): {
        hir.write_var %vacc, %se : %t
        hir.yield
      }]
      hir.yield
    }
    %rd = hir.read_var %vacc : %t
    uir.phase_return %rd : %t
  }
  uir.unreachable
}

//===----------------------------------------------------------------------===//
// Shared arm prefix: dyn content before a nested exit inside a rejoining arm
// runs on both the exit path and the fall-through. Its replay lives once in
// the cf pick's arm alternative, ahead of the arm's own nested reach pick, so
// no content is cloned across record variants. The cf payload nests the arm's
// reach link (NOT_REACHED at the nested break, REACHED with the fall-through
// samples at the arm yield). A post-branch sample (%g, effectful at phase -1)
// exists only on continuing activations; it rides the loop-level reach
// choice's REACHED payload, and exit records carry the empty NOT_REACHED tag.

uir.func @SharedArmPrefix(%n: -1, %x: 0) -> (result: 0) {
  %0 = hir.int_type
  uir.signature (%0, %0) -> (%0)
} {
  %t = hir.int_type
  %ct = hir.choice_type
  %zero = hir.constant_int 0 : %t
  %one = hir.constant_int 1 : %t
  %acc = hir.var : %t
  %p = uir.phase -1 {
    %v = hir.var : %t
    %list = hir.opaque_list_create
    uir.loop (%i = %zero : %t) {
      hir.write_var %v, %i : %t
      %c = hir.geq %i, %n : %t
      %cf = uir.if %c : %ct {
        %d = hir.geq %n, %i : %t
        uir.if %d {
          %a0 = hir.choice 0()
          %c0 = hir.choice 0(%i, %a0)
          %l0 = hir.choice 0()
          %r0 = hir.opaque_pack(%c0, %l0)
          hir.opaque_list_push %list, %r0
          uir.break
        }
        %m = hir.add %i, %one : %t
        %a1 = hir.choice 1(%m)
        %c1 = hir.choice 0(%i, %a1)
        uir.yield %c1 : %ct
      } else {
        %c2 = hir.choice 1()
        uir.yield %c2 : %ct
      }
      %g = hir.read_var %v : %t
      %next = hir.add %i, %one : %t
      %l1 = hir.choice 1(%g)
      %r1 = hir.opaque_pack(%cf, %l1)
      hir.opaque_list_push %list, %r1
      uir.continue %next : %t
    }
    uir.phase_return yield %list
  }
  uir.phase 0 {
    %vacc = hir.var : %t
    hir.replicate (%cfr, %lr) in %p {
      hir.pick %cfr [0 (%si, %alink): {
        hir.write_var %vacc, %si : %t
        hir.pick %alink [0 (): {
          hir.yield
        } 1 (%sm): {
          hir.write_var %vacc, %sm : %t
          hir.yield
        }]
        hir.yield
      } 1 (): {
        hir.yield
      }]
      hir.pick %lr [0 (): {
        hir.yield
      } 1 (%sg): {
        hir.write_var %vacc, %sg : %t
        hir.yield
      }]
      hir.yield
    }
    %rd = hir.read_var %vacc : %t
    uir.phase_return %rd : %t
  }
  uir.unreachable
}
