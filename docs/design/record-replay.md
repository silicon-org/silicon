# Loop Records and Data-Only Replay

Fixes the record shape and replay form of {{< page-link "/design/sampling-model" >}} §6, and the loop-record parts of {{< page-link "/design/phase-ops" >}}.
Function-level exits join the same model through exit normalization ({{< page-link "/design/function-exit-choices" >}}): early returns rewrite to choice/pick value form before analysis, so the machinery below covers them with no exit-specific rules.

## Why two slots per divergence

A record must answer two independent questions about each divergence it crosses: *which arm ran*, and *did control reconverge afterwards*.
Fusing them into a single exit-vs-continued choice forces arm content shared between an exit path and the fall-through — content before a nested exit in a rejoining arm — to be cloned into several receiving pick alternatives.
Keeping the two facts in separate record slots removes the cloning entirely, and is the reason the model below is shaped the way it is.

## The model

1. **A record carries up to two slots per divergence: a cf choice and a reach link.**
   The cf choice tags which arm of the divergence ran; its payload is the arm's own record body (the arm's first-segment samples, then recursively the arm's nested cf and reach slots).
   The reach link gates everything after the divergence: an empty `NOT_REACHED` tag ends the record at an exit, and a `REACHED` tag carries the following segment's samples plus the next divergence's slots.
   A divergence needs a cf slot only when some arm holds later-phase content or the branch has a demanded later-phase result; a content-free divergence contributes only its reach link.

2. **Construction sites follow scope.**
   Arm-interior samples can only be captured inside the arm, so the cf choice is built at every arm-ending point: at an activation-ending terminator inside the arm (exit paths), and at the arm's yield (fall-through), where it becomes the rebuilt branch's last result.
   Post-branch samples exist only on paths that reconverged, so they ride the reach link's `REACHED` payload, built at the push site.
   Every activation-ending terminator pushes exactly one record in place, bottom-up: passed divergences contribute their branch's cf result and a `REACHED` link; the exiting divergence contributes a cf built along the site's own path and a `NOT_REACHED` link.
   Records stay uniform (same pack arity) because path-dependent existence lives inside the sum tags, never in the pack shape — no undef or poison filler.

3. **The receiving replay is data-only.**
   Per level: the segment's content replays first, then the cf pick dispatches the arm that ran (recursing into the arm's own levels), then the reach pick either stops or binds the next segment's samples and continues one level deeper in its `REACHED` alternative.
   No exit terminator appears in replayed content: every exit decision for phase-p CF is made at phase p and recorded, so the tags fully determine which content replays, and the list itself ends after an exit record.
   Function exits are no exception — they are normalized to value form before analysis (see function-exit-choices), so receiving phases thread exit values through the picks to a single `uir.phase_return` instead of holding returns.

4. **Every piece of content replays in exactly one place.**
   Content shared between an exit path and the fall-through sits in the arm's first segment and replays in the cf pick's arm alternative, before the arm's own reach pick — once, whichever way the activation ended.
   This is what makes shared prefixes and nested fall-through work without cloning content into several alternatives.

5. **Safety is SSA dominance.**
   A value is recorded exactly on the paths that compute it, and the replay preserves the source's path structure, so each field is read exactly on the paths that replay its consumers.
   A consumer that could run without its operand cannot exist: it would not have been dominated by the operand's definition in the source.

6. **Every phase boundary records afresh.**
   The producing region for boundary p is the original control flow at the earliest boundary and the receiving `hir.replicate` at every later one; a replicate is an ordinary producing frame, with its layered phase-p control flow, and pushes its own hit list for the content one phase beyond it.
   Per-phase samples therefore ferry hop by hop instead of skipping ahead, and replay is blind: phase p+1 sees only the records phase p pushed, never a record from further back.
   Iteration-invariant values still cross directly, since they are not per-activation data.
   A phase whose replicate does nothing but re-pack one hit list into the next is correct output; collapsing such a pass-through transcoder is a peephole optimization, never part of the lowering.

7. **A later-phase loop result is delivered, not selected.**
   Break values are not a delivery concept of their own.
   A loop result whose providers are later than the loop is lowered onto a var homed in the receiving phase: each `break` stores its later-phase operand into the var and drops it, the loop drops the result, and a read after the loop feeds the result's uses.
   The store is then ordinary later-phase content sitting on the break's record path, replayed once per activation by the machinery above, so no content is sliced per break site and several breaks sharing content need no special treatment.

8. **Demanded later-phase results ride the cf pick's yields.**
   The rebuilt phase-p branch cannot deliver a later value, so the cf pick is valued: each arm alternative yields that arm's operand for the result (its own replayed content, or a value from outside the arm).
   This covers only genuine selections: a result with more than one providing arm.
   A single-provider result whose sole provider is a value defined outside the op — or a floating pure value, materializable anywhere by cloning its tree — is not a selection and never reaches the cf pick: it is a transparent forward, rewritten to its provider operand (or the clone) before extraction runs, in either phase direction (see the phase-sorting `forwardSingleProviderResults` step).
   An arm that provides no value for a result (an exit arm, a short yield, a materialized else) yields `uir.unreachable_value`; its consumers replay only on paths the reach links gate off, and evaluating one is a loud interpreter trap.

## Relation to the program-order rule

Function-exit-choices §1 places continuations in program order after a pick, never nested into alternatives, because nesting duplicates the continuation across several rejoining arms.
The reach pick refines this: it is always binary (one rejoining alternative), so nesting the continuation in its `REACHED` alternative cannot duplicate anything.
Multi-arm dispatch — the actual duplication hazard — lives exclusively in the cf pick, whose alternatives hold only their own arm's content.

## Consequences

- A content-free divergence records only its reach link, so which of several exits fired is not recorded when nothing downstream depends on it.
- The interpreter (M4) must trap on evaluating `uir.unreachable_value` during replay; reaching one means the record invariant was violated.
- Function-level exits, including returns under loops, are covered by the rules above: exit normalization turns them into ordinary value CF before analysis.
- A demanded later result the rejoining arm provides is reconstructed at whichever level defines it: a value produced on the arm's spine after a nested exit threads up through valued reach picks, and a value that is itself a nested divergence's result surfaces as that divergence's own valued cf pick result and threads up identically — one recursion of the valued-carrier mechanism, no new record content.
