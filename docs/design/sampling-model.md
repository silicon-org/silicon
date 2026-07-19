# The Sampling Model (v6, 2026-07-19)

Semantic model for phased execution and phase splitting, distilled from the design interview and hardened by three adversarial review rounds.
v4 adds §11 (references are opaque; home phases are evaluation-time data), settled in the ref-opacity interview on 2026-07-09.
v5 revised the early-exit mechanism per `function-exit-choices.md` (2026-07-10: exit chains at return sites, multiple returns in receiving phases).
v6 follows the 2026-07-19 revision of that document: early returns normalize to single-exit choice/pick value form before phase analysis, receiving phases are single-exit, and the split cf/reach record model of `record-replay.md` covers function exits with no exit-specific mechanism; where §6/§7 below conflict with those documents, they win.
Status: reviewed draft; supersedes control-flow.md, phase-inference.md, and unified-dialect.md where they conflict (docs rewrite scheduled as milestone M6 in `implementation-plan.md`).

## 1. Execution model

- Phases are relative integers; `const` shifts -1, `dyn` shifts +1, stacking.
  Phase p runs to completion before phase p+1 starts.
  After phase p completes, every phase-p value is a frozen constant, observable by all later phases.
- **Causality rule**: an op at phase p can never consume a value produced at a phase later than p.
  PhaseAnalysis enforces this as a hard error (e.g. a hoisted callee-earlier-half whose argument is a later-phase value); every rule below refines but never overrides it.
- **Rearrangement test**: a program's meaning is defined by rearranging its ops into phase order with `---` separators, preserving program order *within* each phase, after cross-phase reads have been made explicit as sampling ops (§3).
  Every rule below must be derivable from this test.
- **Unrolling Law** (sanity check): `for const i in 0..4 { body }` is equivalent to 4 sequential copies of the *at-or-after-body-phase content* of `body`, with `i` substituted as `const { 0 }` .. `const { 3 }`.
  - Scope: content at phases *earlier than the loop* is NOT multiplied — it rearranges out of the loop entirely (§4). A phase -2 write under a phase -1 loop executes once.
  - Carve-out: bodies containing loop-phase `break`/`continue` unroll with exit resolution (later copies after a taken `break` are dead).

## 2. Vars

- A var has a **home phase**: the phase of its declaration site, shifted by `const`/`dyn` modifiers.
  - Writable **only** at exactly the home phase.
  - Readable at the home phase and every later phase. After the home phase completes the var is frozen.
  - Both rules are checked at **evaluation time**, not by phase analysis (§11).
- Storage is **per call frame** (per execution of the enclosing function's phase), never static. Compile-time recursion requires this.
- Reading storage that was never written is **poison** for now; expected to become an error/warning. Long-term the language likely moves to `let mut` with mandatory initializers, making this rare.

## 3. Sampling

- A use of a var (or any earlier-phase state) from a later phase is mediated by a **sample**: a read operation *at the state's home phase*, positioned at the use's program-order position within the home phase.
  Only the snapshot (an ordinary SSA value of the home phase) crosses the boundary.
- Rationale: `k` in source denotes "the value of `k` at this point in the program" (auto-deref intuition). Mem2reg-style cleanup must preserve sampling positions.
- Pure/immutable SSA values degenerate safely: their sample position is irrelevant; sample ops of pure values may float.
- Consequence: a later-phase read positioned before a home-phase write observes the pre-write value; "final frozen value" is just the special case where no writes follow.

## 4. The D-rule: earlier phases are blind to later CF

- Earlier-phase execution is governed **solely** by earlier-phase control flow.
  Later-phase CF is invisible: earlier-phase ops nested under later-phase `if`/`loop` execute unconditionally, once per activation of their *own* phase's enclosing CF.
- Consequences (all intended, all candidates for warnings, none for errors):
  - `const { print(i); }` in a `for const` body keeps printing for iterations after a phase-0 early `return` "hit" — a phase-0 return cannot stop a phase -1 loop.
  - `if sel { const { k = k + 1; } }` with `sel` at phase 0: the write executes unconditionally at phase -1.
  - `if x == 3 { const { break; } }` inside a `for const`: the break executes unconditionally — the loop always exits after one iteration. Legal but surprising; a refinement rule ("break/return may not pass through const/dyn region boundaries") is a possible later restriction.

## 5. Activation records and boundary objects

When a CF region at phase p contains content at phase p+1, the phase-p execution must tell phase p+1 *if and how often* that content is instantiated, and with which snapshots. Per phase boundary:

- **Record**: the tuple of snapshot SSA values that cross the `---` out of one activation of a region — precisely the SSA values that "drop out of the bottom" of the phase-p portion of the region and "pop into the top" of the phase-(p+1) reconstruction. Fields are sample ops positioned at their use sites (§3); the record is assembled at **region exit**.
- **Hit list** (loop-shaped multiplicity): ordered list of records, one appended per activation of the region.
- **Choice** (branch-shaped exclusivity: `if`/`match` arms, loop/function exits — uniformly, value-carrying or not): a tag identifying the taken alternative, plus that alternative's record as payload (possibly empty).
- Records may contain nested hit lists and choices; nesting mirrors the producing phase's CF structure and arises mechanically from §6.

Receiving-phase constructs:

- **`replicate <list> { body }`**: stamps `body` once per record, substituting the record's snapshots. No loop-carried SSA values — bodies communicate through receiving-phase vars. Zero records = no code; var state passes through untouched.
- **`pick <choice> [ tag0(payload): { ... } tag1(payload): { ... } ]`**: instantiates exactly the tagged alternative, substituting its payload.

## 6. Lowering rule: sort content across phase boundaries

Conceptually, lowering **sorts ops by phase** while preserving program order within each phase: earlier-phase content moves *up* across the `---`, later-phase content moves *down*, until everything is phase-ordered. Phase 0 is not special — it is only the default starting point of a function. The values crossing boundaries as a result of these moves *are* the records, contexts, and specialization parameters. Two directions:

**Upward (hoisting).** Earlier-phase content inside a later-phase CF region hoists to immediately before the enclosing CF op (in program order when several ops hoist). Per the D-rule it executes unconditionally, so the hoist needs no record; its results reach back into the region as ordinary downward samples. This includes the earlier-phase halves of calls sitting under later-phase CF.

**Ordering (normative).** At each boundary, hoisting runs to a fixpoint **before** extraction, and extraction asserts that its regions transitively contain no earlier-phase content. Extract-first is wrong: it would drag earlier-phase ops into the replicate body and then hoist them to *after* the CF op, changing their multiplicity — a D-rule violation (e.g. a `const` increment under a dyn `if` inside a `for const` must run once per iteration, before the if; extract-first would run it once, after the loop).

**Downward (extraction).** Applied per phase boundary, **earliest boundary first**:

- Each CF region at phase p that contains content at **any later phase** gets **one** hit list (loops) or choice (branches) and **one** `replicate`/`pick` placed immediately after the CF op, containing *all* of the region's later-phase content inlined **in program order**. Content more than one phase later rides along and is extracted again at the next boundary (the replicate is then the producing CF region).
  - NOT one list per dyn block: per-block lists replay A0 A1 B0 instead of A0 B0 A1 and break the Unrolling Law when sibling blocks share receiving-phase state.
  - Sample fields are read at their original program-order positions within the iteration (e.g. `k` sampled before a `const { k = k + 1; }` between two dyn statements), then bundled into the record at region exit.
- The new list/choice is an ordinary phase-p value. If the CF op is itself nested in more phase-p CF (or in a `replicate`/`pick` — **these count as CF regions for the next boundary**), the enclosing region's record captures it like any other snapshot. This is where nested lists come from, with no special casing.
- **Branch regions get choices uniformly**, value-carrying or not: tag per arm (payloads may be empty). The choice is **constructed per arm** (each arm yields its own tag+payload) and comes out of the CF op as an additional SSA result — the op is rebuilt with the extra result; an else-less `if` gets a materialized else region yielding the not-taken tag. `pick` is valued: variadic results with per-alternative yields. A `replicate`/`pick` is deleted only when **all** its bodies/alternatives are empty (materialize-then-DCE).
- **Early exits in the producing-phase portion of a region** (`break`/`continue`/`return` at phase p, with later-phase content after the exit point): the record ends in an **exit-path choice** recorded per activation, and the post-exit content lands in a **nested pick** in the receiving phase, driven by that choice. A partial activation's record thus ends in an inner choice whose taken payload carries the post-exit samples: for content A, exit₁, B, exit₂, C the record is `(A_samples, c₁)` with `c₁ ∈ {exited(), continued(B_samples, c₂)}`, `c₂ ∈ {exited(), continued(C_samples)}`. Construction follows `function-exit-choices.md`: each exit terminator builds its chain link in place from statically-known path dispositions; arms with later-phase content that fall back out yield a sub-choice covering the arm-local divergence. Exits that do not terminate the receiving phase (`continue`/`break`) nest their post-exit content in the `continued` alternative; a `return` terminates the receiving phase's function, so its skipped continuation stays in program order.
- **Function bodies are regions like any other**, and since the 2026-07-19 revision of `function-exit-choices.md` their exits are too: exit normalization rewrites early returns into choice/pick value form before analysis, extraction applies its ordinary rules to the result, and the function's later-phase result is a demanded later result consumed by the single final return.
- **Later-phase `return` under earlier-phase CF** extracts like any later-phase content; `return`-style terminators are legal inside replicate/pick bodies of the receiving phase. Source-level **definite-return checking** guarantees a trailing return exists, so a zero-activation replicate falls through to it. A `pick` whose choice value was never produced during phase evaluation is an internal hard error at specialization time (defensive; unreachable from definite-return-checked source).
- After processing a boundary (both directions), no content at phases > p remains inside phase-p CF and none at phases < p inside phase-p regions; phase splitting proper reduces to distributing straight-line ops and threading straight-line samples (context).

Worked example — content two phases away (`for const` at -1, var at 0, `dyn` block at +1):

```
hits0 = []
for i in 0..n { hits0.append(i) }
---
hits1 = []
replicate i in hits0 {
  var x; x = y + i; x = x * 2;
  hits1.append(x)        // sample of x at phase 0, inside the replicate
}
---
replicate x in hits1 { emit(x) }
```

Worked example — partial activation (continue at -1 between two dyn statements):

```
// for const i in 0..4 { x = x + 1; const { if i == 2 { continue; } } x = x * 2; }
hits = []
for i in 0..4 {
  if i == 2 { hits.append(choice_exited());  continue }
  hits.append(choice_continued())
}
---
replicate exit in hits {
  x = x + 1                                   // A: every activation
  pick exit [ continued(): { x = x * 2 }      // B: only full activations
              exited():    { } ]
}
// A0 B0  A1 B1  A2  A3 B3
```

Worked example — loop result at a later phase (break value choice-ified):

```
// let y = const { var k = 0; loop { if k*k >= n { break dyn { x + k }; } k = k + 1; } };
k = 0
choice = loop {
  if k * k >= n { break choice0(sample k) }   // tag + payload record
  k = k + 1
}
---
y = pick choice [ 0(k): { x + k } ]
```

## 7. Value selection across phases (floor rule)

A CF op at phase p that selects between values cannot deliver the selection at a phase earlier than p (causality):

- A result provided by more than one terminator site (`return`/`break`/yield) is a **selection**, regardless of the values' identity — no SSA-identity exemption and no semantic-equality analysis; an identity exemption would not compose anyway (nested ifs providing the same value fail SSA identity, so it would have to grow into a CFG traversal).
- The result phase of a selection floors at the op phase: earlier-phase values provided by multiple sites deliver at the op phase, and a demand below the op phase is an **error**.
- In particular, multiple `return` sites providing a phase<0 (const) result are always an error, even when all sites return the same SSA value. A future surface rule may require a single return for functions with phase<0 results.
- The check runs pre-split, in PhaseAnalysis (the implemented floor rule already matches this).
- A CF op at phase p yielding a **later**-phase value is legal: the value is replaced by a choice crossing the boundary; the receiving phase reconstructs the selection with `pick` (§6).
- Same-phase CF is untouched: `if`/`loop` keep SSA results and value-carrying yields when everything is at one phase. Splitting must decay gracefully — single-phase functions pass through the same general algorithm unchanged, never a special case.

## 8. Types

Types are ordinary values computed one phase before the values they describe.
Nothing distinguishes a type crossing a boundary from a value crossing it: per-activation types ride in records like any snapshot (`uint<i+1>` in a `for const` is a record field).
HIR-to-MIR resolves each SSA value in type position to its defining ConstantLike op (internal compiler error otherwise) and materializes the static MLIR type.

## 9. Calls

- A cross-phase call decomposes into per-phase calls. The callee's earlier-phase halves are ordinary ops at the caller's corresponding phase, at the call site's program-order position. Compile-time recursion is plain interpretation of those halves.
- The callee's inter-phase context values are ordinary samples; a call site inside producing-phase CF puts its context into the activation record like everything else.
- Later-phase halves take context values as **specialization parameters**: `helper@0(x; c)` becomes a call to a specialized clone `helper@0<c>(x)` once `c` is a known constant.
- Specialization key: (callee, constant values, **phase offset**) — the same callee invoked at different phase depths does not share specializations.

## 10. Phase evaluation loop

Executing one phase p, then **specialization**:
substitute phase-p results as constants; expand every `replicate`/`pick` whose list/choice is now constant; specialize called functions per record; delete phase-p code and phase-p function arguments. This reveals phase p+1 as a plain program. Repeat until only the final (hardware) phase remains.

## 11. References are opaque; home phases are evaluation-time data

Settled in the ref-opacity interview (2026-07-09); replaces the earlier parking-lot stance that references need their own design before use.

- A reference is an ordinary, **opaque** SSA value. Its **value phase** p obeys all the usual value rules: it floats, freezes into later phases, and floor-bumps at CF selections. Decoupled from p, every reference targets storage with a **home phase** — but home is *runtime data*, not a static property: at evaluation time a reference is a `(cell, home)` pair.
- **Phase analysis knows nothing about home.** `hir.var`/`hir.read_var`/`hir.write_var` are plain side-effecting ops anchored at their block phase; the only static rule is ordinary causality on the reference operand (a reference cannot be consumed before it exists). Reads and writes must never require tracing a reference back to its `hir.var`: a function taking a list of references as arguments must be indistinguishable from locally declared vars.
- Home checks are **evaluation-time checks** (a deliberate stopgap; more static checking may be layered on later): write legal iff the executing phase equals the cell's home; read legal iff executing phase ≥ home; deref of a not-yet-materialized cell is an eval error. Diagnostics are consequently lazy — only executed paths error.
- `&x` of a *later*-phase value is an **explicit future**: p < home is constructible. All (p, home) combinations are meaningful; deref waits for home. This is the sanctioned successor of the transparent-yield futures the model killed.
- **Home cannot live in types.** Types are SSA values that only become known as phases execute; a type statically carrying the phase of its contents is circular. (Parked: a hidden "reference phase" type parameter, Rust-lifetime-shaped, if static checking of boundary-crossing mutable references is ever wanted.)
- CF selection of references is legal and type-blind. A cross-phase floor-bump naturally demotes the result to read-only: a later write finds the cell frozen at eval. Same-phase selection keeps writability — the selection resolves during that phase's evaluation. This enables the routing-table pattern (e.g. an array of references indexed by a phase-local value).
- Aggregates containing references are evaluation values like any other; contained references carry home at runtime. Per-phase SROA-style destructuring (structs/tuples, function args) is a normal form/optimization where indices are structural — dynamic indexing (routing tables) makes it impossible in general, so it is *not* the semantic foundation.
- The hardware phase is never interpreted, so its writes get no eval check — but by then every surviving reference is a specialization-time constant: writes to frozen cells are caught when the second-to-last phase specializes, and what remains are writes to hardware-phase cells, i.e. drives, handled by MIRToCIRCT. A writable reference to a dyn value is an output port; probes/drives and compile-time vars are one mechanism instantiated at different phases.

## Parking lot (deliberately deferred)

- Futures in aggregates (struct field at a later phase); keep aggregate constituents at one phase for now.
- Static home-phase checking for references (§11 defers all home checks to evaluation): mutable/immutable reference split, hidden "reference phase" type parameter, static errors for premature future derefs.
- Break/return "may not pass through const/dyn" restriction (would reject the surprising D-rule break, §4).
- Diagnostics/warnings for surprising-but-legal D-rule programs.
- Surface syntax: var home-phase declaration (`var const k`), `let mut` migration, assignment and loop syntax (none exist in the language yet).
- Whether `uir.loop` keeps loop-carried SSA values for same-phase loops or codegen routes all loop state through vars + later mem2reg.
- Nontermination of compile-time loops/recursion: compiler hangs like any program; no limit (per control-flow.md philosophy).
