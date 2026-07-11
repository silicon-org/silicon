---
name: design-review
description: Adversarial review of changes, a commit, or the whole codebase for design-discipline violations — carve-outs, test-shape pattern matching, rules that don't generalize. Use after implementing a feature or pass, before committing nontrivial changes, or when the user asks whether the code has drifted into patching.
---

Review code for violations of this repo's design discipline: rules that don't generalize, carve-outs that exist to make specific tests pass, and code that absorbs design flaws instead of surfacing them.
The review must run in a subagent with a fresh context: the reviewer must judge the code on its own, free of this session's justifications and attachment to its own work.
Do not review in the main session, and do not skip the review because you are confident the code is clean — fading attention is exactly the failure mode this skill exists to catch.

## Determine the Scope

From the arguments (or the conversation), pick one:

1. No arguments: the uncommitted changes (staged and working tree); if the tree is clean, the HEAD commit.
2. A commit ref: that commit's diff.
3. A pass, dialect, or path: the current state of that code, regardless of when it was written.
4. `all` or `codebase`: the whole compiler; spawn one reviewer subagent per transformation/conversion pass and one per dialect, then merge their reports.

## Spawn the Reviewer

Spawn a subagent whose prompt contains only the scope (diff ref, commit sha, or paths) and the reviewer instructions below, verbatim.
Do not include your own reasoning, the design goals you were pursuing, why the code looks the way it does, or any hint of what you expect the verdict to be.
If the scope is a diff, tell the reviewer the ref/sha and let it run git itself.

Reviewer instructions:

> You are performing an adversarial design review of the Silicon compiler, an MLIR-based hardware language compiler.
>
> Ground rules of this repo: it is a compiler, so every transform must be built of small, orthogonal rewrite rules that hold for ANY op of their kind in ANY context.
> Design docs (docs/design/), tests, and existing implementations are provisional hypotheses, not requirements — they may themselves be wrong, incomplete, or mutually inconsistent.
> Your question is NOT "does the code match the spec" or "do the tests pass"; it is "does the code implement general rules, or does it pattern-match the shapes that happen to appear in the tests and doc examples".
>
> Hunt for these violations:
> - Special-casing on op identity, position, or surrounding structure where the rule should apply uniformly.
> - Conditionals or early-outs whose only purpose is to make a specific test or doc example work.
> - Handling the example shapes instead of the general construct: code that works when a value is defined locally but breaks when the same value arrives as a function argument, out of an aggregate, or across a phase/region boundary.
> - Extra context threaded into a rule that should be local.
> - Def-chain chasing to recover a fact ("trace through the IR to find the real X") — the fact wants to live on the value, in a type, or in runtime data instead.
> - Silent skips, null-tolerance, or fallbacks where the compiler should assert or emit a compiler bug error.
> - Workarounds for "impossible" IR that the design says cannot exist.
> - Tests that encode a carve-out rather than exercise a general rule.
>
> Method, for each rule or code path in scope:
> 1. State the rule it implements in one line with no reference to a specific IR context, plus the invariant it preserves. If you cannot, that is a finding.
> 2. Construct at least one input structurally unlike the existing tests — different nesting, values arriving opaquely, multiple interacting instances, reordered or interleaved ops — and check the rule against it: by hand, or preferably by writing the input to tmp/ and running build/bin/silicon-opt on it.
> 3. Classify the result:
>    - GENERALIZES: the rule holds; no finding.
>    - BUG: the rule is right, the code is wrong.
>    - PATCH: the code works for the shapes at hand but fails for the same construct elsewhere.
>    - DESIGN-FLAW: the rule cannot hold given the other rules; no implementation could be general.
>
> You are review-only: do not modify code, tests, or docs; writing throwaway test inputs under tmp/ is fine.
>
> Report one entry per finding with file:line, the classification, the one-line rule, and the concrete breaking input if you constructed one.
> End with an overall verdict of CLEAN or STOP and a one-paragraph justification.
> Err on the side of STOP: a false alarm costs a short discussion, a missed carve-out costs a redesign later.

## Act on the Verdict

Relay the reviewer's findings to the user verbatim and honestly, including verdicts against code written in this session.

1. BUG findings: fix them; the rule is right and the code is wrong.
2. PATCH and DESIGN-FLAW findings: these are stop signals, not fix lists. Do not rework the code to satisfy the reviewer, do not make the carve-out more elaborate or better hidden, and do not absorb the problem silently. Add a TODO.md entry per finding, then start a terse design interview with the user, one concrete question at a time.
3. If you believe a PATCH or DESIGN-FLAW finding is a false alarm, say so to the user with your reasoning and let them decide; do not overrule the reviewer on your own.

Additional instructions provided by the user: $ARGUMENTS
