---
name: design-review
description: Adversarial review of changes, a commit, or the whole codebase for design-discipline violations — carve-outs, test-shape pattern matching, rules that don't generalize. Use after implementing a feature or pass, before committing nontrivial changes, or when the user asks whether the code has drifted into patching.
---

Review code for violations of this repo's design discipline: rules that don't generalize, carve-outs that exist to make specific tests pass, and code that absorbs design flaws instead of surfacing them.
The review runs in the `design-reviewer` subagent, which pins the model and carries the reviewer instructions; this skill only picks the scope, spawns it, and acts on the verdict.
Never do the review in the main session, and do not skip it because you are confident the code is clean — fading attention is exactly the failure mode this skill exists to catch.

## Determine the Scope

From the arguments (or the conversation), pick one:

1. No arguments: the uncommitted changes (staged and working tree); if the tree is clean, the HEAD commit.
2. A commit ref: that commit's diff.
3. A pass, dialect, or path: the current state of that code, regardless of when it was written.
4. `all` or `codebase`: the whole compiler; spawn one `design-reviewer` per transformation/conversion pass and one per dialect, then merge their reports.

## Spawn the Reviewer

Spawn the `design-reviewer` agent, one per scope unit (for `codebase`, one per pass and per dialect).
Its prompt must contain only the scope — the diff ref, commit sha, or paths.
Do not include your own reasoning, the design goals you were pursuing, why the code looks the way it does, or any hint of what you expect the verdict to be; the reviewer must judge the code free of this session's justifications.

## Act on the Verdict

Relay the reviewer's findings to the user verbatim and honestly, including verdicts against code written in this session.

1. BUG findings: fix them; the rule is right and the code is wrong.
2. PATCH and DESIGN-FLAW findings: these are stop signals, not fix lists. Do not rework the code to satisfy the reviewer, do not make the carve-out more elaborate or better hidden, and do not absorb the problem silently. Add a TODO.md entry per finding, then start a terse design interview with the user, one concrete question at a time.
3. If you believe a PATCH or DESIGN-FLAW finding is a false alarm, say so to the user with your reasoning and let them decide; do not overrule the reviewer on your own.

Additional instructions provided by the user: $ARGUMENTS
