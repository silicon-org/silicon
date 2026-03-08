---
name: review-dialects
description: Reviews the MLIR dialects and ensures that they are properly tested and documented. Use when the user asks to review dialects, ops, or passes
---

Got through the MLIR dialects and transformation passes defined in Silicon and ensure they meet our quality standards:

1. Documentation
   1. Every operation/pass must have a very concise summary that describes the gist of what it does
   2. Every operation/pass must have a detailed description of how it works and examples of how to use it.
      For ops this also includes type constraints, and a detailed description of how the results are computed.
      For passes this should include worked examples and indications which operations are affected and transformed, unless the pass works on an entire dialect.
   3. If a pass has options, they should adequately document what they do.
      Keep in mind that their summary is printed in the CLI tool help, so it should be brief to not spill onto new lines.
2. Types
   1. Operation operands and results should have appropriate types.
      If an op requires specific types or sets of types, don't use an MLIR `AnyType`; instead, actually define what the op needs.
   2. Use constraints to express operands that have the same type, or results that have the same type as operands.
      Good tools are the types-match-with, same-operand-types, same-operand-and-result-types, and C predicate constraints.
      If the semantics of an op require operands and results to follow certain rules, these must be encoded as constraints or caught by the verifier.
   3. Prefer constraints over hand-writing things in the verifier.
3. Ops must have canonicalizers and folders where appropriate
4. Traits
   1. Mark ops `Pure` if they have no side-effects and should be DCE-able and CSE-able
   2. Mark operands `MemRead` and `MemWrite` if the op has read or write semantics, e.g. reading or assigning variables
   3. Mark results `MemAlloc` if the op has allocation semantics, e.g. creating a variable
   4. Ops that have body regions, consider making them have recursive memory effects if the op itself has no side-effects, but ops in the body have.
5. Testing
   1. Every op must have a parsing/printing roundtrip test in the corresponding basic.mlir lit test.
      Every combination of optional/variadic parts in the syntax must be tested.
   2. Every error that the op verifier/constraints can produce must be tested in the corresponding errors.mlir lit test.
   3. Passes must be extremely thoroughly tested, since this is a compiler.
      Every code path in the pass must be triggered in some form by the pass' lit test.
      Every error message that the pass can produce must be tested by the pass' error lit test.
      All corner cases of the pass must be covered.
      If you see potential to trigger assertions, out-of-bounds accesses, and other crashes in the pass, try to exercise them by taking on an adversarial stance and coming up with test cases that break things.
   4. Passes must be able to operate on any valid IR; they cannot simply crash.
      If a pass requires the IR to be in a certain state, it must check for that state and print a proper compiler bug error message and fail otherwise.
      Alternatively, the IR constraints can be tightened by extending op verifiers to impose restrictions that the pass needs, as long as they are valid for all other passes.

Spin up subagents for each dialect's ops, and for each conversion, transformation, or dialect pass in the code base.
The subagent should check against the above list of criteria.
If necessary, the subagent should call `build/bin/silc`, `build/bin/silicon-opt`, or `build/bin/llvm-lit` to run certain tests or exercise individual passes.
Collect any issues found by the subagents and create items in TODO.md for each.
Including any code snippets needed to trigger trigger the issue.

Additional instructions provided by the user: $ARGUMENTS
