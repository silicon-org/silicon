- Investigate why `hir.unify` ops survive past InferTypes/canonicalization/CSE all the way to HIR-to-MIR lowering. Ideally they should all be erased by that point, and HIRToMIR should refuse to lower them rather than silently forwarding an operand.

- The bitwise and multi-phase end-to-end tests are very brittle. Instead of making these end-to-end tests, see if the before/after IR they produce as the pipeline executes can be added to the corresponding pass' lit tests.
  Doing so would make for much more focused testing.
