- Investigate why `hir.unify` ops survive past InferTypes/canonicalization/CSE all the way to HIR-to-MIR lowering. Ideally they should all be erased by that point, and HIRToMIR should refuse to lower them rather than silently forwarding an operand.

- Check if the Interpret pass runs entirely on the MIR (or the Common) dialect, and if it does, move it into the MIR dialect.
- Check if the SpecializeFuncs pass runs entirely on the HIR (or the Common) dialect, and if it does, move it into the HIR dialect.

- The assembly printer for `mir.evaluated_func` inserts a space after the symbol visibility, e.g. `mir.evaluated_func private  @main.0b`.
  There should only be one space: `mir.evaluated_func private @main.0b`.
  Other ops like `mir.func` already do this correctly.

- The bitwise and multi-phase end-to-end tests are very brittle. Instead of making these end-to-end tests, see if the before/after IR they produce as the pipeline executes can be added to the corresponding pass' lit tests.
  Doing so would make for much more focused testing.

- Allow the user to declare `pub fn` in the input language, which translate to public hir.funcs; all others should be private.
