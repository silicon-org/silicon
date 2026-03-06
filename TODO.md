- `dependent-type.mlir` test segfaults — the phased evaluation pipeline crashes when processing functions with dependent types (`identity(%T: -1, %x: 0)` where the result type depends on the first argument).

- Create a `Common` dialect (`si.` prefix) for types and attributes shared between HIR and MIR.
  Now that HIR needs to reason about MIR attributes and types (e.g., `hir.mir_constant` materializing `#mir.opaque` values), the current split is awkward — HIR has to depend on MIR just for attribute types.
  A `Common` dialect would hold the shared types (`!si.int`, `!si.uint<N>`, `!si.unit`, `!si.type`, `!si.opaque`, etc.) and attributes (`#si.int<42>`, `#si.type<!si.int>`, `#si.opaque<[...]>`, etc.).
  This enables replacing dedicated HIR type constructor ops with a single `hir.constant` op:
  - `hir.int_type` → `hir.constant #si.type<!si.int>`
  - `hir.type_type` → `hir.constant #si.type<!si.type>`
  - `hir.uint_type %width` (when width is constant) → `hir.constant #si.type<!si.uint<42>>`
    The same `hir.constant` can materialize `#si.opaque` attributes as HIR values (with type `!hir.any`), replacing the current `hir.mir_constant` op.
    MIR would keep its `mir.constant` but use the common types/attrs (`mir.constant #si.int<42> : !si.int`).

- Prefer C++ templates over macros like `CONVERT_BINARY_OP`
- Check if the Interpret pass runs entirely on the MIR (or the Common) dialect, and if it does, move it into the MIR dialect.
- Check if the SpecializeFuncs pass runs entirely on the HIR (or the Common) dialect, and if it does, move it into the HIR dialect.

- The assembly printer for `mir.evaluated_func` inserts a space after the symbol visibility, e.g. `mir.evaluated_func private  @main.0b`.
  There should only be one space: `mir.evaluated_func private @main.0b`.
  Other ops like `mir.func` already do this correctly.

- The bitwise and multi-phase end-to-end tests are very brittle. Instead of making these end-to-end tests, see if the before/after IR they produce as the pipeline executes can be added to the corresponding pass' lit tests.
  Doing so would make for much more focused testing.

- Allow the user to declare `pub fn` in the input language, which translate to public hir.funcs; all others should be private.
