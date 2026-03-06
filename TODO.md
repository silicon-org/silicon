- `dependent-type.mlir` test segfaults — the phased evaluation pipeline crashes when processing functions with dependent types (`identity(%T: -1, %x: 0)` where the result type depends on the first argument).

- SplitPhases generates wrong structure for functions with all-external const phases.
  For `fn compute(const const a: int, const b: int, c: int)` (phases -2, -1, 0 all externally visible), the current code produces `multiphase_func @compute.const(first a, last b, last c)` with three sub-functions.
  This is wrong: a `multiphase_func` is for eagerly evaluating internal const phases that require no input arguments.
  Since all three phases are externally visible, the `split_func` should have three separate entries (`-2: @compute.const2, -1: @compute.const1, 0: @compute.const0`), each accepting its own arg plus whatever opaque context is needed.
  A `multiphase_func` should only be created when there are internal const phases (phases that the caller doesn't directly provide arguments for).

- Multiphase func "first" args should only be opaque context, never user-visible function arguments.
  The "first" annotation currently gets applied to the earliest-phase arg of the multiphase_func, but this is wrong when that arg is an actual input to the original function.
  The only thing that should be "first" is an opaque context flowing into the multiphase_func from a split_func (e.g., when the multiphase_func is itself a sub-function of a split_func).
  Real input arguments should either be on a dedicated sub-function in the split_func, or be "last" args on the final sub-function of a multiphase_func.
  The purpose of multiphase_func is to eagerly evaluate compile-time phases that need no external input; any real input arg belongs on its own dedicated func.

- Create a `Common` dialect (`si.` prefix) for types and attributes shared between HIR and MIR.
  Now that HIR needs to reason about MIR attributes and types (e.g., `hir.mir_constant` materializing `#mir.opaque` values), the current split is awkward — HIR has to depend on MIR just for attribute types.
  A `Common` dialect would hold the shared types (`!si.int`, `!si.uint<N>`, `!si.unit`, `!si.type`, `!si.opaque`, etc.) and attributes (`#si.int<42>`, `#si.type<!si.int>`, `#si.opaque<[...]>`, etc.).
  This enables replacing dedicated HIR type constructor ops with a single `hir.constant` op:
  - `hir.int_type` → `hir.constant #si.type<!si.int>`
  - `hir.type_type` → `hir.constant #si.type<!si.type>`
  - `hir.uint_type %width` (when width is constant) → `hir.constant #si.type<!si.uint<42>>`
    The same `hir.constant` can materialize `#si.opaque` attributes as HIR values (with type `!hir.any`), replacing the current `hir.mir_constant` op.
    MIR would keep its `mir.constant` but use the common types/attrs (`mir.constant #si.int<42> : !si.int`).

- Revise how SplitPhases creates multiphase funcs and groups phases.
  The pass currently compares phases against hardcoded constants like `minPhase < 0` or `minPhase <= -2`, which looks like it encodes a specific pattern from a few examples rather than implementing a general algorithm.
  The correct general process is:
  1. Find the min and max phase from argument/result phases and body phase analysis.
  2. Identify externally visible phases (the sparse subset of the min..max range that have caller-provided arguments or caller-visible results).
  3. Create a distinct entry in the split_func for each externally visible phase.
  4. Each entry is either a plain `hir.func` (if it covers a single phase) or an `hir.multiphase_func` that aggregates the immediately preceding internal phases that are not externally visible.
     Put differently: all internal (non-externally-visible) phases get merged into the next externally visible phase via a multiphase_func.
     The remaining list of entries (externally visible phases, plus possibly a trailing multiphase_func for internal tail phases) are collected into the split_func.
     The pass should support more than one multiphase_func per split_func if needed, and should not reason about absolute phase numbers beyond ordering.
     This needs rigorous testing and may require updating the phase splits design doc (`docs/design/phase-splits.md`) with the general algorithm.

- Prefer C++ templates over macros like `CONVERT_BINARY_OP`
