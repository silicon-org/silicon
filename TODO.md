# TODO

- If there is no difference between hir.signature and hir.unified_signature except for the parent op they accept, combine them into a single hir.signature op.
- Analyze the usage of ConstBrOp and ConstCondBrOp
  I'm pretty sure these are leftovers from an earlier prototype pass and are not needed for the current implementation.
  Try removing them and replacing any potential leftover uses with BrOp and CondBrOp from CF.
  You might have to get rid of the eval consts pass, too, which I believe we're not running in the silc pipeline anyway.
- Add a `bool` type to the language and use it for comparison results.
  If conditions should have their type unified with bool, to nudge type inference.
  We'll need `true` and `false` constants in the language, too, alongside logic `&&` and `||` operators.
- Add a type operand to `hir.constant_int`, such that we can infer the type to be `int` or `uint<N>`.
  When we infer the concrete type of such a constant, also check that the integer fits into the chosen type.
  We may want to do this as part of type inference, since this is a user-facing error.

- **`if` with comparison condition fails in hardware output.**
  Input: `pub fn max(a: int, b: int) -> int { if a > b { a } else { b } }` — error: `failed to legalize unresolved materialization from ('i1') to ('i64')`.
  Also: `pub fn select(cond: int, a: int, b: int) -> int { if cond { a } else { b } }` — error: `comb.mux operand #0 must be 1-bit signless integer, but got 'i64'`.
  The comparison produces `i1` but `comb.mux` gets wired with `i64`; and non-boolean conditions aren't narrowed to `i1`.
  MIRToCIRCT needs to insert a width cast (trunc or icmp-ne-zero) between the condition and the mux.

- **Comparison result can't be used as a return value.**
  Input: `pub fn gt(a: int, b: int) -> int { a > b }` — error: `failed to legalize unresolved materialization from ('i1') to ('i64')`.
  Comparisons produce `i1` but the return type is `i64`.
  Related to the `if`-with-comparison bug; needs implicit zero-extension or a bool type.

- **`const const fn` (double phase shift) fails.**
  Input: `const const fn very_early(a: int) -> int { a }; fn main() -> int { very_early(5) }` — error: `callee @very_early.1a is not a mir.func (may not have been lowered yet)`.
  The double const shift moves the function to phase -2, but the pipeline only runs enough iterations to handle phase -1.
  Low priority since this is an unusual construct; may just need more PhaseEvalLoop iterations or a check.

- **Implicit type widening (`uint<8>` to `uint<16>`) not supported.**
  Input: `pub fn widen(a: uint<8>) -> uint<16> { a }` — error: `hir.unify survived to HIR-to-MIR lowering`.
  There's no implicit widening conversion.
  Low priority; explicit cast syntax would be the right fix, but that syntax doesn't exist yet either.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
