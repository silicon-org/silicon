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
- Investigate why the `eraseVoidCalls` is needed in function specialization; is this a hack?

- **Implicit type widening (`uint<8>` to `uint<16>`) not supported.**
  Input: `pub fn widen(a: uint<8>) -> uint<16> { a }` — error: `hir.unify survived to HIR-to-MIR lowering`.
  There's no implicit widening conversion.
  Low priority; explicit cast syntax would be the right fix, but that syntax doesn't exist yet either.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
