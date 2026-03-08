# TODO

- Rename `AnyHIRType` to `HIRAnyType`
- Add a type operand to `hir.constant_int`, such that we can infer the type to be `int` or `uint<N>`.
  When we infer the concrete type of such a constant, also check that the integer fits into the chosen type.
  We may want to do this as part of type inference, since this is a user-facing error.
- Investigate why the `eraseVoidCalls` is needed in function specialization; is this a hack?
- Implement CFG-to-dataflow conversion (phi→mux) in MIRToCIRCT.
- **`if` conditions accept any type without a type error.**
  `if cond { ... }` where `cond: int` silently works because `coerce_to_i1` doesn't enforce a bool type.
  The old `hir.if` also didn't check, so this isn't a regression, but we should add a proper check.
  Codegen should unify the condition with `bool` and emit a type error for non-bool conditions; no implicit truthiness.
- **`getTypeOf` returns null for block arguments.**
  After the IfOp removal, block arguments at CFG merge points have no type metadata.
  `getOrCreateTypeOf` falls back to inserting a `TypeOfOp`, which works but is less efficient than directly resolving the type.
  Consider teaching `getTypeOf` to look at predecessor branch operands, with a dominance check to avoid returning values from non-dominating blocks.

- **Implicit type widening (`uint<8>` to `uint<16>`) not supported.**
  Input: `pub fn widen(a: uint<8>) -> uint<16> { a }` — error: `hir.unify survived to HIR-to-MIR lowering`.
  There's no implicit widening conversion.
  Low priority; explicit cast syntax would be the right fix, but that syntax doesn't exist yet either.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
