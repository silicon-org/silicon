# TODO

- If there is no difference between hir.signature and hir.unified_signature except for the parent op they accept, combine them into a single hir.signature op.
- Analyze the usage of ConstBrOp and ConstCondBrOp
  I'm pretty sure these are leftovers from an earlier prototype pass and are not needed for the current implementation.
  Try removing them and replacing any potential leftover uses with BrOp and CondBrOp from CF.
  You might have to get rid of the eval consts pass, too, which I believe we're not running in the silc pipeline anyway.
- Add a `bool` type to the language and use it for comparison results.
  If conditions should have their type unified with bool, to nudge type inference.
  We'll need `true` and `false` constants in the language, too, alongside logic `&&` and `||` operators.

- **Parenthesized expressions not parseable.**
  Input: `fn main() -> int { (2 + 3) * 4 }` — error: `expected expression, found (`.
  This is extremely basic syntax that every user will try.
  The parser likely doesn't have a case for `(` as the start of a primary expression; needs a grouping/paren rule.

- **`if` with comparison condition fails in hardware output.**
  Input: `pub fn max(a: int, b: int) -> int { if a > b { a } else { b } }` — error: `failed to legalize unresolved materialization from ('i1') to ('i64')`.
  Also: `pub fn select(cond: int, a: int, b: int) -> int { if cond { a } else { b } }` — error: `comb.mux operand #0 must be 1-bit signless integer, but got 'i64'`.
  The comparison produces `i1` but `comb.mux` gets wired with `i64`; and non-boolean conditions aren't narrowed to `i1`.
  MIRToCIRCT needs to insert a width cast (trunc or icmp-ne-zero) between the condition and the mux.

- **Negative number literals don't work.**
  Input: `fn main() -> int { -1 }` — error: `compiler bug: unsupported expression kind UnaryExpr`.
  Parsed as unary negation of `1`, which hits the unimplemented unary expr path.
  Fix by implementing `UnaryExpr` in codegen (see next item), or by folding `-<literal>` into a negative literal in the parser.

- **Unary operators (`-x`, `!x`) not implemented in codegen.**
  Input: `fn main() -> int { let x = 5; -x }` — error: `compiler bug: unsupported expression kind UnaryExpr`.
  The parser accepts these but codegen has no case for `UnaryExpr`.
  Negation can lower to `0 - x` (or `hir.sub`), bitwise NOT to `hir.xor x, -1` or a dedicated op.

- **`dyn fn` crashes the compiler with an assertion failure.**
  Input: `dyn fn late(a: int) -> int { a + 1 }; fn main() -> int { late(5) }` — `SmallVector operator[] out of bounds` in PhaseSplit.
  The compiler should never crash on user input.
  SplitPhases likely doesn't handle the case where all phases are shifted later (no phase-0 split exists).

- **Trailing semicolon on return expression gives internal error instead of user-facing diagnostic.**
  Input: `fn main() -> int { let x = 1; x; }` — error: `compiler bug: hir.unify survived to HIR-to-MIR lowering`.
  The trailing `;` makes the block return unit, which conflicts with the declared `int` return type.
  InferTypes can't unify unit with int, and the mismatch leaks as an internal error.
  Should produce a user-facing error like "expected return type `int`, found unit" before lowering.

- **Omitting return type on a function that returns a value gives internal error.**
  Input: `fn main() { 42 }` — error: `compiler bug: hir.unify survived to HIR-to-MIR lowering`.
  Same underlying issue as the trailing-semicolon bug: the declared return type (implicit unit) doesn't match the expression type (int).
  A type-checking pass should catch this and emit a proper diagnostic.

- **Comparison result can't be used as a return value.**
  Input: `pub fn gt(a: int, b: int) -> int { a > b }` — error: `failed to legalize unresolved materialization from ('i1') to ('i64')`.
  Comparisons produce `i1` but the return type is `i64`.
  Related to the `if`-with-comparison bug; needs implicit zero-extension or a bool type.

- **`dyn` on function arguments gives phase mismatch error.**
  Input: `fn send(dyn x: int, y: int) -> int { x + y }` — error: `return value is available at phase 1 but function declares phase 0 return`.
  The `dyn` argument shifts `x` to a later phase, but the return type stays at phase 0.
  This is arguably correct behavior (the user should write `-> dyn int`), but the error message is not helpful for a new user.
  At minimum, improve the diagnostic to suggest adding `dyn` to the return type.

- **`dyn` return type causes compiler bug.**
  Input: `fn make_dyn(x: int) -> dyn int { x }; fn main() -> int { make_dyn(42) }` — error: `compiler bug: op uses value from later phase`.
  The `dyn` return shifts the result to a later phase, and the caller at phase 0 can't use it.
  May be a legitimate phase violation, but should be a user-facing error, not a compiler bug.

- **`const const fn` (double phase shift) fails.**
  Input: `const const fn very_early(a: int) -> int { a }; fn main() -> int { very_early(5) }` — error: `callee @very_early.1a is not a mir.func (may not have been lowered yet)`.
  The double const shift moves the function to phase -2, but the pipeline only runs enough iterations to handle phase -1.
  Low priority since this is an unusual construct; may just need more PhaseEvalLoop iterations or a check.

- **Nested block comments don't work.**
  Input: `/* outer /* inner */ still outer */` — error: `expected item, found identifier 'still'`.
  The docs say block comments are "nestable" but the lexer doesn't track nesting depth.
  Follow what Rust does: if it supports nested block comments, let's support them too.
  Otherwise, let's fix the docs and indicate that these cannot nest.

- **Recursion gives a terse error.**
  Input: `fn factorial(n: int) -> int { if n == 0 { 1 } else { n * factorial(n - 1) } }` — error: `recursive call cycle detected`.
  This is correct (hardware can't have unbounded recursion), but the error doesn't explain _why_ recursion is rejected.
  A note like "recursive functions cannot be synthesized to hardware" would help new users.

- **Implicit type widening (`uint<8>` to `uint<16>`) not supported.**
  Input: `pub fn widen(a: uint<8>) -> uint<16> { a }` — error: `hir.unify survived to HIR-to-MIR lowering`.
  There's no implicit widening conversion.
  Low priority; explicit cast syntax would be the right fix, but that syntax doesn't exist yet either.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
