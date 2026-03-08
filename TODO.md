# TODO

- SplitPhases "op uses value from later phase" and region isolation errors on `dyn` args and `const`/`dyn` return types; update docs silicon code blocks once fixed:
  - `fn send(dyn x: int, y: int) -> int { x + y }` — dyn arg with non-dyn return
  - `fn make_const(x: int) -> const int { x }` — const return type
  - `fn make_dyn(x: int) -> dyn int { x }` — dyn return type
- Double check that the `hir.return` op's verifier ensures that it has the same number of arg/result type operands as the parent function has args/results

- Re-enable SymbolDCE in `silc` pipeline and `PhaseEvalLoop` once proper entry-point / public visibility semantics are in place; currently all user functions are private, so SymbolDCE removes everything

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
