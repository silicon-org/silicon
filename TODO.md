# TODO

- SplitPhases "op uses value from later phase" and region isolation errors on `dyn` args and `const`/`dyn` return types; update docs silicon code blocks once fixed:
  - `fn send(dyn x: int, y: int) -> int { x + y }` — dyn arg with non-dyn return
  - `fn make_const(x: int) -> const int { x }` — const return type
  - `fn make_dyn(x: int) -> dyn int { x }` — dyn return type
- It looks like symbol DCE does not run: even with a simple example, the output still contains private split funcs, even if there are no more calls to them
- Make split func discardable by removing that `canDiscardOnUseEmpty` override
- The input `mod hello(a: uint<42>) {}` to silc triggers the error `error: 'hir.uint_type' op using value defined outside the region`

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
