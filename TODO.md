# TODO

- Consider replacing `hir.unified_return` with `hir.return` in case they are mostly identical
- A recent refactor of CheckCalls has made the pass inline a copy of the signature into a unified func's body.
  Do the same for call sites: inline a copy of the signature just ahead of the call (splitting the block before the call), and do appropriate inference and unification.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
