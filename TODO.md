# TODO

- Consider replacing `hir.unified_return` with `hir.return` in case they are mostly identical

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
