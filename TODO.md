# TODO

- A recent refactor of CheckCalls has made the pass inline a copy of the signature into a unified func's body.
  Do the same for call sites: inline a copy of the signature just ahead of the call (splitting the block before the call), and do appropriate inference and unification.
- If there is no difference between hir.signature and hir.unified_signature except for the parent op they accept, combine them into a single hir.signature op.
- Analyze the usage of ConstBrOp and ConstCondBrOp
  I'm pretty sure these are leftovers from an earlier prototype pass and are not needed for the current implementation.
  Try removing them and replacing any potential leftover uses with BrOp and CondBrOp from CF.
  You might have to get rid of the eval consts pass, too, which I believe we're not running in the silc pipeline anyway.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
