# TODO

- In CheckCalls, don't try to selectively clone ops or be too picky.
  Instead, make sure that there is a single hir.signature terminator by adding an exit block and branching there if there are multiple such terminators.
  Then simply clone the blocks of the signature region (there can be multiple ones, test that this is supported) and place them at the beginning of the function's body block.
  Make sure to remove the block args from the body block and replace them with the block args of the cloned signature entry block.
  Then use the signature terminator's operands to feed into the func body's terminator type operands, and for coerce_type ops inserted after the block arguments.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
