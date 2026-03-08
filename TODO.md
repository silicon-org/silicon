# TODO

- Currently hir.func encode their argument types through the hir.return function, which is exactly what we want.
  However, it is the CheckCalls pass on unified funcs that copies the signature into the body.
  This is where we get the type information for block arguments.
  But unified returns don't have a mechanism to describe func arg types, which means CheckCalls cannot thread the argument types through to the return such that they later land in the hir.return.
  So we lose the information about the block argument types.
  Come up with a fix for this; maybe hir.unified_return should have the same func arg type operands as hir.return, which are set to inferrable by codegen, together with a coerce type on the block args with the same inferrable?
  Maybe unified funcs could use the same hir.return that regular funcs later use?
- In CheckCalls, don't try to selectively clone ops or be too picky.
  Instead, make sure that there is a single hir.signature terminator by adding an exit block and branching there if there are multiple such terminators.
  Then simply clone the blocks of the signature region (there can be multiple ones, test that this is supported) and place them at the beginning of the function's body block.
  Make sure to remove the block args from the body block and replace them with the block args of the cloned signature entry block.
  Then use the signature terminator's operands to feed into the func body's terminator type operands, and for coerce_type ops inserted after the block arguments.

## Postponed Long-Term Fixes

- MIRToCIRCT: `!si.int` is temporarily mapped to `i64`; once bitwidth inference exists, this should be an error diagnostic instead
