- Resolve `hir.type_of` and `hir.unify` before SplitPhases runs.
  Codegen emits `hir.return(%val) : (hir.type_of(hir.unify(...)))` chains instead of the concrete type constructor (e.g., `hir.int_type`).
  After splitting, `resolveHIRType` in HIR-to-MIR can't determine the function's return type from these ops, causing lowering to fail.
  Affected tests: `add-const.si`, `bitwise.si`, `multi-phase.si`, `nested-calls.si`, `operations.si`.

- `shouldLower` in HIRToMIR.cpp only checks `call.getTypeOfResults()`, not `call.getTypeOfArgs()`.
  It should check both to avoid attempting to lower calls with unresolvable arg types.
