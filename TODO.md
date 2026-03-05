- Fix `hir.type_of` and `hir.inferrable` producing non-constant type operands in split-phase functions.
  The strict `shouldLower` check in HIR-to-MIR now rejects functions where `hir.return.typeOfValues` or `hir.call.typeOfResults` come from `type_of` or `inferrable` ops.
  These functions pack values into opaque bundles for cross-phase transfer, so their return types are essentially metadata.
  Earlier passes (split-phases, check-calls) should produce resolvable type operands for these cases, or the type operands should be restructured so they don't block lowering.
  Affected tests: `add-const.si`, `bitwise.si`, `multi-phase.si`, `nested-calls.si`, `operations.si`.
