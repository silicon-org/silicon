// RUN: silicon-opt --lower-hir-to-mir --verify-diagnostics %s

// expected-error @below {{block argument type could not be determined during HIR-to-MIR lowering; add hir.coerce_type}}
hir.func @UntypedArg(%a) -> () {
  hir.return
}
