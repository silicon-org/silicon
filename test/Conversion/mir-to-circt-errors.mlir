// RUN: silicon-opt --lower-mir-to-circt --split-input-file --verify-diagnostics %s

// Module function with unconvertible input type.
// expected-error @+1 {{module function has unconvertible input type: '!si.opaque'}}
mir.func @bad_module(%a: !si.opaque) -> (result: !si.opaque) attributes {isModule} {
  mir.return %a : !si.opaque
}

// -----

// Module function with unconvertible result type.
// expected-error @+1 {{module function has unconvertible result type: '!si.opaque'}}
mir.func @bad_module_result(%a: !si.int) -> (result: !si.opaque) attributes {isModule} {
  %c = mir.constant #si.opaque<[]> : !si.opaque
  mir.return %c : !si.opaque
}
