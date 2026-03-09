// RUN: silicon-opt --split-input-file --verify-diagnostics %s

// expected-error @below {{uint width must be at least 1, got 0}}
func.func @zero_width_uint(%a: !si.uint<0>) {
  return
}
