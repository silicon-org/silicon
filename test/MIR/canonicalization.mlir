// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK: mir.constant #mir.int<42>
func.func @ConstFold() -> !mir.int {
  %0 = mir.constant #mir.int<42>
  return %0 : !mir.int
}
