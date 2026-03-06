// RUN: silicon-opt --canonicalize %s | FileCheck %s

// CHECK: mir.constant #si.int<42>
func.func @ConstFold() -> !si.int {
  %0 = mir.constant #si.int<42>
  return %0 : !si.int
}
