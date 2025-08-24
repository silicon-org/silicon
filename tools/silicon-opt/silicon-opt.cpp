//===-silicon-opt.cpp -----------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/RegisterAll.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  silicon::registerAllDialects(registry);
  silicon::registerAllPasses();
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Silicon modular optimizer driver", registry));
}
