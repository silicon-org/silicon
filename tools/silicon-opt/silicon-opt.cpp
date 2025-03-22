//===-silicon-opt.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "silicon/RegisterAll.h"
#include "llvm/Support/TargetSelect.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerBuiltinDialectTranslation(registry);
  mlir::registerLLVMDialectTranslation(registry);
  silicon::registerAllDialects(registry);
  silicon::registerAllPasses();
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  return mlir::failed(mlir::MlirOptMain(
      argc, argv, "Silicon modular optimizer driver", registry));
}
