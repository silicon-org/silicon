//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Passes.h"
#include "silicon/RegisterAll.h"
#include "mlir/Transforms/Passes.h"

void silicon::registerAllPasses() {
  // Register the MLIR passes.
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();

  // Register the CIRCT passes.
  // none

  // Register the Silicon passes.
  silicon::hir::registerPasses();
}
