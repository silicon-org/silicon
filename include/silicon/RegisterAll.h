//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "silicon/Dialect/HIR/HIRDialect.h"
#include "silicon/Dialect/HIR/HIRPasses.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace silicon {

/// Register all the relevant dialects with the provided registry.
inline void registerAllDialects(mlir::DialectRegistry &registry) {
  // Register the MLIR dialects.
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);

  // Register the CIRCT dialects.
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();

  // Register the Silicon dialects.
  registry.insert<hir::HIRDialect>();
}

/// Register all the relevant passes.
inline void registerAllPasses() {
  // Register the MLIR passes.
  mlir::registerCSEPass();
  mlir::registerInlinerPass();
  mlir::registerCanonicalizerPass();
  mlir::registerViewOpGraphPass();
  mlir::registerSymbolDCEPass();

  // Register the CIRCT passes.
  // none

  // Register the Silicon passes.
  hir::registerPasses();
}

} // namespace silicon
