//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "silicon/HIR/Dialect.h"
#include "silicon/MIR/Dialect.h"
#include "silicon/RegisterAll.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/Extensions/InlinerExtension.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/DialectRegistry.h"

void silicon::registerAllDialects(mlir::DialectRegistry &registry) {
  // Register the MLIR dialects.
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::func::FuncDialect>();
  mlir::func::registerInlinerExtension(registry);

  // Register the CIRCT dialects.
  registry.insert<circt::comb::CombDialect>();
  registry.insert<circt::hw::HWDialect>();
  registry.insert<circt::seq::SeqDialect>();

  // Register the Silicon dialects.
  registry.insert<silicon::hir::HIRDialect>();
  registry.insert<silicon::mir::MIRDialect>();
}
