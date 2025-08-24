//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Codegen.h"
#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Dialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Verifier.h"

using namespace silicon;
using namespace codegen;

OwningOpRef<ModuleOp> silicon::convertToIR(MLIRContext *context, AST &ast) {
  // Make sure the dialects we need are loaded.
  context->loadDialect<hir::HIRDialect, mlir::func::FuncDialect>();

  // Create a new MLIR module to hold the converted AST.
  auto module = ModuleOp::create(UnknownLoc::get(context));

  // Generate IR from the AST.
  Context cx(module);
  if (failed(cx.convertAST(ast)))
    return {};

  // Check that the module we generated is valid. This should never happen. We
  // should catch all user errors at the AST level during conversion. If the
  // verifier fails here, that's a compiler bug.
  if (failed(mlir::verify(module))) {
    emitBug(module.getLoc()) << "compiler generated invalid IR";
    return {};
  }
  return module;
}
