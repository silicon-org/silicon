//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # FlattenCF: Lower Structured UIR Control Flow to Block-Based CF
//
// Converts uir.if, uir.loop, and their terminators (uir.yield, uir.break,
// uir.continue, uir.return, uir.unreachable) into cf.br, cf.cond_br, and
// hir.return. Processes ops innermost-first (post-order) so that each
// lowering step handles only one level of structured CF.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "flatten-cf"

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_FLATTENCFPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

namespace {
struct FlattenCFPass : uir::impl::FlattenCFPassBase<FlattenCFPass> {
  using FlattenCFPassBase::FlattenCFPassBase;

  void runOnOperation() override {
    // TODO: implement
  }
};
} // namespace
