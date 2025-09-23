//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/HIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Syntax/AST.h"

using namespace silicon;
using namespace codegen;

namespace {
struct DeclareItems : public ast::Visitor<DeclareItems> {
  Context &cx;
  DeclareItems(Context &cx) : cx(cx) {}
  using Visitor::preVisitNode;

  void preVisitNode(ast::FnItem &item) {
    auto func = cx.builder.create<hir::UncheckedFuncOp>(
        item.loc, cx.builder.getStringAttr(item.name), StringAttr{});
    cx.funcs.insert({&item, func});
    cx.symbolTable.insert(func);
  }
};
} // namespace

static LogicalResult convert(ast::FnItem &item, Context &cx) {
  return cx.convertFnItem(item);
}

static LogicalResult convert(ast::Root &root, Context &cx) {
  bool hasError = false;
  for (auto *item : root.items)
    if (failed(
            ast::visit(*item, [&](auto &&item) { return convert(item, cx); })))
      hasError = true;
  return failure(hasError);
}

Context::Context(ModuleOp module)
    : module(module), builder(module.getBody(), module.getBody()->end()),
      symbolTable(module) {}

LogicalResult Context::convertAST(AST &ast) {
  DeclareItems declare(*this);
  ast.walk(declare);

  for (auto *root : ast.roots)
    if (failed(convert(*root, *this)))
      return failure();

  return success();
}

unsigned Context::getValueConstness(Value value) {
  for (unsigned idx = 0; idx < constContexts.size(); ++idx)
    if (value.getParentRegion() == constContexts[idx].entry.getParent())
      return idx;
  llvm_unreachable("value not in any region");
}

Value Context::withinConst(llvm::function_ref<Value()> fn) {
  // Populate a region with ops.
  Region region;
  auto ip = builder.saveInsertionPoint();
  builder.setInsertionPointToStart(&region.emplaceBlock());
  auto value = fn();
  if (!value)
    return {};

  // If all operations in the region are side-effect free, we can just inline
  // them into the parent. The const op is only needed to accurately place
  // side-effecting ops into a level of constness. All other ops will have their
  // level of constness determined by a later lowering pass.
  bool allSideEffectFree = llvm::all_of(region, [](auto &block) {
    return llvm::all_of(block,
                        [](auto &op) { return mlir::isMemoryEffectFree(&op); });
  });
  if (allSideEffectFree) {
    // Inline the first block at the location where we would put the const op.
    auto &firstBlock = region.front();
    ip.getBlock()->getOperations().splice(ip.getPoint(),
                                          firstBlock.getOperations());
    region.getBlocks().pop_front();

    // Append the remaining blocks after this block.
    if (!region.empty()) {
      auto &parentRegion = *ip.getBlock()->getParent();
      parentRegion.getBlocks().splice(parentRegion.end(), region.getBlocks());
      // Don't restore the insertion point, since the builder already
      // conveniently points to the end of the last block we spliced in.
    } else {
      // Since there are no other blocks in the region, restore the insertion
      // point to the end of the first block.
      builder.restoreInsertionPoint(ip);
    }
    return value;
  }

  // Otherwise create a yield op to return the value from the region and wrap
  // the region in a const op.
  hir::UncheckedYieldOp::create(builder, value.getLoc(), value);
  builder.restoreInsertionPoint(ip);
  auto op =
      hir::UncheckedConstOp::create(builder, value.getLoc(), value.getType());
  op.getRegion().takeBody(region);
  return op.getResult(0);
}
