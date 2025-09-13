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
    auto func = cx.builder.create<hir::FuncOp>(
        item.loc, cx.builder.getStringAttr(item.name),
        cx.builder.getStringAttr("private"));
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
