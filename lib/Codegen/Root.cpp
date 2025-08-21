//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Codegen/Context.h"
#include "silicon/Dialect/HIR/HIRTypes.h"
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
        item.loc, cx.builder.getStringAttr(item.name), 0);
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

void Context::increaseConstness() {
  ++currentConstness;
  assert(currentConstness <= constContexts.size());
  if (currentConstness == constContexts.size()) {
    auto region = std::make_unique<Region>();
    auto &entry = region->emplaceBlock();
    OpBuilder builder(module.getContext());
    builder.setInsertionPointToStart(&entry);
    auto returnOp =
        hir::ReturnOp::create(builder, UnknownLoc::get(module.getContext()),
                              ValueRange{}, ValueRange{});
    builder.setInsertionPoint(returnOp);
    constContexts.push_back({builder, entry, returnOp, std::move(region),
                             DenseMap<Value, Value>()});
  }
}

void Context::decreaseConstness() {
  assert(currentConstness > 0);
  --currentConstness;
}

unsigned Context::getValueConstness(Value value) {
  for (unsigned idx = 0; idx < constContexts.size(); ++idx)
    if (value.getParentRegion() == constContexts[idx].entry.getParent())
      return idx;
  llvm_unreachable("value not in any region");
}
