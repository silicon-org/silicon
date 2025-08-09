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
        item.loc, cx.builder.getStringAttr(item.name));
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

Value Context::withinConst(Location loc, llvm::function_ref<Value()> fn) {
  // Create a new const op to contain any ops `fn` might create.
  auto type = hir::ConstType::get(builder.getContext(),
                                  hir::TypeType::get(builder.getContext()));
  auto constOp = hir::ConstOp::create(builder, loc, type, {});

  // Create the body block and set the insertion point to it.
  auto guard = OpBuilder::InsertionGuard(builder);
  builder.setInsertionPointToStart(&constOp.getBody().emplaceBlock());

  // Call `fn` to populate the body.
  auto value = fn();
  if (!value)
    return {};

  // Yield the value from the body.
  hir::YieldOp::create(builder, loc, value);

  // Adjust the result type to match the yielded value and return it.
  auto result = constOp.getResult(0);
  result.setType(hir::ConstType::get(builder.getContext(), value.getType()));
  return result;
}
