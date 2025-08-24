//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Types.h"
#include "silicon/LLVM.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace silicon;
using namespace hir;

void HIRDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "silicon/HIR/Types.cpp.inc"
      >();
}

// Pull in the generated type definitions.
#define GET_TYPEDEF_CLASSES
#include "silicon/HIR/Types.cpp.inc"

mlir::Type hir::getLowerKind(mlir::Type type) {
  auto *context = type.getContext();
  if (auto constType = dyn_cast<ConstType>(type))
    return ConstType::get(context, getLowerKind(constType.getInnerType()));
  if (isa<TypeType>(type))
    return ValueType::get(context);
  assert(false && "no lower kind");
  return {};
}
