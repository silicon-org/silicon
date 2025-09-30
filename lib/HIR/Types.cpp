//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Types.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
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

Type hir::getHigherKind(Type type) {
  auto *context = type.getContext();
  if (isa<ValueType>(type))
    return TypeType::get(context);
  return {};
}

Type hir::getLowerKind(Type type) {
  auto *context = type.getContext();
  if (isa<TypeType>(type))
    return ValueType::get(context);
  return {};
}

SmallVector<Type> hir::getHigherKindRange(TypeRange types) {
  return SmallVector<Type>(llvm::map_range(types, getHigherKind));
}

SmallVector<Type> hir::getLowerKindRange(TypeRange types) {
  return SmallVector<Type>(llvm::map_range(types, getLowerKind));
}
