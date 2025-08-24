//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/Support/MLIR.h"

namespace silicon {
struct AST;

/// Resolve names in an AST.
///
/// This function traverses the AST, collects all named nodes, and resolves
/// nodes in the AST that can refer to other things by populating their
/// `ast::Binding` fields.
LogicalResult resolveNames(AST &ast);

} // namespace silicon
