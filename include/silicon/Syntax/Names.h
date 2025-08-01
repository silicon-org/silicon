//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/LLVM.h"

namespace silicon {
struct AST;

/// Resolve names in an AST.
///
/// This function traverses the AST, collects all named nodes, and resolves
/// nodes in the AST that can refer to other things by populating their
/// `ast::Binding` fields.
LogicalResult resolveNames(AST &ast);

} // namespace silicon
