//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include "silicon/LLVM.h"
#include "llvm/Support/Allocator.h"

namespace silicon {
namespace ast {

struct Item;

/// A root node in the AST, corresponding to a parsed source file.
struct Root {
  ArrayRef<Item *> items;
};

/// Base class for all top-level items.
struct Item {};

} // namespace ast

//===----------------------------------------------------------------------===//
// AST Container
//===----------------------------------------------------------------------===//

/// A container that holds an entire AST and owns the memory for all its nodes.
struct AST {
  /// The root nodes of the AST.
  SmallVector<ast::Root *> roots;

  /// Move a single node into the memory owned by the AST and return a reference
  /// to it. The reference remains valid for as long as the AST is alive.
  template <typename T>
  T &create(T &&node) {
    return *new (allocator) T(std::forward<T>(node));
  }

  /// Move an array of nodes into the memory owned by the AST and return an
  /// ArrayRef to it. The ArrayRef remains valid for as long as the AST is
  /// alive.
  template <class Container, typename T = typename std::remove_reference<
                                 Container>::type::value_type>
  ArrayRef<T> array(Container &&container) {
    auto num = llvm::range_size(container);
    T *data = allocator.Allocate<T>(num);
    ArrayRef a(data, num);
    for (auto &&element : container)
      new (data++) T(element);
    return a;
  }

private:
  /// A bump pointer allocator that allocates memory for AST nodes.
  llvm::BumpPtrAllocator allocator;
};

} // namespace silicon
