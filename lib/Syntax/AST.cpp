//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Syntax/AST.h"

using namespace silicon;

ast::Precedence ast::getPrecedence(BinaryOp op) {
  switch (op) {
#define AST_BINARY(NAME, TOKEN, PREC)                                          \
  case BinaryOp::NAME:                                                         \
    return Precedence::PREC;
#include "silicon/Syntax/AST.def"
  };
}
