//===- HIRPasses.h - High-level IR transformation passes ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace mlir {
class Pass;
} // namespace mlir

namespace silicon {
namespace hir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "silicon/Dialect/HIR/HIRPasses.h.inc"

} // namespace hir
} // namespace silicon
