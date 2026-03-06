//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/MIR/Dialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace silicon {
namespace mir {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "silicon/MIR/Passes.h.inc"

} // namespace mir
} // namespace silicon
