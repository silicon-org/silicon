//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/HIR/Dialect.h"
#include "silicon/MIR/Dialect.h"
#include "mlir/Pass/Pass.h"
#include <memory>
#include <optional>

namespace silicon {

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "silicon/Transforms/Passes.h.inc"

} // namespace silicon
