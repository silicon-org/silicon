//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace silicon {

/// Register all the relevant dialects with the provided registry.
void registerAllDialects(mlir::DialectRegistry &registry);

/// Register all the relevant passes.
void registerAllPasses();

} // namespace silicon
