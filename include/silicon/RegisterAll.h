//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
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
