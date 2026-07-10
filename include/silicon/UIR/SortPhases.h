//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "silicon/Support/MLIR.h"
#include "silicon/UIR/Ops.h"

namespace silicon {
namespace uir {

struct PhaseAnalysis;

/// Sort a unified function's content across phase boundaries.
///
/// Consumes the phase analysis of `funcOp` and rewrites the function such
/// that no control flow region contains content of a phase different from
/// the region's own phase: earlier-phase content hoists out of later-phase
/// regions, and later-phase content moves into `hir.replicate`/`hir.pick`
/// receiving constructs fed by hit lists and choices. As a precondition,
/// `uir.expr` and `uir.pin` ops are dissolved (the phase map is the single
/// source of truth afterwards) and `uir.call` ops are decomposed into
/// per-phase `hir.call` ops using the callees' `uir.split_func` witnesses.
///
/// The phase map in `analysis` is invalidated for erased ops and maintained
/// for all created ops, so it remains authoritative for the subsequent
/// distribution of ops into per-phase functions.
LogicalResult sortPhases(FuncOp funcOp, PhaseAnalysis &analysis,
                         SymbolTable &symbolTable);

} // namespace uir
} // namespace silicon
