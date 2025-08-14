//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"

namespace silicon {
using mlir::Block;
using mlir::BlockArgument;
using mlir::emitError;
using mlir::emitRemark;
using mlir::emitWarning;
using mlir::FileLineColLoc;
using mlir::FileLineColRange;
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::OwningOpRef;
using mlir::Region;
using mlir::StringAttr;
using mlir::SymbolTable;
using mlir::UnknownLoc;
using mlir::Value;

/// Emit a diagnostic indicating a compiler bug at the given location.
inline mlir::InFlightDiagnostic emitBug(Location loc, const char *func,
                                        unsigned line) {
  auto d = emitError(loc);
  d << "compiler bug: ";
  d.attachNote() << func << " on line " << line;
  return d;
}
#define emitBug(loc) ::silicon::emitBug(loc, __PRETTY_FUNCTION__, __LINE__)

} // namespace silicon
