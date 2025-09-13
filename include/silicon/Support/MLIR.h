//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
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
#include "mlir/IR/TypeRange.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/DynamicAPInt.h"
#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace silicon {
using llvm::APInt;
using llvm::ArrayRef;
using llvm::cast;
using llvm::dbgs;
using llvm::DenseSet;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::DynamicAPInt;
using llvm::failed;
using llvm::failure;
using llvm::FailureOr;
using llvm::isa;
using llvm::isa_and_nonnull;
using llvm::LogicalResult;
using llvm::ParseResult;
using llvm::PointerUnion;
using llvm::SmallDenseMap;
using llvm::SmallDenseSet;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SMLoc;
using llvm::SourceMgr;
using llvm::StringRef;
using llvm::succeeded;
using llvm::success;
using llvm::Twine;
using mlir::Attribute;
using mlir::Block;
using mlir::BlockArgument;
using mlir::DenseMap;
using mlir::emitError;
using mlir::emitRemark;
using mlir::emitWarning;
using mlir::FileLineColLoc;
using mlir::FileLineColRange;
using mlir::FlatSymbolRefAttr;
using mlir::Location;
using mlir::MLIRContext;
using mlir::ModuleOp;
using mlir::OpAsmParser;
using mlir::OpAsmPrinter;
using mlir::OpBuilder;
using mlir::Operation;
using mlir::OpOperand;
using mlir::OpResult;
using mlir::OwningOpRef;
using mlir::ParseResult;
using mlir::Region;
using mlir::StringAttr;
using mlir::SymbolTable;
using mlir::Type;
using mlir::TypeRange;
using mlir::TypeSwitch;
using mlir::UnknownLoc;
using mlir::Value;
using mlir::ValueRange;

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
