//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"

using namespace mlir;
using namespace silicon;
using namespace mir;

// Pull in the generated dialect definition.
#define GET_OP_CLASSES
#include "silicon/MIR/Ops.cpp.inc"

/// Print an abbreviation of a type that can be used in an assembly name.
static bool getTypeAbbrev(llvm::raw_ostream &os, Type type) {
  return TypeSwitch<Type, bool>(type)
      .Case<IntType>([&](auto type) {
        os << type.getMnemonic();
        return true;
      })
      .Default([](auto) { return false; });
}

/// Print an abbreviation of an attribute that can be used in an assembly name.
static bool getAttrAbbrev(llvm::raw_ostream &os, Attribute attr) {
  return TypeSwitch<Attribute, bool>(attr)
      .Case<IntAttr>([&](auto attr) {
        os << 'c' << attr.getValue();
        return true;
      })
      .Default([](auto) { return false; });
}

//===----------------------------------------------------------------------===//
// ConstantOp
//===----------------------------------------------------------------------===//

OpFoldResult ConstantOp::fold(FoldAdaptor adaptor) { return getValue(); }

void ConstantOp::getAsmResultNames(
    llvm::function_ref<void(Value, StringRef)> setNameFn) {
  SmallString<32> buffer;
  llvm::raw_svector_ostream bufferStream(buffer);
  if (!getAttrAbbrev(bufferStream, getValue()))
    return;
  bufferStream << '_';
  if (!getTypeAbbrev(bufferStream, getType()))
    buffer.pop_back();
  setNameFn(getResult(), buffer);
}
