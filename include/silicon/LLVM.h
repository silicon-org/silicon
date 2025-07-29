//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/ADT/PointerUnion.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/SMLoc.h"
#include "llvm/Support/SourceMgr.h"

namespace silicon {
using llvm::APInt;
using llvm::ArrayRef;
using llvm::cast;
using llvm::dyn_cast;
using llvm::dyn_cast_or_null;
using llvm::failed;
using llvm::failure;
using llvm::FailureOr;
using llvm::isa;
using llvm::isa_and_nonnull;
using llvm::LogicalResult;
using llvm::ParseResult;
using llvm::PointerUnion;
using llvm::SmallString;
using llvm::SmallVector;
using llvm::SMLoc;
using llvm::SourceMgr;
using llvm::StringRef;
using llvm::succeeded;
using llvm::success;
using llvm::Twine;
} // namespace silicon
