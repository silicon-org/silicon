//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_UIR_OPINTERFACES_TD
#define SILICON_UIR_OPINTERFACES_TD

include "mlir/IR/OpBase.td"

def DeclaredFuncOpInterface : OpInterface<"DeclaredFuncOpInterface"> {
  let cppNamespace = "::silicon::uir";
  let description = [{
    An op implementing this interface declares a function interface: a
    signature region computing one type value per argument and result, a
    phase for each of those arguments and results, their names, and whether
    the function is a hardware module.

    A `uir.call` targets such ops interchangeably, and everything that reads
    a declaration -- call checking inlining the signature at a use, call
    decomposition deriving the phase span, the `uir.signature` verifier
    checking arity -- goes through this interface instead of distinguishing
    the implementing ops. `uir.func` declares an interface alongside the
    body implementing it, while `uir.split_func` is the declaration on its
    own, whose implementation is compiled, split into per-phase functions,
    or not present at all.
  }];
  let methods = [
    InterfaceMethod<[{
      The signature region computing the declared argument and result types.

      Its block arguments are the function's arguments, so a dependent type
      may be computed from them, and it exits through `uir.signature`.
    }], "::mlir::Region &", "getSignature">,
    InterfaceMethod<[{
      The phase of each declared argument.
    }], "::llvm::ArrayRef<int32_t>", "getArgPhases">,
    InterfaceMethod<[{
      The phase of each declared result.
    }], "::llvm::ArrayRef<int32_t>", "getResultPhases">,
    InterfaceMethod<[{
      The name of each declared argument.
    }], "::mlir::ArrayAttr", "getArgNames">,
    InterfaceMethod<[{
      The name of each declared result.
    }], "::mlir::ArrayAttr", "getResultNames">,
    InterfaceMethod<[{
      Check whether the declared function is a hardware module.
    }], "bool", "isModule", (ins), [{
      return $_op.getIsModule().value_or(false);
    }]>,
  ];
  let extraSharedClassDeclaration = [{
    /// The declared phase span: the contiguous range from the earliest to the
    /// latest declared argument or result phase, always including phase 0.
    std::pair<int32_t, int32_t> getPhaseSpan() {
      return ::silicon::uir::computePhaseSpan($_op.getArgPhases(),
                                              $_op.getResultPhases());
    }
  }];
}

#endif // SILICON_UIR_OPINTERFACES_TD
