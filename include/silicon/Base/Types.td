//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SILICON_BASE_TYPES_TD
#define SILICON_BASE_TYPES_TD

include "silicon/Base/Dialect.td"
include "mlir/IR/AttrTypeBase.td"

class BaseTypeDef<
  string name,
  list<Trait> traits = [],
  string baseCppClass = "mlir::Type"
> : TypeDef<BaseDialect, name, traits, baseCppClass> {
  let mnemonic = ?;
}

def TypeType : BaseTypeDef<"Type"> {
  let mnemonic = "type";
  let summary = "a value representing a type";
  let description = [{
    The `si.type` type represents a first-class type value. Values of
    this type are used during phased execution to pass types around as
    SSA values, allowing type computations to be expressed as regular
    code that is evaluated at compile time.
  }];
}

def IntType : BaseTypeDef<"Int"> {
  let mnemonic = "int";
  let summary = "generic integer type";
  let description = [{
    A generic, unsized integer type used during compile-time evaluation.
    Unlike `si.uint`, this type does not carry a fixed bit width, and is
    intended for arbitrary-precision integer arithmetic in earlier
    compilation phases. It is lowered to a sized type before hardware
    generation.
  }];
}

def UIntType : BaseTypeDef<"UInt"> {
  let mnemonic = "uint";
  let summary = "sized unsigned integer type";
  let description = [{
    A fixed-width unsigned integer type, parameterized by its bit width.
    This is the primary hardware integer type and maps directly to
    CIRCT's `hw.int` types during lowering. The width must be at least 1.

    Example: `si.uint<8>` is an 8-bit unsigned integer.
  }];
  let parameters = (ins "uint64_t":$width);
  let assemblyFormat = "`<` $width `>`";
  let genVerifyDecl = 1;
}

def UnitType : BaseTypeDef<"Unit"> {
  let mnemonic = "unit";
  let summary = "unit type";
  let description = [{
    The unit type, analogous to `void` in C or `()` in Rust. Used as
    the result type of functions that do not return a meaningful value.
  }];
}

def AnyfuncType : BaseTypeDef<"Anyfunc"> {
  let mnemonic = "anyfunc";
  let summary = "any function type";
  let description = [{
    A type that matches any function. Used as a placeholder type
    constraint in contexts where the specific function signature is not
    yet known or not relevant.
  }];
}

def BoolType : BaseTypeDef<"Bool"> {
  let mnemonic = "bool";
  let summary = "boolean type";
  let description = [{
    A boolean type representing `true` or `false`. Maps to a single-bit
    value in hardware.
  }];
}

def OpaqueType : BaseTypeDef<"Opaque"> {
  let mnemonic = "opaque";
  let summary = "opaque value bundle passed between phases";
  let description = [{
    An opaque type used to bundle multiple SSA values into a single
    value that is passed between split-phase functions. During phase
    splitting, values that need to cross phase boundaries are packed
    into an `si.opaque` value using `hir.opaque_pack` and unpacked on
    the receiving side with `hir.opaque_unpack`. The opaque type hides
    the concrete types of the bundled values, which are resolved after
    specialization.
  }];
}

#endif // SILICON_BASE_TYPES_TD
