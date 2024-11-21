from __future__ import annotations
from silicon.source import Loc
from silicon.dialect.util import try_move_before
from typing import *
from xdsl.dialects.builtin import AnyIntegerAttr
from xdsl.dialects.builtin import IntAttr
from xdsl.dialects.builtin import IntegerAttr
from xdsl.dialects.builtin import StringAttr
from xdsl.ir import OpResult
from xdsl.ir import ParametrizedAttribute
from xdsl.ir import TypeAttribute
from xdsl.ir import Operation
from xdsl.ir import Block
from xdsl.ir import SSAValue
from xdsl.ir import Data
from xdsl.irdl import irdl_attr_definition
from xdsl.irdl import ParameterDef
from xdsl.irdl import irdl_op_definition
from xdsl.irdl import IRDLOperation
from xdsl.irdl import Operand
from xdsl.irdl import operand_def
from xdsl.irdl import opt_operand_def
from xdsl.irdl import prop_def
from xdsl.irdl import result_def
from xdsl.parser import Parser
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.pattern_rewriter import RewritePattern
from xdsl.printer import Printer
from xdsl.traits import ConstantLike
from xdsl.traits import HasCanonicalizationPatternsTrait
from xdsl.traits import Pure
from xdsl.traits import NoMemoryEffect
from xdsl.traits import MemoryReadEffect
from xdsl.traits import MemoryWriteEffect
from xdsl.utils.hints import isa
import xdsl
import silicon

#===------------------------------------------------------------------------===#
# Attributes
#===------------------------------------------------------------------------===#


@irdl_attr_definition
class LocAttr(Data[Loc]):
  name = "loc"

  @classmethod
  def parse_parameter(cls, parser: xdsl.parser.AttrParser) -> Loc:
    if parser.parse_optional_keyword("loc"):
      parser.parse_punctuation("(")
      file = parser.parse_str_literal()
      parser.parse_punctuation(":")
      offset = parser.parse_integer()
      parser.parse_punctuation(":")
      length = parser.parse_integer()
      parser.parse_punctuation(")")
      return Loc(silicon.source.SourceFile(file, ""), offset, length)
    return Loc(
        silicon.source.SourceFile(parser.lexer.input.name,
                                  parser.lexer.input.content), parser.pos - 1,
        0)

  def print_parameter(self, printer: Printer) -> None:
    if self.data != Loc.unknown():
      printer.print_string("loc(")
      printer.print_string_literal(self.data.file.path)
      printer.print_string(f":{self.data.offset}")
      printer.print_string(f":{self.data.length}")
      printer.print_string(")")


#===------------------------------------------------------------------------===#
# Traits
#===------------------------------------------------------------------------===#


class HasCanonicalizeMethod(HasCanonicalizationPatternsTrait):

  @classmethod
  def get_canonicalization_patterns(cls):
    return (ApplyCanonicalizeMethod(),)


class ApplyCanonicalizeMethod(RewritePattern):

  def match_and_rewrite(self, op: Operation, rewriter: PatternRewriter, /):
    if not hasattr(op, "canonicalize"):
      raise NotImplementedError(
          f"operation `{op.name}` has no canonicalize method")
    op.canonicalize(rewriter)


#===------------------------------------------------------------------------===#
# Types
#===------------------------------------------------------------------------===#


# A type indicating that the exact type of an operation has not yet been
# determined.
@irdl_attr_definition
class UnknownType(ParametrizedAttribute, TypeAttribute):
  name = "hir.unknown"


# An unsigned integer with a known width.
@irdl_attr_definition
class UIntType(ParametrizedAttribute, TypeAttribute):
  name = "hir.uint"
  width: ParameterDef[IntAttr]

  def __init__(self, width: int):
    super().__init__([IntAttr(width)])

  @classmethod
  def parse_parameters(cls, parser: xdsl.parser.AttrParser) -> tuple[IntAttr]:
    width = IntAttr.parse_parameter(parser)
    return (IntAttr(width),)

  def print_parameters(self, printer: Printer) -> None:
    self.width.print_parameter(printer)


#===------------------------------------------------------------------------===#
# Constants
#===------------------------------------------------------------------------===#


# Materializes a constant `int` as an SSA value.
@irdl_op_definition
class IntOp(IRDLOperation):
  name = "hir.int"
  loc = prop_def(LocAttr)
  value = prop_def(IntAttr)
  result = result_def(UnknownType())
  traits = frozenset([Pure(), ConstantLike()])

  def __init__(self, value: int, loc: Loc):
    super().__init__(
        properties={
            "value": IntAttr(value),
            "loc": LocAttr(loc)
        },
        result_types=[UnknownType()],
    )

  @classmethod
  def parse(cls, parser: Parser) -> IntOp:
    p0 = parser.pos
    value = parser.parse_attribute()
    loc = LocAttr.parse_parameter(parser)
    if not isa(value, AnyIntegerAttr):
      parser.raise_error("invalid constant value", p0, parser.pos)
    op = IntOp(value.value.data, loc)
    return op

  def print(self, printer: Printer):
    printer.print(" ")
    printer.print(self.value.data)
    printer.print(" ")
    self.loc.print_parameter(printer)


# Materializes a constant type as an SSA value.
@irdl_op_definition
class TypeOp(IRDLOperation):
  name = "hir.type"
  loc: LocAttr = prop_def(LocAttr)
  type: TypeAttribute = prop_def(TypeAttribute)
  result: OpResult = result_def(UnknownType())
  assembly_format = "$type attr-dict $loc"
  traits = frozenset([Pure(), ConstantLike()])

  def __init__(self, type: TypeAttribute, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        properties={
            "type": type,
            "loc": LocAttr(loc)
        })


# Materializes an integer literal of some type as an SSA value.
@irdl_op_definition
class LiteralOp(IRDLOperation):
  name = "hir.literal"
  loc = prop_def(LocAttr)
  value = prop_def(IntAttr)
  type = operand_def(UnknownType())
  result = result_def(UnknownType())
  traits = frozenset([Pure()])

  def __init__(self, value: int, type: SSAValue, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        operands=[type],
        properties={
            "value": IntAttr(value),
            "loc": LocAttr(loc)
        },
    )

  @classmethod
  def parse(cls, parser: Parser) -> LiteralOp:
    p0 = parser.pos
    value = parser.parse_attribute()
    parser.parse_punctuation(",")
    type = parser.parse_operand()
    loc = LocAttr.parse_parameter(parser)
    if not isa(value, AnyIntegerAttr):
      parser.raise_error("invalid constant value", p0, parser.pos)
    op = LiteralOp(value.value.data, type, loc)
    return op

  def print(self, printer: Printer):
    printer.print(" ")
    printer.print(self.value.data)
    printer.print(", ")
    printer.print(self.type)
    printer.print(" ")
    self.loc.print_parameter(printer)


#===------------------------------------------------------------------------===#
# Arithmetic
#===------------------------------------------------------------------------===#


# Add two values.
@irdl_op_definition
class AddOp(IRDLOperation):
  name = "hir.add"
  loc = prop_def(LocAttr)
  lhs = operand_def(UnknownType())
  rhs = operand_def(UnknownType())
  type = operand_def(UnknownType())
  result = result_def(UnknownType())
  assembly_format = "$lhs `,` $rhs `,` $type attr-dict $loc"
  traits = frozenset([Pure(), HasCanonicalizeMethod()])

  def __init__(self, lhs: SSAValue, rhs: SSAValue, type: SSAValue, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        operands=[lhs, rhs, type],
        properties={"loc": LocAttr(loc)})

  def canonicalize(self, rewriter: PatternRewriter):
    # if isinstance(self.lhs.owner, IntOp) and isinstance(self.rhs.owner, IntOp):
    #   const = IntOp(self.lhs.owner.value.data + self.rhs.owner.value.data,
    #                 self.loc.data)
    #   rewriter.replace_matched_op(const)
    #   return
    return


#===------------------------------------------------------------------------===#
# Type Construction
#===------------------------------------------------------------------------===#


# Create an int type.
@irdl_op_definition
class IntTypeOp(IRDLOperation):
  name = "hir.int_type"
  loc: LocAttr = prop_def(LocAttr)
  result: OpResult = result_def(UnknownType())
  assembly_format = "attr-dict $loc"
  traits = frozenset([Pure()])

  def __init__(self, loc: Loc):
    super().__init__(
        result_types=[UnknownType()], properties={"loc": LocAttr(loc)})


# Create a uint type.
@irdl_op_definition
class UIntTypeOp(IRDLOperation):
  name = "hir.uint_type"
  loc: LocAttr = prop_def(LocAttr)
  width: Operand = operand_def(UnknownType())
  result: OpResult = result_def(UnknownType())
  assembly_format = "$width attr-dict $loc"
  traits = frozenset([Pure(), HasCanonicalizeMethod()])

  def __init__(self, width: SSAValue, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        operands=[width],
        properties={"loc": LocAttr(loc)})

  def canonicalize(self, rewriter: PatternRewriter):
    # if isinstance(self.width.owner, IntOp):
    #   width = self.width.owner.value.data
    #   type = TypeOp(UIntType(width), self.loc.data)
    #   rewriter.replace_matched_op(type)
    #   return
    return


# Create a reference type.
@irdl_op_definition
class RefTypeOp(IRDLOperation):
  name = "hir.ref_type"
  loc = prop_def(LocAttr)
  inner = operand_def(UnknownType())
  result = result_def(UnknownType())
  assembly_format = "$inner attr-dict $loc"
  traits = frozenset([Pure()])

  def __init__(self, inner: SSAValue, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        operands=[inner],
        properties={"loc": LocAttr(loc)})


# Unpack a reference type to the type the reference points to.
@irdl_op_definition
class UnpackRefOp(IRDLOperation):
  name = "hir.unpack_ref"
  loc: LocAttr = prop_def(LocAttr)
  input: Operand = operand_def(UnknownType())
  result: OpResult = result_def(UnknownType())
  assembly_format = "$input attr-dict $loc"
  traits = frozenset([Pure(), HasCanonicalizeMethod()])

  def canonicalize(self, rewriter: PatternRewriter):
    # unpack_ref(ref_type(a)) -> a
    if isinstance(self.input.owner, RefTypeOp):
      rewriter.replace_matched_op([], [self.input.owner.inner])
      return


# Get the type of a value.
@irdl_op_definition
class TypeOfOp(IRDLOperation):
  name = "hir.type_of"
  loc: LocAttr = prop_def(LocAttr)
  input: Operand = operand_def(UnknownType())
  result: OpResult = result_def(UnknownType())
  assembly_format = "$input attr-dict $loc"
  traits = frozenset([Pure(), HasCanonicalizeMethod()])

  def canonicalize(self, rewriter: PatternRewriter):
    # type_of(let(_, T)) -> ref_type(T)
    if isinstance(self.input.owner, LetOp):
      ref_type = RefTypeOp(self.input.owner.type, self.loc.data)
      rewriter.replace_matched_op(ref_type)
      return

    # type_of(int) -> int_type
    if isinstance(self.input.owner, IntOp):
      int_type = IntTypeOp(self.loc.data)
      rewriter.replace_matched_op(int_type)
      return

    # Handle general type ops.
    if isinstance(self.input.owner, (LiteralOp, AddOp, LoadOp)):
      rewriter.replace_matched_op([], [self.input.owner.type])
      return


#===------------------------------------------------------------------------===#
# Declarations
#===------------------------------------------------------------------------===#


@irdl_op_definition
class LetOp(IRDLOperation):
  name = "hir.let"
  loc: LocAttr = prop_def(LocAttr)
  decl_name: StringAttr = prop_def(StringAttr)
  type = operand_def(UnknownType())
  result: OpResult = result_def(UnknownType())
  assembly_format = "$decl_name `,` $type attr-dict $loc"
  traits = frozenset([NoMemoryEffect(), HasCanonicalizeMethod()])

  def canonicalize(self, rewriter: PatternRewriter):
    if self.type and isinstance(self.type.owner, TypeOp):
      # self.result.type = self.type.owner.type

      # self.type.erase()
      # rewriter.replace_matched_op(self)
      # rewriter.insert_op_after(self, EraseTypeOp(self.result, self.loc.data))
      return


# Read a value from a reference.
@irdl_op_definition
class LoadOp(IRDLOperation):
  name = "hir.load"
  loc = prop_def(LocAttr)
  target = operand_def(UnknownType())
  type = operand_def(UnknownType())
  result = result_def(UnknownType())
  assembly_format = "$target `,` $type attr-dict $loc"
  traits = frozenset([MemoryReadEffect()])


# Store a value to a reference.
@irdl_op_definition
class StoreOp(IRDLOperation):
  name = "hir.store"
  loc = prop_def(LocAttr)
  target = operand_def(UnknownType())
  value = operand_def(UnknownType())
  type = operand_def(UnknownType())
  assembly_format = "$target `,` $value `,` $type attr-dict $loc"
  traits = frozenset([MemoryWriteEffect()])


#===------------------------------------------------------------------------===#
# Inference
#===------------------------------------------------------------------------===#


# An operation that acts as a standin for an inferrable value. The `unify`
# operation will replace this op with a concrete value if necessary.
@irdl_op_definition
class InferrableOp(IRDLOperation):
  name = "hir.inferrable"
  loc: LocAttr = prop_def(LocAttr)
  result: OpResult = result_def(UnknownType())
  assembly_format = "attr-dict $loc"
  traits = frozenset([NoMemoryEffect()])


# Unify two types or values into one common type. Potentially assigns concrete
# values to inferrable ops in order to make the two sides equal.
@irdl_op_definition
class UnifyOp(IRDLOperation):
  name = "hir.unify"
  loc = prop_def(LocAttr)
  lhs = operand_def(UnknownType())
  rhs = operand_def(UnknownType())
  result = result_def(UnknownType())
  assembly_format = "$lhs `,` $rhs attr-dict $loc"
  traits = frozenset([NoMemoryEffect(), HasCanonicalizeMethod()])

  def __init__(self, lhs: SSAValue, rhs: SSAValue, loc: Loc):
    super().__init__(
        result_types=[UnknownType()],
        operands=[lhs, rhs],
        properties={"loc": LocAttr(loc)})

  def canonicalize(self, rewriter: PatternRewriter):
    # unify(a, a) -> a
    if self.lhs == self.rhs:
      rewriter.replace_matched_op([], [self.lhs])
      return

    # unify(inferrable, non-inferrable) -> unify(non-inferrable, inferrable)
    if isinstance(self.lhs.owner, InferrableOp) and not isinstance(
        self.rhs.owner, InferrableOp):
      swapped = UnifyOp(self.rhs, self.lhs, self.loc.data)
      rewriter.replace_matched_op(swapped)
      return

    # unify(a, inferrable) -> a; inferrable = a
    if isinstance(self.rhs.owner, InferrableOp):
      if try_move_before(self.lhs, self.rhs):
        rewriter.replace_op(self.rhs.owner, [], [self.lhs])
        rewriter.replace_matched_op([], [self.lhs])
        return

    # unify(uint_type(a), uint_type(b)) -> uint_type(unify(a, b))
    if isinstance(self.lhs.owner, UIntTypeOp) and isinstance(
        self.rhs.owner, UIntTypeOp):
      if try_move_before(self.lhs, self.rhs):
        width = UnifyOp(self.lhs.owner.width, self.rhs.owner.width,
                        self.loc.data)
        type = UIntTypeOp(width.result, self.lhs.owner.loc.data)
        rewriter.insert_op_before([width, type], self.rhs.owner)
        rewriter.replace_op(self.rhs.owner, [], [type.result])
        rewriter.replace_op(self.lhs.owner, [], [type.result])
        rewriter.replace_matched_op([], [type.result])
        return


#===------------------------------------------------------------------------===#
# Dialect
#===------------------------------------------------------------------------===#

HIRDialect = xdsl.ir.Dialect("hir", [
    AddOp,
    InferrableOp,
    IntOp,
    IntTypeOp,
    LetOp,
    LiteralOp,
    LoadOp,
    RefTypeOp,
    StoreOp,
    TypeOfOp,
    TypeOp,
    UIntTypeOp,
    UnifyOp,
    UnpackRefOp,
], [
    UIntType,
    UnknownType,
])
