from __future__ import annotations

from typing import Annotated, Dict, TypeVar, List, Generic, Union, Sequence, Tuple
from dataclasses import dataclass

from silicon import ast
from silicon.diagnostics import *
from silicon.source import Loc, SourceFile
from silicon.ty import (
    Type,
    ConstIntParam as TyConstIntParam,
    FreeIntParam as TyFreeIntParam,
    RegType as TyRegType,
    TupleType as TyTupleType,
    UIntType as TyUIntType,
    UnitType as TyUnitType,
    WireType as TyWireType,
    RefType as TyRefType,
)

import xdsl
import xdsl.dialects.utils
from xdsl.builder import (Builder, InsertPoint)
from xdsl.dialects.builtin import (
    AnyIntegerAttr,
    ArrayAttr,
    FlatSymbolRefAttr,
    FunctionType,
    IntAttr,
    IntegerAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
    SymbolRefAttr,
)
from xdsl.ir import (
    Attribute,
    Block,
    Dialect,
    OpResult,
    Operation,
    ParametrizedAttribute,
    Region,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParamAttrConstraint,
    ParameterDef,
    VarOpResult,
    VarOperand,
    attr_def,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    prop_def,
    region_def,
    result_def,
    var_operand_def,
    var_result_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    ConstantLike,
    HasParent,
    IsTerminator,
    IsolatedFromAbove,
    NoTerminator,
    Pure,
    SymbolOpInterface,
    SymbolTable,
)
from xdsl.utils.exceptions import VerifyException
from xdsl.utils.hints import isa

__all__ = ["convert_ast_to_ir"]


def get_loc(value: Union[SSAValue, Operation]) -> Loc:
    if isinstance(value, Operation) and hasattr(value, "loc"):
        assert isinstance(value.loc, Loc)
        return value.loc
    if isinstance(value, SSAValue) and hasattr(value.owner, "loc"):
        assert isinstance(value.owner.loc, Loc)
        return value.owner.loc
    return Loc.unknown()


#===------------------------------------------------------------------------===#
# Types
#===------------------------------------------------------------------------===#


@irdl_attr_definition
class UnitType(ParametrizedAttribute, TypeAttribute):
    name = "si.unit"


@irdl_attr_definition
class TupleType(ParametrizedAttribute, TypeAttribute):
    name = "si.tuple"
    fields: ParameterDef[ArrayAttr[TypeAttribute]]

    def __init__(self, fields: list[TypeAttribute] | ArrayAttr[TypeAttribute]):
        if isinstance(fields, list):
            fields = ArrayAttr(fields)
        super().__init__([fields])


_WireInnerT = TypeVar("_WireInnerT", bound=Attribute)
_RegInnerT = TypeVar("_RegInnerT", bound=Attribute)
_RefInnerT = TypeVar("_RefInnerT", bound=Attribute)


@irdl_attr_definition
class WireType(Generic[_WireInnerT], ParametrizedAttribute, TypeAttribute):
    name = "si.wire"
    inner: ParameterDef[_WireInnerT]

    def __init__(self, inner: _WireInnerT):
        super().__init__([inner])


@irdl_attr_definition
class RegType(Generic[_RegInnerT], ParametrizedAttribute, TypeAttribute):
    name = "si.reg"
    inner: ParameterDef[_RegInnerT]

    def __init__(self, inner: _RegInnerT):
        super().__init__([inner])


@irdl_attr_definition
class RefType(Generic[_RefInnerT], ParametrizedAttribute, TypeAttribute):
    name = "si.ref"
    inner: ParameterDef[_RefInnerT]

    def __init__(self, inner: _RefInnerT):
        super().__init__([inner])


#===------------------------------------------------------------------------===#
# Dialect
#===------------------------------------------------------------------------===#

# See the following for reference:
# - https://github.com/xdslproject/xdsl/blob/main/docs/Toy/toy/dialects/toy.py
# - https://github.com/xdslproject/xdsl/blob/main/docs/Toy/toy/frontend/ir_gen.py


@irdl_op_definition
class SiModuleOp(IRDLOperation):
    name = "si.module"
    sym_name: StringAttr = prop_def(StringAttr)
    body: Region = region_def("single_block")
    traits = frozenset([
        IsolatedFromAbove(),
        SymbolOpInterface(),
        NoTerminator(),
    ])

    def __init__(self, *args, loc: Loc, **kwargs):
        super().__init__(*args, **kwargs)
        self.loc = loc

    @staticmethod
    def with_empty_body(name: str, loc: Loc) -> SiModuleOp:
        return SiModuleOp(
            regions=[Region()],
            properties={"sym_name": StringAttr(name)},
            loc=loc,
        )

    @classmethod
    def parse(cls, parser: Parser) -> SiModuleOp:
        name = parser.parse_symbol_name().data
        attrs = parser.parse_optional_attr_dict_with_keyword()
        region = parser.parse_region()
        if not region.blocks:
            region.add_block(Block())
        op = SiModuleOp(properties={"sym_name": StringAttr(name)},
                        attributes=attrs,
                        regions=[region],
                        loc=Loc.unknown())
        return op

    def print(self, printer: Printer):
        printer.print(f" @{self.sym_name.data}")
        printer.print_op_attributes(self.attributes, print_keyword=True)
        printer.print(" ")
        printer.print_region(self.body, False, False)


@irdl_op_definition
class FuncOp(IRDLOperation):
    name = "si.func"
    sym_name: StringAttr = prop_def(StringAttr)
    function_type: FunctionType = prop_def(FunctionType)
    body: Region = region_def("single_block")
    traits = frozenset([
        IsolatedFromAbove(),
        SymbolOpInterface(),
    ])

    def __init__(
        self,
        name: str,
        function_type: FunctionType,
        loc: Loc,
        region: Region | type[Region.DEFAULT] = Region.DEFAULT,
    ):
        if not isinstance(region, Region):
            region = Region(Block(arg_types=function_type.inputs))
        props = {
            "sym_name": StringAttr(name),
            "function_type": function_type,
        }
        super().__init__(properties=props, regions=[region])
        self.loc = loc

    def verify_(self) -> None:
        # Check that the argument types in the function type match the types of
        # the block arguments.
        types_func = self.function_type.inputs.data
        types_block = tuple(arg.type for arg in self.body.blocks[0].args)
        if types_func != types_block:
            raise VerifyException(
                "entry block argument types must match function input types")

    @classmethod
    def parse(cls, parser: Parser) -> FuncOp:
        (
            name,
            input_types,
            output_types,
            region,
            attributes,
            arg_attrs,
        ) = xdsl.dialects.utils.parse_func_op_like(
            parser,
            reserved_attr_names=("sym_name", "function_type"),
        )
        op = FuncOp(
            name=name,
            function_type=FunctionType.from_lists(input_types, output_types),
            loc=Loc.unknown(),
            region=region,
        )
        if attributes is not None:
            op.attributes |= attributes.data
        return op

    def print(self, printer: Printer):
        xdsl.dialects.utils.print_func_op_like(
            printer,
            self.sym_name,
            self.function_type,
            self.body,
            self.attributes,
            reserved_attr_names=("sym_name", "function_type"),
        )


#===------------------------------------------------------------------------===#
# Statements
#===------------------------------------------------------------------------===#


@irdl_op_definition
class InputPortOp(IRDLOperation):
    name = "si.input"
    port_name: StringAttr = attr_def(StringAttr)
    value: OpResult = result_def()
    assembly_format = "$port_name `:` type($value) attr-dict"
    traits = frozenset([HasParent(SiModuleOp)])

    def __init__(self, name: str, ty: TypeAttribute, loc: Loc):
        super().__init__(result_types=[ty],
                         attributes={"port_name": StringAttr(name)})
        self.loc = loc


@irdl_op_definition
class OutputPortOp(IRDLOperation):
    name = "si.output"
    port_name: StringAttr = attr_def(StringAttr)
    value: Operand = operand_def()
    assembly_format = "$port_name `,` $value `:` type($value) attr-dict"
    traits = frozenset([HasParent(SiModuleOp)])

    def __init__(self, name: str | StringAttr, value: SSAValue, loc: Loc):
        if isinstance(name, str):
            name = StringAttr(name)
        super().__init__(operands=[value], attributes={"port_name": name})
        self.loc = loc


class DeclOpBase(IRDLOperation):
    decl_name: StringAttr = attr_def(StringAttr)
    result: OpResult = result_def()
    assembly_format = "$decl_name `:` type($result) attr-dict"

    def __init__(self, name: str, ty: TypeAttribute, loc: Loc):
        super().__init__(result_types=[RefType(ty)],
                         attributes={"decl_name": StringAttr(name)})
        self.loc = loc


@irdl_op_definition
class OutputDeclOp(DeclOpBase):
    name = "si.output_decl"
    traits = frozenset([HasParent(SiModuleOp)])


@irdl_op_definition
class VarDeclOp(DeclOpBase):
    name = "si.var_decl"


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "si.assign"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    dst: Operand = operand_def(RefType[T])
    src: Operand = operand_def(T)
    assembly_format = "$dst `,` $src `:` type($dst) attr-dict"

    def __init__(self, dst: SSAValue, src: SSAValue, loc: Loc):
        super().__init__(operands=[dst, src])
        self.loc = loc


@irdl_op_definition
class WireDeclOp(IRDLOperation):
    name = "si.wire_decl"
    result: OpResult = result_def()
    assembly_format = "`:` type($result) attr-dict"

    def __init__(self, ty: TypeAttribute, loc: Loc):
        super().__init__(result_types=[WireType(ty)])
        self.loc = loc


@irdl_op_definition
class RegDeclOp(IRDLOperation):
    name = "si.reg_decl"
    clock: Operand = operand_def(IntegerType(1))
    result: OpResult = result_def()
    assembly_format = "$clock `:` type($result) attr-dict"

    def __init__(self, ty: TypeAttribute, clock: SSAValue, loc: Loc):
        super().__init__(result_types=[RegType(ty)], operands=[clock])
        self.loc = loc


@irdl_op_definition
class WireSetOp(IRDLOperation):
    name = "si.wire_set"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    wire: Operand = operand_def(WireType[T])
    value: Operand = operand_def(T)
    assembly_format = "$wire `,` $value `:` type($wire) attr-dict"

    def __init__(self, wire: SSAValue, value: SSAValue, loc: Loc):
        super().__init__(operands=[wire, value])
        self.loc = loc


@irdl_op_definition
class WireGetOp(IRDLOperation):
    name = "si.wire_get"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    wire: Operand = operand_def(WireType[T])
    result: OpResult = result_def(T)
    assembly_format = "$wire `:` type($wire) attr-dict"

    def __init__(self, wire: SSAValue, loc: Loc):
        assert isinstance(wire.type, WireType)
        super().__init__(result_types=[wire.type.inner], operands=[wire])
        self.loc = loc


@irdl_op_definition
class RegNextOp(IRDLOperation):
    name = "si.reg_next"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    reg: Operand = operand_def(RegType[T])
    value: Operand = operand_def(T)
    assembly_format = "$reg `,` $value `:` type($reg) attr-dict"

    def __init__(self, reg: SSAValue, value: SSAValue, loc: Loc):
        super().__init__(operands=[reg, value])
        self.loc = loc


@irdl_op_definition
class RegCurrentOp(IRDLOperation):
    name = "si.reg_current"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    reg: Operand = operand_def(RegType[T])
    result: OpResult = result_def(T)
    assembly_format = "$reg `:` type($reg) attr-dict"

    def __init__(self, reg: SSAValue, loc: Loc):
        assert isinstance(reg.type, RegType)
        super().__init__(result_types=[reg.type.inner], operands=[reg])
        self.loc = loc


@irdl_op_definition
class RegOp(IRDLOperation):
    name = "si.reg"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    clock: Operand = operand_def(IntegerType(1))
    next: Operand = operand_def(T)
    current: OpResult = result_def(T)
    assembly_format = "$clock `,` $next `:` type($current) attr-dict"

    def __init__(self, clock: SSAValue, next: SSAValue, loc: Loc):
        super().__init__(result_types=[next.type], operands=[clock, next])
        self.loc = loc


@irdl_op_definition
class ReturnOp(IRDLOperation):
    name = "si.return"
    args: VarOperand = var_operand_def()
    traits = frozenset([IsTerminator()])

    def __init__(self, args: Sequence[SSAValue], loc: Loc):
        super().__init__(operands=[args])
        self.loc = loc

    def verify_(self) -> None:
        # Ensure we're nested within a function.
        func_op = self.parent_op()
        while func_op and not isinstance(func_op, FuncOp):
            func_op = func_op.parent_op()
        if not func_op:
            raise VerifyException(
                f"'{self.name}' must be nested within '{FuncOp.name}'")
        assert isinstance(func_op, FuncOp)

        # Ensure the values we return match the function type's output.
        types_func = func_op.function_type.outputs.data
        types_return = tuple(arg.type for arg in self.args)
        if types_func != types_return:
            raise VerifyException(
                "return argument types must match function output types")

    @classmethod
    def parse(cls, parser: Parser) -> ReturnOp:
        attrs, args = xdsl.dialects.utils.parse_return_op_like(parser)
        op = ReturnOp(args, Loc.unknown())
        op.attributes |= attrs
        return op

    def print(self, printer: Printer):
        xdsl.dialects.utils.print_return_op_like(printer, self.attributes,
                                                 self.args)


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "si.constant"
    T = Annotated[IntegerType, ConstraintVar("T")]
    value: IntegerAttr = prop_def(IntegerAttr[T])
    result: OpResult = result_def(T)
    traits = frozenset([Pure(), ConstantLike()])

    def __init__(self, value: IntegerAttr, loc: Loc):
        super().__init__(
            result_types=[value.type],
            properties={"value": value},
        )
        self.loc = loc

    @staticmethod
    def from_value(value: int, width: int, loc: Loc) -> ConstantOp:
        return ConstantOp(IntegerAttr(value, IntegerType(width)), loc)

    @classmethod
    def parse(cls, parser: Parser) -> ConstantOp:
        p0 = parser.pos
        value = parser.parse_attribute()
        if not isa(value, AnyIntegerAttr):
            parser.raise_error("invalid constant value", p0, parser.pos)
        attrs = parser.parse_optional_attr_dict()
        op = ConstantOp(value, Loc.unknown())
        op.attributes |= attrs
        return op

    def print(self, printer: Printer):
        printer.print(" ")
        printer.print_attribute(self.value)
        printer.print_op_attributes(self.attributes)


# The unit constant `()`, similar to `void` in C, but being an actual value.
@irdl_op_definition
class ConstantUnitOp(IRDLOperation):
    name = "si.constant_unit"
    result: OpResult = result_def(UnitType)
    assembly_format = "`:` type($result) attr-dict"
    traits = frozenset([Pure(), ConstantLike()])

    def __init__(self, loc: Loc):
        super().__init__(result_types=[UnitType()])
        self.loc = loc


class UnaryOpBase(IRDLOperation):
    T = Annotated[IntegerType, ConstraintVar("T")]
    arg: Operand = operand_def(T)
    result: OpResult = result_def(T)
    assembly_format = "$arg `:` type($arg) attr-dict"
    traits = frozenset([Pure()])

    def __init__(self, arg: SSAValue, loc: Loc):
        super().__init__(result_types=[arg.type], operands=[arg])
        self.loc = loc


@irdl_op_definition
class NegOp(UnaryOpBase):
    name = "si.neg"


@irdl_op_definition
class NotOp(UnaryOpBase):
    name = "si.not"


class BinaryOpBase(IRDLOperation):
    T = Annotated[IntegerType, ConstraintVar("T")]
    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)
    assembly_format = "$lhs `,` $rhs `:` type($lhs) attr-dict"
    traits = frozenset([Pure()])

    def __init__(self, lhs: SSAValue, rhs: SSAValue, loc: Loc):
        super().__init__(result_types=[lhs.type], operands=[lhs, rhs])
        self.loc = loc


@irdl_op_definition
class AddOp(BinaryOpBase):
    name = "si.add"


@irdl_op_definition
class SubOp(BinaryOpBase):
    name = "si.sub"


@irdl_op_definition
class ConcatOp(IRDLOperation):
    name = "si.concat"
    args: VarOperand = var_operand_def(IntegerType)
    result: OpResult = result_def(IntegerType)
    assembly_format = "$args `:` `(` type($args) `)` `->` type($result) attr-dict"
    traits = frozenset([Pure()])

    def __init__(self, args: List[SSAValue], loc: Loc):
        width = 0
        for arg in args:
            assert isinstance(arg.type, IntegerType)
            width += arg.type.width.data
        super().__init__(result_types=[IntegerType(width)], operands=[args])
        self.loc = loc

    def verify_(self) -> None:
        width = 0
        for arg in self.args:
            assert isinstance(arg.type, IntegerType)
            width += arg.type.width.data
        assert isinstance(self.result.type, IntegerType)
        if width != self.result.type.width.data:
            raise VerifyException(
                f"result width {self.result.type.width.data} must match sum of operand widths {width}"
            )


@irdl_op_definition
class ExtractOp(IRDLOperation):
    name = "si.extract"
    arg: Operand = operand_def(IntegerType)
    offset: IntAttr = attr_def(IntAttr)
    result: OpResult = result_def(IntegerType)
    assembly_format = "$arg `,` $offset `:` type($arg) `->` type($result) attr-dict"
    traits = frozenset([Pure()])

    def __init__(self, arg: SSAValue, offset: int, width: int, loc: Loc):
        super().__init__(result_types=[IntegerType(width)],
                         operands=[arg],
                         attributes={"offset": IntAttr(offset)})
        self.loc = loc

    def verify_(self) -> None:
        offset: int = self.offset.data
        assert isinstance(self.arg.type, IntegerType)
        assert isinstance(self.result.type, IntegerType)
        if offset < 0 or offset + self.result.type.width.data > self.arg.type.width.data:
            raise VerifyException(
                f"extract of {self.result.type} from offset {offset} is outside range of input {self.arg.type}"
            )


@irdl_op_definition
class MuxOp(IRDLOperation):
    name = "si.mux"
    T = Annotated[IntegerType, ConstraintVar("T")]
    condition: Operand = operand_def(IntegerType(1))
    true_value: Operand = operand_def(T)
    false_value: Operand = operand_def(T)
    result: OpResult = result_def(T)
    assembly_format = "$condition `,` $true_value `,` $false_value `:` type($true_value) attr-dict"
    traits = frozenset([Pure()])

    def __init__(self, condition: SSAValue, true_value: SSAValue,
                 false_value: SSAValue, loc: Loc):
        super().__init__(result_types=[true_value.type],
                         operands=[condition, true_value, false_value])
        self.loc = loc


# A call to a function.
@irdl_op_definition
class CallOp(IRDLOperation):
    name = "si.call"
    args: VarOperand = var_operand_def()
    callee: FlatSymbolRefAttr = prop_def(FlatSymbolRefAttr)
    res: VarOpResult = var_result_def()

    def __init__(
        self,
        callee: str | SymbolRefAttr,
        args: Sequence[SSAValue | Operation],
        result_types: Sequence[Attribute],
        loc: Loc,
    ):
        if isinstance(callee, str):
            callee = SymbolRefAttr(callee)
        super().__init__(
            operands=[args],
            result_types=[result_types],
            properties={"callee": callee},
        )
        self.loc = loc

    def verify_(self) -> None:
        callee = SymbolTable.lookup_symbol(self, self.callee)
        if not callee:
            raise VerifyException(f"{self.callee} does not exist")
        if not isinstance(callee, FuncOp):
            raise VerifyException(f"{self.callee} cannot be called")

        function_type = FunctionType.from_lists(
            [arg.type for arg in self.args], [res.type for res in self.res])
        if function_type != callee.function_type:
            raise VerifyException(
                f"call type {function_type} differs from callee type {callee.function_type}"
            )

    def get_called_func(self) -> FuncOp:
        callee = SymbolTable.lookup_symbol(self, self.callee)
        assert isinstance(callee, FuncOp)
        return callee

    @classmethod
    def parse(cls, parser: Parser) -> CallOp:
        callee, args, res, attrs = xdsl.dialects.utils.parse_call_op_like(
            parser, reserved_attr_names=["callee"])
        op = CallOp(callee, args, res, Loc.unknown())
        if attrs is not None:
            op.attributes |= attrs.data
        return op

    def print(self, printer: Printer):
        xdsl.dialects.utils.print_call_op_like(
            printer,
            self,
            self.callee,
            self.args,
            self.attributes,
            reserved_attr_names=["callee"],
        )


# Create a tuple.
@irdl_op_definition
class TupleCreateOp(IRDLOperation):
    name = "si.tuple_create"
    args: VarOperand = var_operand_def()
    res: OpResult = result_def()
    assembly_format = "$args `:` `(` type($args) `)` `->` type($res) attr-dict"

    def __init__(
        self,
        args: Sequence[SSAValue],
        loc: Loc,
    ):
        fields: List[TypeAttribute] = []
        for arg in args:
            assert isa(arg.type, TypeAttribute)
            fields.append(arg.type)
        result_type = TupleType(fields)
        super().__init__(
            operands=[args],
            result_types=[result_type],
        )
        self.loc = loc

    # TODO: verify


# Get the value of a tuple field.
@irdl_op_definition
class TupleGetOp(IRDLOperation):
    name = "si.tuple_get"
    arg: Operand = operand_def()
    field: IntAttr = prop_def(IntAttr)
    res: OpResult = result_def()
    assembly_format = "$arg `,` $field `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        field: int,
        loc: Loc,
    ):
        assert isa(arg.type, TupleType)
        super().__init__(
            operands=[arg],
            result_types=[arg.type.fields.data[field]],
            properties={"field": IntAttr(field)},
        )
        self.loc = loc

    # TODO: verify


# Get a reference to a tuple field.
@irdl_op_definition
class TupleGetRefOp(IRDLOperation):
    name = "si.tuple_get_ref"
    arg: Operand = operand_def(RefType[TupleType])
    field: IntAttr = prop_def(IntAttr)
    res: OpResult = result_def(RefType)
    assembly_format = "$arg `,` $field `:` type($arg) `->` type($res) attr-dict"

    def __init__(
        self,
        arg: SSAValue,
        field: int,
        loc: Loc,
    ):
        assert isa(arg.type, RefType)
        assert isa(arg.type.inner, TupleType)
        super().__init__(
            operands=[arg],
            result_types=[RefType(arg.type.inner.fields.data[field])],
            properties={"field": IntAttr(field)},
        )
        self.loc = loc

    def verify_(self) -> None:
        assert isa(self.arg.type, RefType)
        assert isa(self.arg.type.inner, TupleType)
        assert isa(self.res.type, RefType)
        field = self.field.data
        fields = self.arg.type.inner.fields.data
        if field >= len(fields):
            raise VerifyException(f"field {field} out of bounds")
        actual = self.res.type
        expected = RefType(fields[field])
        if expected != actual:
            raise VerifyException(
                f"result type {actual} must match tuple field type {expected}")


@irdl_op_definition
class RefOp(IRDLOperation):
    name = "si.ref"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    arg: Operand = operand_def(T)
    result: OpResult = result_def(RefType[T])
    assembly_format = "$arg `:` type($result) attr-dict"

    def __init__(self, arg: SSAValue, loc: Loc):
        super().__init__(result_types=[RefType(arg.type)], operands=[arg])
        self.loc = loc


@irdl_op_definition
class DerefOp(IRDLOperation):
    name = "si.deref"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    arg: Operand = operand_def(RefType[T])
    result: OpResult = result_def(T)
    assembly_format = "$arg `:` type($arg) attr-dict"

    def __init__(self, arg: SSAValue, loc: Loc):
        assert isinstance(arg.type, RefType)
        super().__init__(result_types=[arg.type.inner], operands=[arg])
        self.loc = loc


#===------------------------------------------------------------------------===#
# Dialect
#===------------------------------------------------------------------------===#

SiliconDialect = Dialect("silicon", [
    AddOp,
    AssignOp,
    CallOp,
    ConcatOp,
    ConstantOp,
    ConstantUnitOp,
    DerefOp,
    ExtractOp,
    FuncOp,
    InputPortOp,
    MuxOp,
    NegOp,
    NotOp,
    OutputDeclOp,
    OutputPortOp,
    RefOp,
    RegCurrentOp,
    RegDeclOp,
    RegNextOp,
    RegOp,
    ReturnOp,
    SiModuleOp,
    SubOp,
    TupleCreateOp,
    TupleGetOp,
    TupleGetRefOp,
    VarDeclOp,
    WireDeclOp,
    WireGetOp,
    WireSetOp,
], [
    RefType,
    RegType,
    TupleType,
    UnitType,
    WireType,
])

#===------------------------------------------------------------------------===#
# Conversion from AST to IR
#===------------------------------------------------------------------------===#


@dataclass(init=False)
class Converter:
    module: ModuleOp
    builder: Builder
    name_uniquification_ids: Dict[str, int]
    named_values: Dict[ast.AstNode, SSAValue]
    is_terminated_at: Loc | None
    warned_about_terminator_at: Loc | None
    generated_fns: Dict[Tuple[ast.FnItem, Tuple[int, ...]], str]
    fn_worklist: List[Tuple[ast.FnItem, Tuple[int, ...], str]]
    params: Dict[int, int]

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])
        self.name_uniquification_ids = dict()
        self.named_values = dict()
        self.is_terminated_at = None
        self.warned_about_terminator_at = None
        self.generated_fns = dict()
        self.fn_worklist = []
        self.params = dict()

    def convert_type(self, ty: Type | None, loc: Loc) -> TypeAttribute:
        assert ty is not None, "type checking should have assigned types"

        if isinstance(ty, TyUnitType):
            return UnitType()
        if isinstance(ty, TyTupleType):
            return TupleType(
                [self.convert_type(field, loc) for field in ty.fields])
        if isinstance(ty, TyUIntType):
            if isinstance(ty.width, TyConstIntParam):
                return IntegerType(ty.width.value)
            if isinstance(ty.width, TyFreeIntParam):
                param = self.params.get(ty.width.num)
                if param is None:
                    emit_error(loc, f"param {ty.width} not set")
                return IntegerType(param)
        if isinstance(ty, TyWireType):
            return WireType(self.convert_type(ty.inner, loc))
        if isinstance(ty, TyRegType):
            return RegType(self.convert_type(ty.inner, loc))
        if isinstance(ty, TyRefType):
            return RefType(self.convert_type(ty.inner, loc))

        emit_error(loc,
                   f"unsupported in IR conversion: {ty.__class__.__name__}")

    def convert_root(self, root: ast.Root) -> ModuleOp:
        for item in root.items:
            if isinstance(item, ast.ModItem):
                self.convert_mod(item)
            elif isinstance(item, ast.FnItem):
                if not item.params:
                    self.schedule_convert_fn(item, ())
            else:
                emit_error(
                    item.loc,
                    f"unsupported in IR conversion: {item.__class__.__name__}")
            while self.fn_worklist:
                fn, params, name = self.fn_worklist.pop(0)
                self.convert_fn(fn, params, name)

        # Verify the module.
        self.module.verify()
        return self.module

    # Returns a version of the given `name` that is guaranteed to be unique in
    # the IR.
    def uniquify_name(self, name: str) -> str:
        idx = self.name_uniquification_ids.get(name)
        if idx is None:
            self.name_uniquification_ids[name] = 0
            return name

        while True:
            try_name = f"{name}_{idx}"
            idx += 1
            if try_name not in self.name_uniquification_ids:
                break
        self.name_uniquification_ids[name] = idx
        self.name_uniquification_ids[try_name] = 0
        return try_name

    def convert_mod(self, mod: ast.ModItem):
        name = self.uniquify_name(mod.name.spelling())
        op = self.builder.insert(SiModuleOp.with_empty_body(name, mod.loc))
        ip = self.builder.insertion_point
        self.builder.create_block_at_end(op.body)
        self.is_terminated_at = None
        for stmt in mod.stmts:
            self.convert_stmt(stmt)
        self.builder.insertion_point = ip

    def schedule_convert_fn(self, fn: ast.FnItem, params: Tuple[int,
                                                                ...]) -> str:
        if existing_fn_name := self.generated_fns.get((fn, params)):
            return existing_fn_name

        name = fn.name.spelling()
        for param in params:
            name += f"_{param}"
        name = self.uniquify_name(name)
        self.generated_fns[(fn, params)] = name
        self.fn_worklist.append((fn, params, name))
        return name

    def convert_fn(self, fn: ast.FnItem, params: Tuple[int, ...],
                   func_name: str):
        for func_param, value in zip(fn.params, params):
            assert func_param.free_param is not None
            self.params[func_param.free_param.num] = value

        # Determine the input and output types for the function.
        input_types = [
            self.convert_type(arg.fty, arg.ty.loc) for arg in fn.args
        ]
        output_types = []
        if not isinstance(fn.return_fty, TyUnitType):
            assert fn.return_ty is not None
            output_types.append(
                self.convert_type(fn.return_fty, fn.return_ty.loc))
        function_type = FunctionType.from_lists(input_types, output_types)

        # Build the function itself, and apply some name hints to the arguments.
        op = self.builder.insert(FuncOp(func_name, function_type, fn.loc))
        for block_arg, ast_arg in zip(op.body.blocks[0].args, fn.args):
            block_arg.name_hint = ast_arg.name.spelling()
            self.named_values[ast_arg] = block_arg

        # Convert the body statements.
        ip = self.builder.insertion_point
        self.builder.insertion_point = InsertPoint.at_end(op.body.blocks[0])
        self.is_terminated_at = None
        for stmt in fn.stmts:
            self.convert_stmt(stmt)

        # Insert a missing return for functions without results, or complain
        # about a missing return.
        if not self.is_terminated_at:
            if not output_types:
                self.builder.insert(ReturnOp([], fn.loc))
            else:
                emit_error(fn.loc,
                           f"missing return at end of `{fn.name.spelling()}`")
        self.builder.insertion_point = ip

    def convert_stmt(self, stmt: ast.Stmt):
        # Ignore statements after we have already returned and emit a warning.
        if self.is_terminated_at:
            if not self.warned_about_terminator_at:
                emit_info(
                    self.is_terminated_at,
                    f"any code following this expression is unreachable")
                emit_warning(stmt.full_loc, "unreachable statement")
                self.warned_about_terminator_at = self.is_terminated_at
            return

        if isinstance(stmt, ast.ExprStmt):
            self.convert_expr_rvalue(stmt.expr)
            return

        if isinstance(stmt, ast.InputStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            value = self.builder.insert(
                InputPortOp(stmt.name.spelling(), ty, stmt.loc)).value
            self.named_values[stmt] = value
            return

        if isinstance(stmt, ast.OutputStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            decl = self.builder.insert(
                OutputDeclOp(stmt.name.spelling(), ty, stmt.loc)).result
            self.named_values[stmt] = decl
            if stmt.expr:
                expr = self.convert_expr_rvalue(stmt.expr)
                self.builder.insert(AssignOp(decl, expr, stmt.expr.loc))
            return

        if isinstance(stmt, ast.LetStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            decl = self.builder.insert(
                VarDeclOp(stmt.name.spelling(), ty, stmt.loc)).result
            self.named_values[stmt] = decl
            if stmt.expr:
                expr = self.convert_expr_rvalue(stmt.expr)
                self.builder.insert(AssignOp(decl, expr, stmt.expr.loc))
            return

        if isinstance(stmt, ast.ReturnStmt):
            self.is_terminated_at = stmt.loc
            if not stmt.expr:
                self.builder.insert(ReturnOp([], stmt.loc))
                return
            expr = self.convert_expr_rvalue(stmt.expr)
            self.builder.insert(ReturnOp([expr], stmt.loc))
            return

        if isinstance(stmt, ast.AssignStmt):
            dst = self.convert_expr_lvalue(stmt.lhs)
            src = self.convert_expr_rvalue(stmt.rhs)
            self.builder.insert(AssignOp(dst, src, stmt.loc))
            return

        # No idea how to convert this statement.
        emit_error(stmt.loc,
                   f"unsupported in IR conversion: {stmt.__class__.__name__}")

    def convert_expr_lvalue(self, expr: ast.Expr) -> SSAValue:
        value = self.try_convert_expr_lvalue(expr)
        if value is not None:
            return value
        emit_error(expr.loc, f"expression is not assignable")

    def try_convert_expr_lvalue(self, expr: ast.Expr) -> SSAValue | None:
        if isinstance(expr, ast.IdentExpr):
            target = expr.binding.get()
            named_value = self.named_values[target]
            if isinstance(target, (ast.LetStmt, ast.OutputStmt)):
                return named_value
            return None

        if isinstance(expr, ast.TupleFieldExpr):
            if value := self.try_convert_expr_lvalue(expr.target):
                return self.builder.insert(
                    TupleGetRefOp(value, expr.field, expr.full_loc)).res
            return None

        if isinstance(expr, ast.UnaryExpr):
            # Getting an lvalue for a dereferenced `*x` is a no-op.
            if expr.op == ast.UnaryOp.DEREF:
                return self.convert_expr_rvalue(expr.arg)

        return None

    def convert_expr_rvalue(self, expr: ast.Expr) -> SSAValue:
        op: IRDLOperation

        if isinstance(expr, ast.IntLitExpr):
            assert isinstance(expr.fty, TyUIntType)
            if not isinstance(expr.fty.width, TyConstIntParam):
                emit_error(
                    expr.loc,
                    f"`{expr.loc.spelling}` has unknown width {expr.fty.width}"
                )
            return self.builder.insert(
                ConstantOp.from_value(expr.value, expr.fty.width.value,
                                      expr.loc)).result

        if isinstance(expr, ast.UnitLitExpr):
            return self.builder.insert(ConstantUnitOp(expr.loc)).result

        if isinstance(expr, ast.IdentExpr):
            target = expr.binding.get()
            value = self.named_values[target]
            if isinstance(target, (ast.LetStmt, ast.OutputStmt)):
                value = self.builder.insert(DerefOp(value, expr.loc)).result
            return value

        if isinstance(expr, ast.ParenExpr):
            return self.convert_expr_rvalue(expr.expr)

        if isinstance(expr, ast.TupleExpr):
            fields = [self.convert_expr_rvalue(field) for field in expr.fields]
            op = self.builder.insert(TupleCreateOp(fields, expr.loc))
            return op.results[0]

        if isinstance(expr, ast.TupleFieldExpr):
            op = self.builder.insert(
                TupleGetOp(self.convert_expr_rvalue(expr.target), expr.field,
                           expr.full_loc))
            return op.results[0]

        if isinstance(expr, ast.UnaryExpr):
            # Taking a reference `&x` is a no-op if the `x` can be lowered to an
            # lvalue. A variable declaration for example already is an reference
            # which we can return immediately.
            if expr.op == ast.UnaryOp.REF:
                if maybe_arg := self.try_convert_expr_lvalue(expr.arg):
                    return maybe_arg
                arg = self.convert_expr_rvalue(expr.arg)
                return self.builder.insert(RefOp(arg, expr.loc)).result

            arg = self.convert_expr_rvalue(expr.arg)

            if expr.op == ast.UnaryOp.NEG:
                op = self.builder.insert(NegOp(arg, expr.loc))
            elif expr.op == ast.UnaryOp.NOT:
                op = self.builder.insert(NotOp(arg, expr.loc))
            elif expr.op == ast.UnaryOp.DEREF:
                op = self.builder.insert(DerefOp(arg, expr.loc))
            else:
                emit_error(expr.loc,
                           f"unsupported unary operator `{expr.op.name}`")

            return op.results[0]

        if isinstance(expr, ast.BinaryExpr):
            lhs = self.convert_expr_rvalue(expr.lhs)
            rhs = self.convert_expr_rvalue(expr.rhs)

            if expr.op == ast.BinaryOp.ADD:
                op = self.builder.insert(AddOp(lhs, rhs, expr.loc))
            elif expr.op == ast.BinaryOp.SUB:
                op = self.builder.insert(SubOp(lhs, rhs, expr.loc))
            else:
                emit_error(expr.loc,
                           f"unsupported binary operator `{expr.op.name}`")

            return op.results[0]

        if isinstance(expr, ast.CallExpr):
            return self.convert_call_expr(expr)

        if isinstance(expr, ast.FieldCallExpr):
            return self.convert_field_call_expr(expr)

        # No idea how to convert this expression.
        emit_error(expr.loc,
                   f"unsupported in IR conversion: {expr.__class__.__name__}")

    def convert_call_expr(self, expr: ast.CallExpr) -> SSAValue:
        if target := expr.binding.target:
            assert isinstance(target, ast.FnItem)
            assert expr.call_params is not None
            params = []
            for func_param, call_param in zip(target.params, expr.call_params):
                if not isinstance(call_param, TyConstIntParam):
                    emit_error(
                        expr.loc,
                        f"parameter `{func_param.name.spelling()}` is not a constant integer"
                    )
                params.append(call_param.value)
            func_name = self.schedule_convert_fn(target, tuple(params))
            args = [self.convert_expr_rvalue(arg) for arg in expr.args]
            if isinstance(expr.fty, TyUnitType):
                self.builder.insert(CallOp(func_name, args, [], expr.loc))
                return self.builder.insert(ConstantUnitOp(expr.loc)).result
            else:
                results = [self.convert_type(expr.fty, expr.loc)]
                op = self.builder.insert(
                    CallOp(func_name, args, results, expr.loc))
                return op.results[0]

        # Handle builtin functions.
        name = expr.name.spelling()

        if name == "concat":
            args = [self.convert_expr_rvalue(arg) for arg in expr.args]
            return self.builder.insert(ConcatOp(args, expr.loc)).result

        if name == "wire":
            ty = self.convert_type(expr.args[0].fty, expr.args[0].loc)
            wire = self.builder.insert(WireDeclOp(ty, expr.loc)).result
            init = self.convert_expr_rvalue(expr.args[0])
            self.builder.insert(WireSetOp(wire, init, expr.loc))
            return wire

        if name == "reg":
            ty = self.convert_type(expr.args[1].fty, expr.args[1].loc)
            clock = self.convert_expr_rvalue(expr.args[0])
            reg = self.builder.insert(RegDeclOp(ty, clock, expr.loc)).result
            init = self.convert_expr_rvalue(expr.args[1])
            self.builder.insert(RegNextOp(reg, init, expr.loc))
            return reg

        emit_error(expr.name.loc, f"unknown function `{name}`")

    def convert_field_call_expr(self, expr: ast.FieldCallExpr) -> SSAValue:
        name = expr.name.spelling()

        if name == "bit":
            arg = self.convert_expr_rvalue(expr.target)
            offset = require_int_lit(expr.args[0])
            return self.builder.insert(ExtractOp(arg, offset, 1,
                                                 expr.loc)).result

        if name == "slice":
            offset = require_int_lit(expr.args[0])
            width = require_int_lit(expr.args[1])
            arg = self.convert_expr_rvalue(expr.target)
            return self.builder.insert(ExtractOp(arg, offset, width,
                                                 expr.loc)).result

        if name == "mux":
            cond = self.convert_expr_rvalue(expr.target)
            true = self.convert_expr_rvalue(expr.args[0])
            false = self.convert_expr_rvalue(expr.args[1])
            return self.builder.insert(MuxOp(cond, true, false,
                                             expr.loc)).result

        if name == "set":
            target = self.convert_expr_rvalue(expr.target)
            arg = self.convert_expr_rvalue(expr.args[0])
            self.builder.insert(WireSetOp(target, arg, expr.loc))
            return target

        if name == "get":
            target = self.convert_expr_rvalue(expr.target)
            return self.builder.insert(WireGetOp(target, expr.loc)).result

        if name == "next":
            target = self.convert_expr_rvalue(expr.target)
            arg = self.convert_expr_rvalue(expr.args[0])
            self.builder.insert(RegNextOp(target, arg, expr.loc))
            return target

        if name == "current":
            target = self.convert_expr_rvalue(expr.target)
            return self.builder.insert(RegCurrentOp(target, expr.loc)).result

        emit_error(expr.name.loc, f"unknown function `{name}`")


def require_int_lit(arg) -> int:
    assert isinstance(arg, ast.IntLitExpr)
    return arg.value


def convert_ast_to_ir(root: ast.Root) -> ModuleOp:
    return Converter().convert_root(root)
