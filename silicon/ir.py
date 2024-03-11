from __future__ import annotations

from typing import Annotated, Dict, TypeVar, List
from dataclasses import dataclass

from silicon import ast
from silicon.diagnostics import *
from silicon.source import Loc, SourceFile
from silicon.ty import Type, UIntType

import xdsl
from xdsl.builder import Builder
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntAttr,
    IntegerType,
    ModuleOp,
    StringAttr,
)
from xdsl.ir import (
    Dialect,
    OpResult,
    Operation,
    Region,
    Block,
    SSAValue,
    TypeAttribute,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParamAttrConstraint,
    VarOperand,
    attr_def,
    region_def,
    prop_def,
    irdl_op_definition,
    operand_def,
    result_def,
    var_operand_def,
)
from xdsl.parser import Parser
from xdsl.printer import Printer
from xdsl.traits import (
    IsolatedFromAbove,
    SymbolOpInterface,
    NoTerminator,
)
from xdsl.utils.exceptions import VerifyException

__all__ = ["convert_ast_to_ir", "SiliconDialect"]


def get_loc(value: SSAValue) -> Loc:
    if hasattr(value, "loc"):
        assert isinstance(value.loc, Loc)
        return value.loc
    if hasattr(value.owner, "loc"):
        assert isinstance(value.owner.loc, Loc)
        return value.owner.loc
    return Loc.unknown()


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


#===------------------------------------------------------------------------===#
# Statements
#===------------------------------------------------------------------------===#


class DeclOpBase(IRDLOperation):
    decl_name: StringAttr = attr_def(StringAttr)
    result: OpResult = result_def()
    assembly_format = "$decl_name `:` type($result) attr-dict"

    def __init__(self, name: str, ty: TypeAttribute, loc: Loc):
        super().__init__(result_types=[ty],
                         attributes={"decl_name": StringAttr(name)})
        self.loc = loc


@irdl_op_definition
class InputDeclOp(DeclOpBase):
    name = "si.input_decl"


@irdl_op_definition
class OutputDeclOp(DeclOpBase):
    name = "si.output_decl"


@irdl_op_definition
class VarDeclOp(DeclOpBase):
    name = "si.var_decl"


@irdl_op_definition
class AssignOp(IRDLOperation):
    name = "si.assign"
    T = Annotated[TypeAttribute, ConstraintVar("T")]
    dst: Operand = operand_def(T)
    src: Operand = operand_def(T)
    assembly_format = "$dst `,` $src `:` type($dst) attr-dict"

    def __init__(self, dst: SSAValue, src: SSAValue, loc: Loc):
        super().__init__(operands=[dst, src])
        self.loc = loc


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#


@irdl_op_definition
class ConstantOp(IRDLOperation):
    name = "si.constant"
    T = Annotated[IntegerType, ConstraintVar("T")]
    value: IntegerAttr = attr_def(IntegerAttr[T])
    result: OpResult = result_def(T)
    assembly_format = "$value `:` type($result) attr-dict"

    def __init__(self, value: IntegerAttr, loc: Loc):
        super().__init__(
            result_types=[value.type],
            attributes={"value": value},
        )
        self.loc = loc

    @staticmethod
    def from_value(value: int, width: int, loc: Loc) -> ConstantOp:
        return ConstantOp(IntegerAttr(value, IntegerType(width)), loc)


class UnaryOpBase(IRDLOperation):
    T = Annotated[IntegerType, ConstraintVar("T")]
    arg: Operand = operand_def(T)
    result: OpResult = result_def(T)
    assembly_format = "$arg `:` type($arg) attr-dict"

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

    def __init__(self, condition: SSAValue, true_value: SSAValue,
                 false_value: SSAValue, loc: Loc):
        super().__init__(result_types=[true_value.type],
                         operands=[condition, true_value, false_value])
        self.loc = loc


#===------------------------------------------------------------------------===#
# Conversion from AST to IR
#===------------------------------------------------------------------------===#


@dataclass(init=False)
class Converter:
    module: ModuleOp
    builder: Builder
    named_values: Dict[ast.AstNode, SSAValue]

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])
        self.named_values = dict()

    def convert_type(self, ty: Type | None, loc: Loc) -> TypeAttribute:
        assert ty is not None, "type checking should have assigned types"

        if isinstance(ty, UIntType):
            return IntegerType(ty.width)

        emit_error(loc,
                   f"unsupported in IR conversion: {ty.__class__.__name__}")

    def convert_root(self, root: ast.Root) -> ModuleOp:
        for item in root.items:
            if isinstance(item, ast.ModItem):
                self.convert_mod(item)
            else:
                emit_error(
                    item.loc,
                    f"unsupported in IR conversion: {item.__class__.__name__}")

        # Verify the module.
        self.module.verify()
        return self.module

    def convert_mod(self, mod: ast.ModItem):
        op = self.builder.insert(
            SiModuleOp.with_empty_body(mod.name.spelling(), mod.loc))
        ip = self.builder.insertion_point
        self.builder.create_block_at_end(op.body)
        for stmt in mod.stmts:
            self.convert_stmt(stmt)
        self.builder.insertion_point = ip

    def convert_stmt(self, stmt: ast.Stmt):
        if isinstance(stmt, ast.ExprStmt):
            self.convert_expr(stmt.expr)
            return

        if isinstance(stmt, ast.InputStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            value = self.builder.insert(
                InputDeclOp(stmt.name.spelling(), ty, stmt.loc)).result
            self.named_values[stmt] = value
            return

        if isinstance(stmt, ast.OutputStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            value = self.builder.insert(
                OutputDeclOp(stmt.name.spelling(), ty, stmt.loc)).result
            self.named_values[stmt] = value
            if stmt.expr:
                expr = self.convert_expr(stmt.expr)
                self.builder.insert(AssignOp(value, expr, stmt.expr.loc))
            return

        if isinstance(stmt, ast.LetStmt):
            ty = self.convert_type(stmt.fty, stmt.loc)
            value = self.builder.insert(
                VarDeclOp(stmt.name.spelling(), ty, stmt.loc)).result
            self.named_values[stmt] = value
            if stmt.expr:
                expr = self.convert_expr(stmt.expr)
                self.builder.insert(AssignOp(value, expr, stmt.expr.loc))
            return

        if isinstance(stmt, ast.AssignStmt):
            dst = self.convert_expr(stmt.lhs)
            src = self.convert_expr(stmt.rhs)
            self.builder.insert(AssignOp(dst, src, stmt.loc))
            return

        # No idea how to convert this statement.
        emit_error(stmt.loc,
                   f"unsupported in IR conversion: {stmt.__class__.__name__}")

    def convert_expr(self, expr: ast.Expr) -> SSAValue:
        op: IRDLOperation

        if isinstance(expr, ast.IntLitExpr):
            assert isinstance(expr.fty, UIntType)
            return self.builder.insert(
                ConstantOp.from_value(expr.value, expr.fty.width,
                                      expr.loc)).result

        if isinstance(expr, ast.IdentExpr):
            target = expr.binding.get()
            # TODO: This should probably be a `si.read_var` op or similar.
            return self.named_values[target]

        if isinstance(expr, ast.UnaryExpr):
            arg = self.convert_expr(expr.arg)

            if expr.op == ast.UnaryOp.NEG:
                op = self.builder.insert(NegOp(arg, expr.loc))
            elif expr.op == ast.UnaryOp.NOT:
                op = self.builder.insert(NotOp(arg, expr.loc))
            else:
                emit_error(expr.loc,
                           f"unsupported unary operator `{expr.op.name}`")

            return op.results[0]

        if isinstance(expr, ast.BinaryExpr):
            lhs = self.convert_expr(expr.lhs)
            rhs = self.convert_expr(expr.rhs)

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
        name = expr.name.spelling()

        if name == "concat":
            args = [self.convert_expr(arg) for arg in expr.args]
            return self.builder.insert(ConcatOp(args, expr.loc)).result

        # if name == "wire":
        #     init = codegen_expr(expr.args[0], cx)
        #     return hw.WireOp(init).result

        # if name == "reg":
        #     clock = codegen_expr(expr.args[0], cx)
        #     init = codegen_expr(expr.args[1], cx)
        #     return seq.CompRegOp(init.type, init,
        #                          seq.ToClockOp(clock).result).result

        emit_error(expr.name.loc, f"unknown function `{name}`")

    def convert_field_call_expr(self, expr: ast.FieldCallExpr) -> SSAValue:
        name = expr.name.spelling()

        if name == "bit":
            arg = self.convert_expr(expr.target)
            offset = require_int_lit(expr.args[0])
            return self.builder.insert(ExtractOp(arg, offset, 1,
                                                 expr.loc)).result

        if name == "slice":
            offset = require_int_lit(expr.args[0])
            width = require_int_lit(expr.args[1])
            arg = self.convert_expr(expr.target)
            return self.builder.insert(ExtractOp(arg, offset, width,
                                                 expr.loc)).result

        if name == "mux":
            cond = self.convert_expr(expr.target)
            true = self.convert_expr(expr.args[0])
            false = self.convert_expr(expr.args[1])
            return self.builder.insert(MuxOp(cond, true, false,
                                             expr.loc)).result

        # if name == "set":
        #     target = codegen_expr(expr.target, cx)
        #     arg = codegen_expr(expr.args[0], cx)
        #     # Update the input operand of the wire.
        #     target.owner.operands[0] = arg
        #     return target

        # if name == "get":
        #     return codegen_expr(expr.target, cx)

        # if name == "next":
        #     target = codegen_expr(expr.target, cx)
        #     arg = codegen_expr(expr.args[0], cx)
        #     # Update the input operand of the register.
        #     target.owner.operands[0] = arg
        #     return target

        # if name == "current":
        #     return codegen_expr(expr.target, cx)

        emit_error(expr.name.loc, f"unknown function `{name}`")


def require_int_lit(arg) -> int:
    assert isinstance(arg, ast.IntLitExpr)
    return arg.value


def convert_ast_to_ir(root: ast.Root):
    ir = Converter().convert_root(root)
    print(ir)