from __future__ import annotations

from typing import *
from dataclasses import dataclass

from silicon import ast
from silicon.diagnostics import *
from silicon.source import Loc, SourceFile

import xdsl
from xdsl.builder import Builder
from xdsl.dialects.builtin import (
    IntegerAttr,
    IntegerType,
    ModuleOp,
)
from xdsl.ir import (
    Dialect,
    OpResult,
    Operation,
    SSAValue,
)
from xdsl.irdl import (
    ConstraintVar,
    IRDLOperation,
    Operand,
    ParamAttrConstraint,
    attr_def,
    irdl_op_definition,
    operand_def,
    result_def,
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
    return Loc(SourceFile("unknown", ""), 0, 0)


#===------------------------------------------------------------------------===#
# Dialect
#===------------------------------------------------------------------------===#

# See the following for reference:
# - https://github.com/xdslproject/xdsl/blob/main/docs/Toy/toy/dialects/toy.py
# - https://github.com/xdslproject/xdsl/blob/main/docs/Toy/toy/frontend/ir_gen.py


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


@irdl_op_definition
class AddOp(IRDLOperation):
    name = "si.add"
    T = Annotated[IntegerType, ConstraintVar("T")]
    lhs: Operand = operand_def(T)
    rhs: Operand = operand_def(T)
    result: OpResult = result_def(T)
    assembly_format = "$lhs `,` $rhs `:` type($lhs) attr-dict"

    def __init__(self, lhs: SSAValue, rhs: SSAValue, loc: Loc):
        super().__init__(result_types=[lhs.type], operands=[lhs, rhs])
        self.loc = loc


SiliconDialect = Dialect("si", [], [])

#===------------------------------------------------------------------------===#
# Conversion from AST to IR
#===------------------------------------------------------------------------===#


@dataclass(init=False)
class Converter:
    module: ModuleOp
    builder: Builder

    def __init__(self):
        self.module = ModuleOp([])
        self.builder = Builder.at_end(self.module.body.blocks[0])

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
        for stmt in mod.stmts:
            self.convert_stmt(stmt)

    def convert_stmt(self, stmt: ast.Stmt):
        if isinstance(stmt, ast.ExprStmt):
            expr = self.convert_expr(stmt.expr)
            # Check that we can access the location of the expression.
            emit_info(get_loc(expr), "generated expression statement")
            return

        # No idea how to convert this statement.
        emit_error(stmt.loc,
                   f"unsupported in IR conversion: {stmt.__class__.__name__}")

    def convert_expr(self, expr: ast.Expr) -> SSAValue:
        if isinstance(expr, ast.IntLitExpr):
            if expr.width is None:
                emit_error(
                    expr.loc,
                    f"integer literal `{expr.loc.spelling()}` requires a width"
                )
            return self.builder.insert(
                ConstantOp.from_value(expr.value, expr.width, expr.loc)).result

        if isinstance(expr, ast.BinaryExpr):
            lhs = self.convert_expr(expr.lhs)
            rhs = self.convert_expr(expr.rhs)

            if expr.op == ast.BinaryOp.ADD:
                op = self.builder.insert(AddOp(lhs, rhs, expr.loc))
            else:
                emit_error(expr.loc,
                           f"unsupported binary operator `{expr.op.name}`")

            return op.result

        # No idea how to convert this expression.
        emit_error(expr.loc,
                   f"unsupported in IR conversion: {expr.__class__.__name__}")


def convert_ast_to_ir(root: ast.Root):
    print("converting to IR")
    ir = Converter().convert_root(root)
    print(ir)
