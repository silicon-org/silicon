from __future__ import annotations

from typing import Dict, List, Optional
from dataclasses import dataclass, field

from silicon import ast
from silicon.diagnostics import *
from silicon.source import Loc
from silicon.ty import Type, UIntType, WireType, RegType

import circt
from circt.ir import Context, InsertionPoint, IntegerType, Location, Module
from circt.ir import Type as IRType, Value as IRValue, Operation as IROperation
from circt.dialects import hw, comb, seq
from circt.support import BackedgeBuilder, connect

__all__ = ["codegen"]


@dataclass
class CodegenContext:
    named_values: Dict[int, IRValue] = field(default_factory=dict)
    assigned_values: Dict[int, IRValue] = field(default_factory=dict)


def codegen(root: ast.Root):
    with Context() as ctx, Location.unknown():
        circt.register_dialects(ctx)
        module = Module.create()
        with InsertionPoint(module.body):
            for item in root.items:
                if isinstance(item, ast.ModItem):
                    codegen_mod(item)
                else:
                    emit_error(item.loc, "not supported for codegen")
        print(module)
        module.operation.verify()


def codegen_mod(mod: ast.ModItem):
    cx = CodegenContext()

    # Collect the input and output ports.
    input_stmts: List[ast.InputStmt] = []
    output_stmts: List[ast.OutputStmt] = []
    used_names: Dict[str, ast.AstNode] = dict()

    def check_port_name(name: str, node: ast.AstNode):
        if existing := used_names.get(name):
            emit_info(existing.loc,
                      f"previous definition of port `{name}` was here")
            emit_error(node.loc, f"port `{name}` already defined")
        used_names[name] = node

    for stmt in mod.stmts:
        if isinstance(stmt, ast.InputStmt):
            check_port_name(stmt.name.spelling(), stmt)
            input_stmts.append(stmt)
        elif isinstance(stmt, ast.OutputStmt):
            check_port_name(stmt.name.spelling(), stmt)
            output_stmts.append(stmt)

    # Create the module and entry block.
    hwmod = hw.HWModuleOp(
        name=mod.name.spelling(),
        input_ports=[(stmt.name.spelling(),
                      codegen_type(stmt.fty, cx, stmt.loc))
                     for stmt in input_stmts],
        output_ports=[(stmt.name.spelling(),
                       codegen_type(stmt.fty, cx, stmt.loc))
                      for stmt in output_stmts],
    )
    block = hwmod.add_entry_block()

    # Add the module inputs to the codegen context. This allows identifier
    # expressions to refer to them.
    for stmt, arg in zip(input_stmts, block.arguments):
        cx.named_values[id(stmt)] = arg

    with InsertionPoint(block):
        # Emit the statements in the module body.
        for stmt in mod.stmts:
            codegen_stmt(stmt, cx)

        # Collect the output port assignments to build the terminator.
        output_values = []
        for stmt in output_stmts:
            value = cx.assigned_values.get(id(stmt))
            if not value:
                emit_error(
                    stmt.loc,
                    f"output `{stmt.name.spelling()}` has not been assigned")
            output_values.append(value)

        # Create the termiantor.
        hw.OutputOp(output_values)


def codegen_type(ty: Optional[Type], cx: CodegenContext, loc: Loc) -> IRType:
    assert ty is not None, "type checking should have assigned types"

    if isinstance(ty, UIntType):
        return IntegerType.get_signless(ty.width)
    if isinstance(ty, WireType) or isinstance(ty, RegType):
        return codegen_type(ty.inner, cx, loc)

    emit_error(loc, f"type `{ty}` not supported for codegen")


def codegen_stmt(stmt: ast.Stmt, cx: CodegenContext) -> IRValue:
    if isinstance(stmt, ast.InputStmt):
        return

    if isinstance(stmt, ast.OutputStmt):
        if stmt.expr:
            cx.assigned_values[id(stmt)] = codegen_expr(stmt.expr, cx)
        return

    if isinstance(stmt, ast.LetStmt):
        if stmt.expr:
            cx.assigned_values[id(stmt)] = codegen_expr(stmt.expr, cx)
        return

    if isinstance(stmt, ast.ExprStmt):
        codegen_expr(stmt.expr, cx)
        return

    if isinstance(stmt, ast.AssignStmt):
        if not isinstance(stmt.lhs, ast.IdentExpr):
            emit_error(
                stmt.lhs.loc,
                f"expression `{stmt.lhs.loc.spelling()}` cannot appear on left-hand side of `=`"
            )

        lhs = stmt.lhs.binding.get()
        if not isinstance(lhs, ast.OutputStmt) and not isinstance(
                lhs, ast.LetStmt):
            emit_info(lhs.loc,
                      f"name `{stmt.lhs.name.spelling()}` defined here")
            emit_error(
                stmt.lhs.loc,
                f"expression `{stmt.lhs.loc.spelling()}` cannot be assigned")

        cx.assigned_values[id(lhs)] = codegen_expr(stmt.rhs, cx)
        return

    emit_error(stmt.loc, f"statement not supported for codegen")


def codegen_expr(expr: ast.Expr, cx: CodegenContext) -> IRValue:
    if isinstance(expr, ast.IntLitExpr):
        assert isinstance(expr.fty, UIntType)
        return hw.ConstantOp.create(IntegerType.get_signless(expr.fty.width),
                                    expr.value).result

    if isinstance(expr, ast.IdentExpr):
        target = expr.binding.get()

        if value := cx.named_values.get(id(target)):
            return value
        if value := cx.assigned_values.get(id(target)):
            return value

        emit_error(expr.loc,
                   f"`{expr.name.spelling()}` is unassigned at this point")

    if isinstance(expr, ast.UnaryExpr):
        if expr.op == ast.UnaryOp.NEG:
            arg = codegen_expr(expr.arg, cx)
            zero = hw.ConstantOp.create(arg.type, 0)
            return comb.SubOp(zero, arg).result

        if expr.op == ast.UnaryOp.NOT:
            arg = codegen_expr(expr.arg, cx)
            ones = hw.ConstantOp.create(arg.type, -1)
            return comb.XorOp([ones, arg]).result

    if isinstance(expr, ast.BinaryExpr):
        if expr.op == ast.BinaryOp.ADD:
            return comb.AddOp(
                [codegen_expr(expr.lhs, cx),
                 codegen_expr(expr.rhs, cx)]).result

        if expr.op == ast.BinaryOp.SUB:
            return comb.SubOp(codegen_expr(expr.lhs, cx),
                              codegen_expr(expr.rhs, cx)).result

    if isinstance(expr, ast.CallExpr):
        return codegen_call_expr(expr, cx)

    if isinstance(expr, ast.FieldCallExpr):
        return codegen_field_call_expr(expr, cx)

    emit_error(
        expr.loc,
        f"expression `{expr.loc.spelling()}` not supported for codegen")


def require_num_args(expr, num: int):
    if len(expr.args) != num:
        emit_error(expr.loc, "typeck should have checked arg count")


def require_int_lit(arg: ast.Expr) -> int:
    if not isinstance(arg, ast.IntLitExpr):
        emit_error(arg.loc, "typeck should have checked arg is int literal")
    return arg.value


def codegen_call_expr(
    expr: ast.CallExpr,
    cx: CodegenContext,
) -> IRValue:
    name = expr.name.spelling()

    if name == "concat":
        args = [codegen_expr(arg, cx) for arg in expr.args]
        return comb.ConcatOp(args).result

    if name == "wire":
        init = codegen_expr(expr.args[0], cx)
        return hw.WireOp(init).result

    if name == "reg":
        clock = codegen_expr(expr.args[0], cx)
        init = codegen_expr(expr.args[1], cx)
        return seq.CompRegOp(init.type, init,
                             seq.ToClockOp(clock).result).result

    emit_error(expr.name.loc, f"unknown function `{name}`")


def codegen_field_call_expr(
    expr: ast.FieldCallExpr,
    cx: CodegenContext,
) -> IRValue:
    name = expr.name.spelling()

    if name == "bit":
        arg = codegen_expr(expr.target, cx)
        offset = require_int_lit(expr.args[0])
        return comb.ExtractOp(IntegerType.get_signless(1), arg, offset).result

    if name == "slice":
        offset = require_int_lit(expr.args[0])
        width = require_int_lit(expr.args[1])
        arg = codegen_expr(expr.target, cx)
        return comb.ExtractOp(IntegerType.get_signless(width), arg,
                              offset).result

    if name == "mux":
        target = codegen_expr(expr.target, cx)
        lhs = codegen_expr(expr.args[0], cx)
        rhs = codegen_expr(expr.args[1], cx)
        return comb.MuxOp(target, lhs, rhs).result

    if name == "set":
        target = codegen_expr(expr.target, cx)
        arg = codegen_expr(expr.args[0], cx)
        # Update the input operand of the wire.
        target.owner.operands[0] = arg
        return target

    if name == "get":
        return codegen_expr(expr.target, cx)

    if name == "next":
        target = codegen_expr(expr.target, cx)
        arg = codegen_expr(expr.args[0], cx)
        # Update the input operand of the register.
        target.owner.operands[0] = arg
        return target

    if name == "current":
        return codegen_expr(expr.target, cx)

    emit_error(expr.name.loc, f"unknown function `{name}`")
