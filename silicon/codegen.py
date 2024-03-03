from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
from silicon import ast
from silicon.diagnostics import *
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
        input_ports=[(stmt.name.spelling(), codegen_type(stmt.ty, cx))
                     for stmt in input_stmts],
        output_ports=[(stmt.name.spelling(), codegen_type(stmt.ty, cx))
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


def codegen_type(ty: ast.AstType, cx: CodegenContext) -> IRType:
    if isinstance(ty, ast.UIntType):
        return IntegerType.get_signless(ty.size)

    emit_error(ty.loc, f"type `{ty.loc.spelling()}` not supported for codegen")


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
        if expr.width is None:
            emit_error(
                expr.loc,
                f"integer literal `{expr.loc.spelling()}` requires a width")
        return hw.ConstantOp.create(IntegerType.get_signless(expr.width),
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
    if len(expr.args) == num: return
    name = expr.name.spelling()
    emit_error(
        expr.loc,
        f"argument mismatch: `{name}` takes {num} arguments, but got {len(expr.args)} instead"
    )


def require_int_lit(
    expr,
    idx: int,
    min: int | None = None,
    max: int | None = None,
) -> int:
    name = expr.name.spelling()
    arg = expr.args[idx]
    if not isinstance(arg, ast.IntLitExpr):
        emit_error(
            arg.loc,
            f"invalid argument: `{name}` requires argument {idx} be an integer literal"
        )
    if min is not None and arg.value < min:
        emit_error(
            arg.loc,
            f"invalid argument: `{name}` requires argument {idx} to have a minimum value of {min}, but got {arg.value} instead"
        )
    if max is not None and arg.value > max:
        emit_error(
            arg.loc,
            f"invalid argument: `{name}` requires argument {idx} to have a maximum value of {max}, but got {arg.value} instead"
        )
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
        require_num_args(expr, 1)
        init = codegen_expr(expr.args[0], cx)
        return hw.WireOp(init).result

    if name == "reg":
        require_num_args(expr, 2)
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
        require_num_args(expr, 1)
        arg = codegen_expr(expr.target, cx)
        offset = require_int_lit(expr,
                                 0,
                                 min=0,
                                 max=IntegerType(arg.type).width - 1)
        return comb.ExtractOp(IntegerType.get_signless(1), arg, offset).result

    if name == "slice":
        require_num_args(expr, 2)
        offset = require_int_lit(expr, 0, min=0)
        width = require_int_lit(expr, 1, min=0)
        arg = codegen_expr(expr.target, cx)
        arg_width = IntegerType(arg.type).width
        if offset + width > arg_width:
            emit_info(expr.target.loc,
                      f"sliced value is {arg_width} bits wide")
            emit_info(expr.args[0].loc | expr.args[1].loc,
                      f"but slice accesses bits {offset}..{offset+width}")
            emit_error(expr.loc, f"slice out of bounds")
        return comb.ExtractOp(IntegerType.get_signless(width), arg,
                              offset).result

    if name == "mux":
        require_num_args(expr, 2)
        target = codegen_expr(expr.target, cx)
        if target.type != IntegerType.get_signless(1):
            emit_error(expr.target.loc,
                       "mux requires select signal to be a single bit")

        lhs = codegen_expr(expr.args[0], cx)
        rhs = codegen_expr(expr.args[1], cx)
        if lhs.type != rhs.type:
            emit_error(
                expr.args[0].loc | expr.args[1].loc,
                f"mux requires both arguments to have the same type, but got {lhs.type} and {rhs.type} instead"
            )

        return comb.MuxOp(target, lhs, rhs).result

    if name == "set":
        require_num_args(expr, 1)
        target = codegen_expr(expr.target, cx)
        if not isinstance(target.owner,
                          IROperation) or target.owner.name != "hw.wire":
            emit_error(expr.target.loc,
                       "invalid receiver: `set` must be called on a wire")

        arg = codegen_expr(expr.args[0], cx)
        if target.type != arg.type:
            emit_error(
                expr.name.loc,
                f"type mismatch: wire is {target.type} but set value is {arg.type}"
            )

        # Update the input operand of the wire.
        target.owner.operands[0] = arg
        return target

    if name == "next":
        require_num_args(expr, 1)
        target = codegen_expr(expr.target, cx)
        if not isinstance(target.owner,
                          IROperation) or target.owner.name != "seq.compreg":
            emit_error(
                expr.target.loc,
                "invalid receiver: `next` must be called on a register")

        arg = codegen_expr(expr.args[0], cx)
        if target.type != arg.type:
            emit_error(
                expr.name.loc,
                f"type mismatch: register is {target.type} but next value is {arg.type}"
            )

        # Update the input operand of the register.
        target.owner.operands[0] = arg
        return target

    emit_error(expr.name.loc, f"unknown function `{name}`")
