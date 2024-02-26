from __future__ import annotations
from typing import *
from dataclasses import dataclass, field
from silicon import ast
from silicon.diagnostics import *
import circt
from circt.ir import Context, InsertionPoint, IntegerType, Location, Module
from circt.ir import Type as IRType, Value as IRValue, Operation as IROperation
from circt.dialects import hw, comb
from circt.support import BackedgeBuilder, connect

__all__ = ["codegen"]


@dataclass
class CodegenContext:
    named_values: Dict[int, IRValue] = field(default_factory=dict)
    named_decls: Dict[int, IROperation] = field(default_factory=dict)


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
    input_indices: Dict[int, int] = dict()
    output_indices: Dict[int, int] = dict()
    input_ports: List[Tuple[str, IRType]] = []
    output_ports: List[Tuple[str, IRType]] = []

    for stmt in mod.stmts:
        if isinstance(stmt, ast.InputStmt):
            input_stmts.append(stmt)
            input_indices[id(stmt)] = len(input_ports)
            input_ports.append(
                (stmt.name.spelling(), codegen_type(stmt.ty, cx)))
        elif isinstance(stmt, ast.OutputStmt):
            output_stmts.append(stmt)
            output_indices[id(stmt)] = len(output_ports)
            output_ports.append(
                (stmt.name.spelling(), codegen_type(stmt.ty, cx)))

    # Create the module and entry block.
    hwmod = hw.HWModuleOp(name=mod.name.spelling(),
                          input_ports=input_ports,
                          output_ports=output_ports)
    block = hwmod.add_entry_block()

    # Add the module inputs to the codegen context. This allows identifier
    # expressions to refer to them.
    for stmt, arg in zip(input_stmts, block.arguments):
        cx.named_values[id(stmt)] = arg

    # Create placeholder values for the module outputs.
    with InsertionPoint(block):
        output_placeholders = [
            IROperation.create("builtin.unrealized_conversion_cast", [ty])
            for _, ty in output_ports
        ]

        for stmt, op in zip(output_stmts, output_placeholders):
            cx.named_values[id(stmt)] = op.result
            cx.named_decls[id(stmt)] = op

        output_op = hw.OutputOp([op.result for op in output_placeholders])

    with InsertionPoint(output_op):
        for stmt in mod.stmts:
            if isinstance(stmt, ast.InputStmt):
                continue
            if isinstance(stmt, ast.OutputStmt):
                value = codegen_expr(stmt.expr, cx)
                cx.named_values[id(stmt)] = value
                if op := cx.named_decls.get(id(stmt)):
                    op.result.replace_all_uses_with(value)
                    op.erase()
                continue
            emit_error(stmt.loc, f"statement not supported for codegen")


def codegen_type(ty: ast.AstType, cx: CodegenContext) -> IRType:
    if isinstance(ty, ast.UIntType):
        return IntegerType.get_signless(ty.size)

    emit_error(ty.loc, f"type `{ty.loc.spelling()}` not supported for codegen")


def codegen_expr(expr: ast.Expr, cx: CodegenContext) -> IRValue:
    if isinstance(expr, ast.IntLitExpr):
        return hw.ConstantOp.create(IntegerType.get_signless(expr.width),
                                    expr.value).result

    if isinstance(expr, ast.IdentExpr):
        target = expr.binding.get()
        if value := cx.named_values.get(id(target)):
            return value

        if isinstance(target, ast.InputStmt) or isinstance(
                target, ast.OutputStmt):
            ty = codegen_type(target.ty, cx)
            edge = IROperation.create("builtin.unrealized_conversion_cast",
                                      [ty])
            cx.named_decls[id(target)] = edge
            cx.named_values[id(target)] = edge.result
            return edge.result

        emit_error(
            expr.loc,
            f"name `{expr.name.spelling()}` cannot be used in an expression")

    if isinstance(expr, ast.BinaryExpr):
        if expr.op == ast.BinaryOp.ADD:
            return comb.AddOp(
                [codegen_expr(expr.lhs, cx),
                 codegen_expr(expr.rhs, cx)]).result
        if expr.op == ast.BinaryOp.SUB:
            return comb.SubOp(codegen_expr(expr.lhs, cx),
                              codegen_expr(expr.rhs, cx)).result

    emit_error(
        expr.loc,
        f"expression `{expr.loc.spelling()}` not supported for codegen")
