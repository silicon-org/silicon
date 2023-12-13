from __future__ import annotations
from typing import *
from silicon import ast
from silicon.diagnostics import *
import circt
from circt.ir import Context, InsertionPoint, IntegerType, Location, Module
from circt.ir import Type as IRType, Value as IRValue
from circt.dialects import hw, comb


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
    # Collect the input and output ports.
    input_ports: List[Tuple[str, IRType]] = []
    output_ports: List[Tuple[str, IRType]] = []
    for stmt in mod.stmts:
        if isinstance(stmt, ast.InputStmt):
            input_ports.append((stmt.name.spelling(), codegen_type(stmt.ty)))
        elif isinstance(stmt, ast.OutputStmt):
            output_ports.append((stmt.name.spelling(), codegen_type(stmt.ty)))

    def build_body(module):
        output_values: Dict[str, IRValue] = {}

        for stmt in mod.stmts:
            if isinstance(stmt, ast.InputStmt):
                continue
            if isinstance(stmt, ast.OutputStmt):
                output_values[stmt.name.spelling()] = codegen_expr(stmt.expr)
                continue
            emit_error(stmt.loc, f"statement not supported for codegen")

        return output_values

    hw.HWModuleOp(name=mod.name.spelling(),
                  input_ports=input_ports,
                  output_ports=output_ports,
                  body_builder=build_body)


def codegen_type(ty: ast.AstType) -> IRType:
    if isinstance(ty, ast.UIntType):
        return IntegerType.get_signless(ty.size)

    emit_error(ty.loc, f"type `{ty.loc.spelling()}` not supported for codegen")


def codegen_expr(expr: ast.Expr) -> IRValue:
    if isinstance(expr, ast.IntLitExpr):
        return hw.ConstantOp.create(IntegerType.get_signless(expr.width),
                                    expr.value)

    if isinstance(expr, ast.BinaryExpr):
        if expr.op == ast.BinaryOp.ADD:
            return comb.AddOp([codegen_expr(expr.lhs), codegen_expr(expr.rhs)])
        if expr.op == ast.BinaryOp.SUB:
            return comb.SubOp(codegen_expr(expr.lhs), codegen_expr(expr.rhs))

    emit_error(
        expr.loc,
        f"expression `{expr.loc.spelling()}` not supported for codegen")
