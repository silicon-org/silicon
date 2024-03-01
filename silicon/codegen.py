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
    assignable_wires: Dict[int, hw.WireOp] = field(default_factory=dict)


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
    let_stmts: List[ast.LetStmt] = []

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
            input_indices[id(stmt)] = len(input_ports)
            input_ports.append(
                (stmt.name.spelling(), codegen_type(stmt.ty, cx)))
        elif isinstance(stmt, ast.OutputStmt):
            check_port_name(stmt.name.spelling(), stmt)
            output_stmts.append(stmt)
            output_indices[id(stmt)] = len(output_ports)
            output_ports.append(
                (stmt.name.spelling(), codegen_type(stmt.ty, cx)))
        elif isinstance(stmt, ast.LetStmt):
            let_stmts.append(stmt)

    # Create the module and entry block.
    hwmod = hw.HWModuleOp(name=mod.name.spelling(),
                          input_ports=input_ports,
                          output_ports=output_ports)
    block = hwmod.add_entry_block()

    # Add the module inputs to the codegen context. This allows identifier
    # expressions to refer to them.
    for stmt, arg in zip(input_stmts, block.arguments):
        cx.named_values[id(stmt)] = arg

    placeholders: List[Tuple[ast.AstNode, IROperation]] = []

    with InsertionPoint(block):
        # Create placeholder values for the module outputs.
        output_values = []
        for stmt in output_stmts:
            ty = codegen_type(stmt.ty, cx)
            placeholder = IROperation.create(
                "builtin.unrealized_conversion_cast", [ty])
            placeholders.append((stmt, placeholder))
            wire = hw.WireOp(placeholder)
            cx.named_values[id(stmt)] = wire.result
            cx.assignable_wires[id(stmt)] = wire
            output_values.append(wire.result)

        # Create placeholder values for let bindings.
        for stmt in let_stmts:
            ty = codegen_type(stmt.ty, cx)
            placeholder = IROperation.create(
                "builtin.unrealized_conversion_cast", [ty])
            placeholders.append((stmt, placeholder))
            wire = hw.WireOp(placeholder, name=stmt.name.spelling())
            cx.named_values[id(stmt)] = wire.result
            cx.assignable_wires[id(stmt)] = wire

        # Emit the statements in the module body.
        for stmt in mod.stmts:
            codegen_stmt(stmt, cx)

        # Create the termiantor.
        hw.OutputOp(output_values)

    # Delete all placeholders. A placeholder that still has users is an
    # indication that a something is missing an assignment.
    for node, placeholder in placeholders:
        if list(placeholder.result.uses):
            if isinstance(node, ast.LetStmt):
                emit_error(
                    node.loc,
                    f"binding `{node.name.spelling()}` has not been assigned")
            if isinstance(node, ast.OutputStmt):
                emit_error(
                    node.loc,
                    f"output `{node.name.spelling()}` has not been assigned")
            emit_error(node.loc, f"declaration has not been assigned")
        placeholder.erase()

    # Canonicalize unnamed assignable wires away.
    for wire in cx.assignable_wires.values():
        if not wire.name:
            wire.result.replace_all_uses_with(wire.input)
            wire.erase()


def codegen_type(ty: ast.AstType, cx: CodegenContext) -> IRType:
    if isinstance(ty, ast.UIntType):
        return IntegerType.get_signless(ty.size)

    emit_error(ty.loc, f"type `{ty.loc.spelling()}` not supported for codegen")


def codegen_stmt(stmt: ast.Stmt, cx: CodegenContext) -> IRValue:
    if isinstance(stmt, ast.InputStmt):
        return

    if isinstance(stmt, ast.OutputStmt):
        if stmt.expr:
            value = codegen_expr(stmt.expr, cx)
            cx.assignable_wires[id(stmt)].operands[0] = value
        return

    if isinstance(stmt, ast.LetStmt):
        if stmt.expr:
            value = codegen_expr(stmt.expr, cx)
            cx.assignable_wires[id(stmt)].operands[0] = value
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

        if wire := cx.assignable_wires.get(id(lhs)):
            rhs = codegen_expr(stmt.rhs, cx)
            wire.operands[0] = rhs
            return

        emit_error(stmt.loc,
                   f"name `{stmt.lhs.name.spelling()}` is not assignable")

    emit_error(stmt.loc, f"statement not supported for codegen")


def codegen_expr(expr: ast.Expr, cx: CodegenContext) -> IRValue:
    if isinstance(expr, ast.IntLitExpr):
        return hw.ConstantOp.create(IntegerType.get_signless(expr.width),
                                    expr.value).result

    if isinstance(expr, ast.IdentExpr):
        target = expr.binding.get()
        if value := cx.named_values.get(id(target)):
            return value

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
