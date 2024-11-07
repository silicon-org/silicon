from __future__ import annotations
from silicon import ast
from silicon import ir
from silicon.diagnostics import *
from silicon.source import *
from typing import *

__all__ = [
    "const_eval_ast",
    "const_eval_ir",
    "try_const_eval_ast",
    "try_const_eval_ir",
]


def try_const_eval_ir(value: ir.SSAValue,
                      must: bool = False) -> ir.IntegerAttr | None:
  assert isinstance(value.owner, ir.Operation)
  op: ir.Operation = value.owner

  if isinstance(op, ir.ConstantOp):
    return op.value

  if isinstance(op, ir.CmpOpBase):
    lhs = try_const_eval_ir(op.lhs, must)
    rhs = try_const_eval_ir(op.rhs, must)
    if lhs is None or rhs is None:
      return None
    if isinstance(op, ir.GtOp):
      bool_result = lhs.value.data > rhs.value.data
    elif isinstance(op, ir.GeqOp):
      bool_result = lhs.value.data >= rhs.value.data
    elif isinstance(op, ir.LtOp):
      bool_result = lhs.value.data < rhs.value.data
    elif isinstance(op, ir.LeqOp):
      bool_result = lhs.value.data <= rhs.value.data
    elif isinstance(op, ir.EqOp):
      bool_result = lhs.value.data == rhs.value.data
    elif isinstance(op, ir.NeqOp):
      bool_result = lhs.value.data != rhs.value.data
    else:
      if must:
        emit_error(ir.get_loc(op), "unsupported comparison operator")
      return None
    return ir.IntegerAttr(bool_result, ir.IntegerType(1))

  if must:
    emit_error(ir.get_loc(value), f"expression is not a constant")
  return None


def const_eval_ir(value: ir.SSAValue) -> ir.IntegerAttr:
  result = try_const_eval_ir(value, must=True)
  assert result is not None
  return result


class AstContext:
  values: Dict[ast.AstNode, int]

  def __init__(self, values: Dict[ast.AstNode, int] = {}):
    self.values = values


def try_const_eval_ast(
    cx: AstContext,
    node: ast.AstNode,
    must: bool = False,
) -> int | None:
  if (fixed := cx.values.get(node)) is not None:
    return fixed

  if isinstance(node, ast.ParenExpr):
    return try_const_eval_ast(cx, node.expr, must)

  if isinstance(node, ast.IdentExpr):
    target = node.binding.get()
    return try_const_eval_ast(cx, target, must)

  if isinstance(node, ast.IntLitExpr):
    return node.value

  if isinstance(node, ast.BinaryExpr):
    lhs = try_const_eval_ast(cx, node.lhs, must)
    rhs = try_const_eval_ast(cx, node.rhs, must)
    if lhs is None or rhs is None:
      return None

    if node.op == ast.BinaryOp.ADD:
      return int(lhs + rhs)
    if node.op == ast.BinaryOp.SUB:
      return int(lhs - rhs)
    if node.op == ast.BinaryOp.MUL:
      return int(lhs * rhs)
    if node.op == ast.BinaryOp.DIV:
      return int(lhs // rhs)
    if node.op == ast.BinaryOp.MOD:
      return int(lhs % rhs)
    if node.op == ast.BinaryOp.LT:
      return int(lhs < rhs)
    if node.op == ast.BinaryOp.LE:
      return int(lhs <= rhs)
    if node.op == ast.BinaryOp.GT:
      return int(lhs > rhs)
    if node.op == ast.BinaryOp.GE:
      return int(lhs >= rhs)
    if node.op == ast.BinaryOp.EQ:
      return int(lhs == rhs)
    if node.op == ast.BinaryOp.NE:
      return int(lhs != rhs)

    if must:
      emit_error(
          node.loc,
          f"operator `{node.loc.spelling()}` does not have a constant value")
    return None

  if must:
    emit_error(
        node.full_loc if hasattr(node, "full_loc") else node.loc,
        "expression is not a constant")
  return None


def const_eval_ast(cx: AstContext, node: ast.AstNode) -> int:
  result = try_const_eval_ast(cx, node, must=True)
  assert result is not None
  return result
