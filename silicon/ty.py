from __future__ import annotations

import z3
from math import log2, floor
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from itertools import count

from silicon.diagnostics import *
from silicon.source import Loc
from silicon import ast

__all__ = ["typeck"]


# Compute the minimum number of bits required to represent a given integer.
def min_bits_for_int(value: int) -> int:
  return floor(log2(value)) + 1 if value > 0 else 0


#===------------------------------------------------------------------------===#
# Parameters
#===------------------------------------------------------------------------===#


@dataclass(eq=False)
class IntParam:
  # The type-checking context which created this parameter.
  cx: Typeck
  pass


@dataclass
class ConstIntParam(IntParam):
  value: int

  def __str__(self) -> str:
    return f"{self.value}"

  def __eq__(self, other) -> bool:
    return isinstance(other, ConstIntParam) and self.value == other.value

  def __hash__(self) -> int:
    return hash(self.value)


@dataclass(eq=False)
class FreeIntParam(IntParam):
  name_hint: Optional[str] = None
  num: int = field(default_factory=count().__next__)

  def __str__(self) -> str:
    return self.name_hint or f"@{self.num}"


@dataclass(eq=False)
class InferrableIntParam(IntParam):
  num: int = field(default_factory=count().__next__)

  # The value this parameter has been inferred to.
  inferred: Optional[IntParam] = None

  # The minimum number of bits needed by the literal that generated this
  # parameter. If the parameter remains uninferred, this width shall be used
  # instead.
  width_hint: int = 0

  def __str__(self) -> str:
    return f"?{self.num}"


@dataclass(eq=False)
class DerivedIntParam(IntParam):
  expr: ast.Expr

  def __str__(self) -> str:
    return format_constraint(self.cx, self.expr)


#===------------------------------------------------------------------------===#
# Type Hierarchy
#===------------------------------------------------------------------------===#


@dataclass(eq=False)
class Type:
  pass


@dataclass
class UnitType(Type):

  def __str__(self) -> str:
    return "()"


# The type of an integer literal. This is essentially an inferrable integer type
# with a hint about the concrete literal. It can be inferred to a generic
# integer type, or a concrete fixed-width integer.
@dataclass(eq=False)
class LiteralIntType(Type):
  # The minimum number of bits needed by the literal that generated this type.
  # If the type remains uninferred, this width is used to construct a UIntType.
  width_hint: int

  # The concrete type this has been inferred to.
  inferred: LiteralIntType | GenericIntType | UIntType | None = None

  def __str__(self) -> str:
    return "{literal}"


@dataclass(eq=False)
class GenericIntType(Type):
  # The concrete integer type that has been inferred.
  inferred: Optional[Type] = None

  def __str__(self) -> str:
    return "{integer}"


@dataclass
class TupleType(Type):
  fields: List[Type]

  def __str__(self) -> str:
    return "(" + ", ".join(map(str, self.fields)) + ")"


@dataclass
class UIntType(Type):
  width: IntParam

  def __str__(self) -> str:
    return f"uint<{self.width}>"


@dataclass
class WireType(Type):
  inner: Type

  def __str__(self) -> str:
    return f"Wire<{self.inner}>"


@dataclass
class RegType(Type):
  inner: Type

  def __str__(self) -> str:
    return f"Reg<{self.inner}>"


@dataclass
class RefType(Type):
  inner: Type

  def __str__(self) -> str:
    return f"&{self.inner}"


@dataclass(eq=False)
class InferrableType(Type):
  # A unique integer identifier.
  num: int = field(default_factory=count().__next__)

  # The type this uint has been inferred to.
  inferred: Type | None = None

  def __str__(self) -> str:
    return f"?{self.num}"


#===------------------------------------------------------------------------===#
# Constraints
#===------------------------------------------------------------------------===#


@dataclass(eq=False)
class Constraint:
  lhs: IntParam
  relation: str
  rhs: ConstraintFn | IntParam | ast.Expr
  loc: Loc


@dataclass(eq=False)
class ConstraintFn:
  op: str
  arg: ConstraintFn | IntParam | ast.Expr


#===------------------------------------------------------------------------===#
# Type Checking and Inference
#===------------------------------------------------------------------------===#


def require_num_args(expr, num: int):
  if len(expr.args) == num:
    return
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


class Typeck:
  root: ast.ModItem | ast.FnItem
  params: Dict[ast.AstNode, IntParam]
  solver: z3.Solver
  solver_params: Dict[IntParam, z3.Int]
  solver_nodes: Dict[ast.Expr, z3.Ast]
  constraints: list

  def __init__(self, root: ast.ModItem | ast.FnItem) -> None:
    self.root = root
    self.params = dict()
    self.solver = z3.Solver()
    self.solver_params = dict()
    self.solver_nodes = dict()
    self.constraints = []

  # Convert an AST type node into an actual type.
  def convert_ast_type(self, ty: ast.AstType) -> Type:
    if isinstance(ty, ast.UnitType):
      return UnitType()
    if isinstance(ty, ast.TupleType):
      return TupleType([self.convert_ast_type(field) for field in ty.fields])
    if isinstance(ty, ast.UIntType):
      size_ty = self.typeck_expr(ty.size)
      self.unify_types(GenericIntType(), size_ty, ty.size.full_loc)
      if isinstance(ty.size, ast.IntLitExpr):
        return UIntType(ConstIntParam(self, ty.size.value))
      if isinstance(ty.size, ast.IdentExpr):
        binding = ty.size.binding.get()
        param = self.params.get(binding)
        if not param:
          emit_info(binding.loc, f"`{ty.size.name.spelling()}` defined here")
          emit_error(ty.size.loc, f"invalid parameter for `uint` width")
        return UIntType(param)
      if isinstance(ty.size, ast.Expr):
        return UIntType(DerivedIntParam(self, ty.size))
      emit_error(ty.size.loc, f"unsupported expression for `uint` width")
    if isinstance(ty, ast.WireType):
      return WireType(self.convert_ast_type(ty.inner))
    if isinstance(ty, ast.RegType):
      return RegType(self.convert_ast_type(ty.inner))
    if isinstance(ty, ast.RefType):
      return RefType(self.convert_ast_type(ty.inner))
    emit_error(ty.loc, f"unsupported type: `{ty.loc.spelling()}`")

  # Type-check just the signature of the `root` node.
  def typeck_signature(self):
    if isinstance(self.root, ast.FnItem):
      # Declare the parameters.
      for param in self.root.params:
        param.free_param = FreeIntParam(self, param.name.spelling())
        self.params[param] = param.free_param
        solver_param = z3.Int(param.name.spelling())
        self.solver.add(solver_param >= 0)
        self.solver_params[param.free_param] = solver_param

      # Apply where clauses.
      for expr in self.root.wheres:
        self.typeck_where(expr)

      # Check the arguments.
      for arg in self.root.args:
        arg.fty = self.convert_ast_type(arg.ty)

      # Check the return type.
      self.root.return_fty = self.convert_ast_type(
          self.root.return_ty) if self.root.return_ty else UnitType()

  def typeck_where(self, expr: ast.Expr):
    self.typeck_expr(expr)
    solver_expr = convert_to_solver_expr(self, expr)

    # Check if the clause is trivially true.
    self.solver.push()
    self.solver.add(~solver_expr)
    if self.solver.check() == z3.unsat:
      emit_warning(expr.loc, "constraint always holds and can be removed")
    self.solver.pop()

    # Check if the clause renders the constraints unsatisfiable.
    self.solver.add(solver_expr)
    if self.solver.check() == z3.unsat:
      emit_error(expr.loc, "constraint is unsatisfiable")

  # Type-check the entire AST starting at the `root` node.
  def typeck(self):
    for stmt in self.root.stmts:
      self.typeck_stmt(stmt)

    check_constraints(self)
    self.finalize(self.root)

  def typeck_stmt(self, stmt: ast.Stmt):
    if isinstance(stmt, ast.ExprStmt):
      self.typeck_expr(stmt.expr)
      return

    if isinstance(stmt, ast.InputStmt):
      stmt.fty = self.convert_ast_type(stmt.ty)
      return

    if isinstance(stmt, ast.OutputStmt):
      ty = self.convert_ast_type(stmt.ty)
      if stmt.expr:
        self.unify_types(ty, self.typeck_expr(stmt.expr), stmt.loc)
      stmt.fty = self.simplify_type(ty)
      return

    if isinstance(stmt, ast.LetStmt):
      ty = self.convert_ast_type(stmt.ty) if stmt.ty else InferrableType()
      if stmt.expr:
        self.unify_types(ty, self.typeck_expr(stmt.expr), stmt.loc)
      stmt.fty = self.simplify_type(ty)
      return

    if isinstance(stmt, ast.ReturnStmt):
      if not isinstance(self.root, ast.FnItem):
        emit_error(stmt.loc, "can only return from functions")
      ty = self.typeck_expr(stmt.expr) if stmt.expr else UnitType()
      assert self.root.return_fty is not None
      lhs_loc = self.root.return_ty.loc if self.root.return_ty else self.root.loc
      rhs_loc = stmt.expr.full_loc if stmt.expr else stmt.loc
      self.unify_types(self.root.return_fty, ty, stmt.loc, lhs_loc, rhs_loc)
      return

    if isinstance(stmt, ast.AssignStmt):
      lhs = self.typeck_expr(stmt.lhs)
      rhs = self.typeck_expr(stmt.rhs)
      self.unify_types(lhs, rhs, stmt.loc)
      return

    if isinstance(stmt, ast.IfStmt):
      cond = self.typeck_expr(stmt.cond)
      self.unify_types(
          UIntType(ConstIntParam(self, 1)),
          cond,
          stmt.loc,
          rhs_loc=stmt.cond.full_loc)
      solver_cond = convert_to_solver_expr(self, stmt.cond)
      if not isinstance(solver_cond, z3.BoolRef):
        solver_cond = (solver_cond != 0)
      for sub_stmt in stmt.then_stmts:
        self.solver.push()
        self.solver.add(solver_cond)
        self.typeck_stmt(sub_stmt)
        self.solver.pop()
      for sub_stmt in stmt.else_stmts:
        self.solver.push()
        self.solver.add(~solver_cond)
        self.typeck_stmt(sub_stmt)
        self.solver.pop()
      return

    emit_error(stmt.loc, f"unsupported in typeck: {stmt.__class__.__name__}")

  def typeck_expr(self, expr: ast.Expr) -> Type:
    expr.fty = self.simplify_type(self.type_of_expr(expr))
    return expr.fty

  def type_of_expr(self, expr: ast.Expr) -> Type:
    if isinstance(expr, ast.IntLitExpr):
      if expr.width is None:
        return LiteralIntType(min_bits_for_int(expr.value))
      else:
        return UIntType(ConstIntParam(self, expr.width))

    if isinstance(expr, ast.UnitLitExpr):
      return UnitType()

    if isinstance(expr, ast.IdentExpr):
      target = expr.binding.get()
      if isinstance(target, ast.FnParam):
        return GenericIntType()
      if hasattr(target, "fty") and target.fty and isinstance(target.fty, Type):
        return target.fty
      emit_info(target.loc, f"name `{expr.name.spelling()}` defined here")
      emit_error(expr.loc, f"`{expr.name.spelling()}` has no known type")

    if isinstance(expr, ast.ParenExpr):
      return self.typeck_expr(expr.expr)

    if isinstance(expr, ast.TupleExpr):
      fields = [self.typeck_expr(field) for field in expr.fields]
      return TupleType(fields)

    if isinstance(expr, ast.TupleFieldExpr):
      ty = self.typeck_expr(expr.target)
      if not isinstance(ty, TupleType):
        emit_error(expr.loc, f"cannot access tuple field in `{ty}`")
      if expr.field >= len(ty.fields):
        emit_info(
            expr.target.loc,
            f"the field index `{expr.field}` is outside the tuple `{ty}` whose field range is `0..{len(ty.fields)}`"
        )
        emit_error(expr.loc, f"tuple field out of bounds for `{ty}`")
      return ty.fields[expr.field]

    if isinstance(expr, ast.UnaryExpr):
      arg = self.typeck_expr(expr.arg)

      if expr.op == ast.UnaryOp.REF:
        return RefType(arg)

      if expr.op == ast.UnaryOp.DEREF:
        if isinstance(arg, RefType):
          return arg.inner
        ty = InferrableType()
        self.unify_types(arg, RefType(ty), expr.loc, expr.arg.full_loc,
                         expr.loc)
        return ty

      return arg

    if isinstance(expr, ast.BinaryExpr):
      lhs = self.typeck_expr(expr.lhs)
      rhs = self.typeck_expr(expr.rhs)
      self.unify_types(lhs, rhs, expr.loc, expr.lhs.full_loc, expr.rhs.full_loc)

      if expr.op in (ast.BinaryOp.EQ, ast.BinaryOp.NE, ast.BinaryOp.LT,
                     ast.BinaryOp.LE, ast.BinaryOp.GT, ast.BinaryOp.GE):
        return UIntType(ConstIntParam(self, 1))

      return lhs

    if isinstance(expr, ast.CallExpr):
      return self.type_of_call_expr(expr)

    if isinstance(expr, ast.FieldCallExpr):
      return self.type_of_field_call_expr(expr)

    emit_error(expr.loc, f"unsupported in typeck: {expr.__class__.__name__}")

  def type_of_call_expr(
      self,
      expr: ast.CallExpr,
  ) -> Type:
    name = expr.name.spelling()

    if target := expr.binding.target:
      # Make sure the name refers to a function.
      if not isinstance(target, ast.FnItem):
        emit_info(target.loc, f"`{name}` defined here")
        emit_error(expr.loc, f"cannot call `{name}`")

      # Create a new type-checking context for the called function, to
      # evaluate parameters.
      target_typeck = Typeck(target)

      # Define inferrable parameters for the called function's parameters.
      call_params = []
      for param in target.params:
        p = InferrableIntParam(self)
        target_typeck.params[param] = p
        # self.solver_params[param] = z3.FreshConst(z3.IntSort())
        call_params.append(p)

      # Make sure the arguments match.
      if len(expr.args) != len(target.args):
        emit_error(
            expr.loc,
            f"argument mismatch: `{name}` takes {len(target.args)} arguments, but got {len(expr.args)} instead"
        )
      for call_arg, target_arg in zip(expr.args, target.args):
        call_ty = self.typeck_expr(call_arg)
        func_ty = target_typeck.convert_ast_type(target_arg.ty)
        self.unify_types(func_ty, call_ty, call_arg.loc, lhs_loc=target_arg.loc)

      # Annotate the call parameters.
      expr.call_params = [self.simplify_param(p) for p in call_params]
      call_solver_params = []
      for p2 in expr.call_params:
        call_solver_params.append(convert_to_solver_expr(p2.cx, p2))

      # Check that we adhere to the where clauses.
      for where in target.wheres:
        solver_expr = convert_to_solver_expr(target_typeck, where)
        self.solver.push()
        self.solver.add(~solver_expr)
        good = (self.solver.check() == z3.unsat)
        if not good:
          model = self.solver.model()
          for ast_param, free_param in zip(target.params, call_solver_params):
            value = model.eval(free_param)
            param_str = str(free_param)
            value_str = str(value)
            if param_str != value_str:
              value_str = f"{param_str} = {value_str}"
            emit_info(where.loc,
                      f"consider {ast_param.loc.spelling()} = {value_str}")
          emit_error(
              expr.loc,
              f"parameter mismatch: `{name}` requires {where.full_loc.spelling()}"
          )
        self.solver.pop()

      # Propagate the return type.
      return target_typeck.convert_ast_type(
          target.return_ty) if target.return_ty else UnitType()

    # Handle builtin functions.
    expr.call_params = []

    if name == "concat":
      width = 0
      for arg in expr.args:
        ty = self.typeck_expr(arg)
        if not isinstance(ty, UIntType) or not isinstance(
            ty.width, ConstIntParam):
          emit_error(arg.loc, f"cannot concatenate `{ty}`")
        width += ty.width.value
      return UIntType(ConstIntParam(self, width))

    if name == "wire":
      require_num_args(expr, 1)
      return WireType(self.typeck_expr(expr.args[0]))

    if name == "reg":
      require_num_args(expr, 2)
      clock = self.typeck_expr(expr.args[0])
      self.unify_types(
          UIntType(ConstIntParam(self, 1)),
          clock,
          expr.loc,
          rhs_loc=expr.args[0].full_loc)
      return RegType(self.typeck_expr(expr.args[1]))

    emit_error(expr.name.loc, f"unknown function `{name}`")

  def type_of_field_call_expr(
      self,
      expr: ast.FieldCallExpr,
  ) -> Type:
    name = expr.name.spelling()

    if name == "bit":
      require_num_args(expr, 1)
      target = self.typeck_expr(expr.target)
      self.typeck_expr(expr.args[0])
      if not isinstance(target, UIntType):
        emit_error(expr.loc, f"cannot access bits in `{target}`")

      offset = require_int_lit(expr, 0)
      if isinstance(target.width, ConstIntParam):
        if offset < 0 or offset >= target.width.value:
          emit_info(
              expr.target.loc,
              f"the bit index `{offset}` is outside the type `{target}` whose bit range is `0..{target.width}`"
          )
          emit_error(expr.args[0].loc,
                     f"bit index out of bounds for `{target}`")
      else:
        self.constraints.append(
            Constraint(target.width, ">", expr.args[0], expr.loc))
        good, counter = self.prove_greater_than(target.width, offset)
        if not good:
          emit_info(
              expr.target.loc,
              f"the bit index may be outside the type `{target}` whose bit range is `0..{target.width}`"
          )
          for param, loc, value in counter:
            emit_info(loc, f"for example consider {param} = {value}")
          emit_error(expr.args[0].loc,
                     f"bit index out of bounds for `{target}`")

      return UIntType(ConstIntParam(self, 1))

    if name == "slice":
      require_num_args(expr, 2)
      offset = require_int_lit(expr, 0)
      width = require_int_lit(expr, 1, min=0)
      target = self.typeck_expr(expr.target)
      self.typeck_expr(expr.args[0])
      self.typeck_expr(expr.args[1])
      if not isinstance(target, UIntType) or not isinstance(
          target.width, ConstIntParam):
        emit_error(expr.loc, f"cannot slice `{target}`")
      if offset + width > target.width.value:
        emit_info(
            expr.args[0].loc | expr.args[1].loc,
            f"the slice `{offset}..{offset+width}` is outside the type `{target}` whose bit range is `0..{target.width}`"
        )
        emit_error(expr.args[0].loc | expr.args[1].loc,
                   f"slice out of bounds for `{target}`")
      return UIntType(ConstIntParam(self, width))

    if name == "mux":
      require_num_args(expr, 2)
      target = self.typeck_expr(expr.target)
      self.unify_types(
          UIntType(ConstIntParam(self, 1)),
          target,
          expr.loc,
          rhs_loc=expr.target.full_loc)

      lhs = self.typeck_expr(expr.args[0])
      rhs = self.typeck_expr(expr.args[1])
      self.unify_types(lhs, rhs, expr.loc, expr.args[0].full_loc,
                       expr.args[1].full_loc)
      return lhs

    if name == "set":
      require_num_args(expr, 1)
      target = self.typeck_expr(expr.target)
      inner = InferrableType()
      self.unify_types(
          target, WireType(inner), expr.loc, lhs_loc=expr.target.full_loc)
      arg = self.typeck_expr(expr.args[0])
      self.unify_types(
          inner,
          arg,
          expr.loc,
          lhs_loc=expr.target.full_loc,
          rhs_loc=expr.args[0].full_loc)
      return target

    if name == "get":
      require_num_args(expr, 0)
      target = self.typeck_expr(expr.target)
      inner = InferrableType()
      self.unify_types(
          target, WireType(inner), expr.loc, lhs_loc=expr.target.full_loc)
      return inner

    if name == "next":
      require_num_args(expr, 1)
      target = self.typeck_expr(expr.target)
      inner = InferrableType()
      self.unify_types(
          target, RegType(inner), expr.loc, lhs_loc=expr.target.full_loc)
      arg = self.typeck_expr(expr.args[0])
      self.unify_types(
          inner,
          arg,
          expr.loc,
          lhs_loc=expr.target.full_loc,
          rhs_loc=expr.args[0].full_loc)
      return target

    if name == "current":
      require_num_args(expr, 0)
      target = self.typeck_expr(expr.target)
      inner = InferrableType()
      self.unify_types(
          target, RegType(inner), expr.loc, lhs_loc=expr.target.full_loc)
      return inner

    if name == "as_uint":
      require_num_args(expr, 0)

      # Make sure we are operating on a generic integer.
      target = self.typeck_expr(expr.target)
      if not isinstance(target, GenericIntType):
        emit_error(expr.target.full_loc, "`as_uint` requires a generic integer")

      # Create a `uint<?A>` return type with `?A` constrained to be large enough
      # to hold the generic integer.
      param = InferrableIntParam(self)
      self.constraints.append(
          Constraint(param, ">=", ConstraintFn("clog2", expr.target), expr.loc))
      return UIntType(param)

    if name == "clog2":
      require_num_args(expr, 0)
      return self.typeck_expr(expr.target)

    if name == "trunc":
      require_num_args(expr, 0)
      target = self.typeck_expr(expr.target)
      target_param = InferrableIntParam(self)
      self.unify_types(
          target,
          UIntType(target_param),
          expr.loc,
          lhs_loc=expr.target.full_loc)
      result_param = InferrableIntParam(self)
      self.constraints.append(
          Constraint(result_param, "<=", target_param, expr.loc))
      return UIntType(result_param)

    if name == "zext":
      require_num_args(expr, 0)
      target = self.typeck_expr(expr.target)
      target_param = InferrableIntParam(self)
      self.unify_types(
          target,
          UIntType(target_param),
          expr.loc,
          lhs_loc=expr.target.full_loc)
      result_param = InferrableIntParam(self)
      self.constraints.append(
          Constraint(result_param, ">=", target_param, expr.loc))
      return UIntType(result_param)

    emit_error(expr.name.loc, f"unknown function `{name}`")

  # Finalize the types in the AST. This replaces all type variables with the
  # actual type that was inferred.
  def finalize(self, root: ast.AstNode):
    for node in root.walk(ast.WalkOrder.PostOrder):
      if hasattr(node, "fty"):
        if node.fty is None:
          emit_error(node.loc, f"uninferred type: {node.__class__.__name__}")
        node.fty = self.finalize_type(node.fty)
      self.finalize_node(node)

  # A final chance to modify the type of each node in the AST. This is useful
  # to do a final cleanup and get rid of type variables, or replace uninferred
  # types with some default.
  def finalize_type(self, ty: Type) -> Type:
    ty = self.simplify_type(ty)

    if isinstance(ty, WireType):
      return WireType(self.finalize_type(ty.inner))
    if isinstance(ty, RegType):
      return RegType(self.finalize_type(ty.inner))
    if isinstance(ty, RefType):
      return RefType(self.finalize_type(ty.inner))
    if isinstance(ty, TupleType):
      return TupleType([self.finalize_type(fty) for fty in ty.fields])

    # Map `{literal}` to the smallest possible `uint<N>`.
    if isinstance(ty, LiteralIntType):
      return UIntType(ConstIntParam(self, ty.width_hint))

    # Finalize parameters.
    if isinstance(ty, UIntType):
      ty.width = self.finalize_param(ty.width)

    return ty

  # A final chance to modify a parameter at the end of type-checking. This is
  # useful to do a final cleanup and get rid of parameters, or replace
  # uninferred parameters with some default.
  def finalize_param(self, param: IntParam) -> IntParam:
    param = self.simplify_param(param)

    # Make uninferred int parameters the minimum size for their literals.
    if isinstance(param, InferrableIntParam) and param.width_hint is not None:
      param.inferred = ConstIntParam(self, param.width_hint)
      param = param.inferred

    return param

  def finalize_node(self, node: ast.AstNode):
    # Make sure that integer literals fit into their inferred type.
    if isinstance(node, ast.IntLitExpr) and node.width is None:
      width = min_bits_for_int(node.value)
      if width > 0 and isinstance(
          node.fty, UIntType) and (not isinstance(node.fty.width, ConstIntParam)
                                   or width > node.fty.width.value):
        emit_info(
            node.loc,
            f"the literal `{node.value}` does not fit into the type `{node.fty}` since it requires at least {width} bits"
        )
        emit_error(node.loc, f"literal out of range for `{node.fty}`")

    # Unary and binary operators must operate on integers.
    if isinstance(node, ast.UnaryExpr) or isinstance(node, ast.BinaryExpr):
      if not isinstance(
          node.fty,
          (UIntType, GenericIntType, LiteralIntType)) and node.op not in (
              ast.UnaryOp.REF, ast.UnaryOp.DEREF):
        word = {
            ast.UnaryOp.NEG: "negate",
            ast.UnaryOp.NOT: "invert",
            ast.BinaryOp.AND: "bitwise and",
            ast.BinaryOp.OR: "bitwise or",
            ast.BinaryOp.XOR: "bitwise xor",
            ast.BinaryOp.ADD: "add",
            ast.BinaryOp.SUB: "subtract",
            ast.BinaryOp.MUL: "multiply",
            ast.BinaryOp.DIV: "divide",
            ast.BinaryOp.MOD: "compute the remainder of",
            ast.BinaryOp.SHL: "shift",
            ast.BinaryOp.SHR: "shift",
            ast.BinaryOp.EQ: "compare",
            ast.BinaryOp.NE: "compare",
            ast.BinaryOp.LT: "compare",
            ast.BinaryOp.LE: "compare",
            ast.BinaryOp.GT: "compare",
            ast.BinaryOp.GE: "compare",
        }.get(node.op, node.op.name)
        emit_error(node.loc, f"cannot {word} `{node.fty}`")

    # Finalize call parameters.
    if isinstance(node, ast.CallExpr):
      assert node.call_params is not None
      node.call_params = [self.finalize_param(p) for p in node.call_params]

  # TODO: This should return the final unified type to simplify code in other
  # places.
  def unify_types(
      self,
      lhs: Type,
      rhs: Type,
      loc: Loc,
      lhs_loc: Optional[Loc] = None,
      rhs_loc: Optional[Loc] = None,
  ):
    lhs = self.simplify_type(lhs)
    rhs = self.simplify_type(rhs)
    if lhs == rhs:
      return

    # unify(Wire<?A>, Wire<?B>) -> Wire<unify(?A, ?B)>
    if isinstance(lhs, WireType) and isinstance(rhs, WireType):
      self.unify_types(lhs.inner, rhs.inner, loc, lhs_loc, rhs_loc)
      return

    # unify(Reg<?A>, Reg<?B>) -> Reg<unify(?A, ?B)>
    if isinstance(lhs, RegType) and isinstance(rhs, RegType):
      self.unify_types(lhs.inner, rhs.inner, loc, lhs_loc, rhs_loc)
      return

    # unify(&?A, &?B) -> &unify(?A, ?B)
    if isinstance(lhs, RefType) and isinstance(rhs, RefType):
      self.unify_types(lhs.inner, rhs.inner, loc, lhs_loc, rhs_loc)
      return

    # Unify fields of tuples.
    if isinstance(lhs, TupleType) and isinstance(rhs, TupleType):
      if len(lhs.fields) == len(rhs.fields):
        for a, b in zip(lhs.fields, rhs.fields):
          self.unify_types(a, b, loc, lhs_loc, rhs_loc)
        return

    # unify(uint<?A>, uint<?B>) -> uint<?A> with maximized width hint
    # unify(uint<?A>, uint<N>) -> uint<N>
    # unify(uint<N>, uint<?B>) -> uint<N>
    if isinstance(lhs, UIntType) and isinstance(rhs, UIntType):
      if isinstance(lhs.width, InferrableIntParam) and isinstance(
          rhs.width, InferrableIntParam):
        rhs.width.inferred = lhs.width
        lhs.width.width_hint = max(lhs.width.width_hint, rhs.width.width_hint)
        return
      if isinstance(rhs.width, InferrableIntParam):
        rhs.width.inferred = lhs.width
        return
      if isinstance(lhs.width, InferrableIntParam):
        lhs.width.inferred = rhs.width
        return

      if isinstance(lhs.width, DerivedIntParam) and isinstance(
          rhs.width, DerivedIntParam):
        if lhs.width.cx == rhs.width.cx and exprs_match(lhs.width.expr,
                                                        rhs.width.expr):
          return

    # unify(uint<?A>, {literal}) -> uint<?A> with maximized width hint
    # unify({literal}, uint<?B>) -> uint<?B> with maximized width hint
    # unify(uint<N>, {literal}) -> uint<N>
    # unify({literal}, uint<N>) -> uint<N>
    if isinstance(lhs, UIntType) and isinstance(rhs, LiteralIntType):
      rhs.inferred = lhs
      if isinstance(lhs.width, InferrableIntParam):
        lhs.width.width_hint = max(lhs.width.width_hint, rhs.width_hint)
      return
    if isinstance(lhs, LiteralIntType) and isinstance(rhs, UIntType):
      lhs.inferred = rhs
      if isinstance(rhs.width, InferrableIntParam):
        rhs.width.width_hint = max(rhs.width.width_hint, lhs.width_hint)
      return

    # unify({literal}, {literal}) -> {literal} with maximized width hint
    if isinstance(lhs, LiteralIntType) and isinstance(rhs, LiteralIntType):
      rhs.inferred = lhs
      lhs.width_hint = max(lhs.width_hint, rhs.width_hint)
      rhs.width_hint = lhs.width_hint
      return

    # unify({integer}, {integer}) -> {integer}
    if isinstance(lhs, GenericIntType) and isinstance(rhs, GenericIntType):
      rhs.inferred = lhs
      return

    # unify({integer}, {literal}) -> {integer}
    # unify({literal}, {integer}) -> {integer}
    if isinstance(lhs, GenericIntType) and isinstance(rhs, LiteralIntType):
      rhs.inferred = lhs
      return
    if isinstance(lhs, LiteralIntType) and isinstance(rhs, GenericIntType):
      lhs.inferred = rhs
      return

    # unify(?A, ?B) -> ?A
    # unify(?A, T) -> T
    # unify(T, ?B) -> T
    if isinstance(lhs, InferrableType) and isinstance(rhs, InferrableType):
      rhs.inferred = lhs
      return
    if isinstance(lhs, InferrableType):
      lhs.inferred = rhs
      return
    if isinstance(rhs, InferrableType):
      rhs.inferred = lhs
      return

    # If we get here the types could not be unified.
    if lhs_loc is not None:
      emit_info(lhs_loc, f"type: `{lhs}`")
    if rhs_loc is not None:
      emit_info(rhs_loc, f"type: `{rhs}`")
    emit_error(loc, f"incompatible types: `{lhs}` and `{rhs}`")

  # Simplify a type by replacing type variables with their assigned type.
  def simplify_type(self, ty: Type) -> Type:
    if isinstance(ty, TupleType):
      ty.fields = [self.simplify_type(field) for field in ty.fields]
      return ty

    if isinstance(ty, InferrableType) and ty.inferred:
      ty.inferred = self.simplify_type(ty.inferred)
      return ty.inferred

    if isinstance(ty, LiteralIntType) and ty.inferred:
      inner = self.simplify_type(ty.inferred)
      assert isinstance(inner, (LiteralIntType, GenericIntType, UIntType))
      ty.inferred = inner
      return ty.inferred

    if isinstance(ty, GenericIntType) and ty.inferred:
      inner = self.simplify_type(ty.inferred)
      assert isinstance(inner, (GenericIntType, UIntType))
      ty.inferred = inner
      return ty.inferred

    if isinstance(ty, UIntType):
      ty.width = self.simplify_param(ty.width)

    if isinstance(ty, (RefType, WireType, RegType)):
      ty.inner = self.simplify_type(ty.inner)

    return ty

  # Simplify a parameter by replacing variables with their assigned value.
  def simplify_param(self, param: IntParam) -> IntParam:
    if isinstance(param, InferrableIntParam) and param.inferred:
      param.inferred = self.simplify_param(param.inferred)
      return param.inferred

    if isinstance(param, DerivedIntParam):
      from silicon import consteval
      const_cx = consteval.AstContext()
      for child in param.expr.walk():
        if isinstance(child, ast.IdentExpr):
          binding = child.binding.get()
          if isinstance(binding, ast.FnParam):
            p = param.cx.simplify_param(param.cx.params[binding])
            if isinstance(p, ConstIntParam):
              const_cx.values[child] = p.value
      value = consteval.try_const_eval_ast(const_cx, param.expr)
      if value is not None:
        return ConstIntParam(param.cx, value)

    return param

  # Prove that the given parameter is greater than the given constant value.
  def prove_greater_than(
      self,
      param: IntParam,
      value: int,
  ) -> Tuple[bool, List[Tuple[IntParam, Loc, int]]]:
    self.solver.push()
    solver_param = self.solver_params[param]
    self.solver.add(~(solver_param > value))
    good = (self.solver.check() == z3.unsat)
    counter = []
    if not good:
      model = self.solver.model()
      if model:
        for ast_param, free_param in self.params.items():
          solver_param = self.solver_params[free_param]
          value = model.eval(solver_param)
          counter.append((free_param, ast_param.loc, value))
    self.solver.pop()
    return good, counter


def format_constraint(
    cx: Typeck, con: Constraint | ConstraintFn | IntParam | ast.Expr) -> str:
  if isinstance(con, Constraint):
    lhs = format_constraint(cx, con.lhs)
    rhs = format_constraint(cx, con.rhs)
    return f"{lhs} {con.relation} {rhs}"

  if isinstance(con, ConstraintFn):
    return f"{con.op}({format_constraint(cx, con.arg)})"

  if isinstance(con, IntParam):
    con = cx.simplify_param(con)
  if isinstance(con, ConstIntParam):
    return str(con.value)
  if isinstance(con, FreeIntParam):
    return str(con)
  if isinstance(con, DerivedIntParam):
    return format_constraint(con.cx, con.expr)

  if isinstance(con, ast.IntLitExpr):
    if con.width is not None:
      return f"{con.value}u{con.width}"
    return str(con.value)

  if isinstance(con, ast.IdentExpr):
    binding = con.binding.get()
    if isinstance(binding, ast.FnParam):
      return format_constraint(cx, cx.params[binding])

  if isinstance(con, ast.FieldCallExpr):
    args = ", ".join(format_constraint(cx, arg) for arg in con.args)
    return f"{format_constraint(cx, con.target)}.{con.name.spelling()}({args})"

  if isinstance(con, ast.BinaryExpr):
    return "(" + format_constraint(
        cx, con.lhs) + con.loc.spelling() + format_constraint(cx, con.rhs) + ")"

  assert False, f"cannot format {con}"


def check_constraints(cx: Typeck):
  # Make sure the constraints reference the final inferred parameters.
  for con in cx.constraints:
    con.lhs = cx.simplify_param(con.lhs)

  # Check each constraint.
  for con in cx.constraints:
    cx.solver.push()
    lhs = convert_to_solver_expr(cx, con.lhs)
    rhs = convert_to_solver_expr(cx, con.rhs)
    if con.relation == ">=":
      cond = lhs >= rhs
    elif con.relation == ">":
      cond = lhs > rhs
    elif con.relation == "<=":
      cond = lhs <= rhs
    elif con.relation == "<":
      cond = lhs < rhs
    elif con.relation == "==":
      cond = lhs == rhs
    elif con.relation == "!=":
      cond = lhs != rhs
    else:
      assert False, f"`{con.relation}` relations not implemented in solver"
    # Add (A > B) -> (clog2(A) >= clog2(B)) helpers.
    if lhs.decl().name() == "clog2" and rhs.decl().name() == "clog2":
      cx.solver.add(z3.Implies(lhs.arg(0) > rhs.arg(0), lhs >= rhs))
      cx.solver.add(z3.Implies(lhs.arg(0) < rhs.arg(0), lhs <= rhs))
      cx.solver.add(z3.Implies(lhs.arg(0) == rhs.arg(0), lhs == rhs))
    cx.solver.add(~cond)

    if cx.solver.check() == z3.sat:
      model = cx.solver.model()
      for c, e in ((con.lhs, lhs), (con.rhs, rhs)):
        c_str = format_constraint(cx, c)
        e_str = str(model.eval(e))
        if c_str != e_str:
          emit_info(con.loc, f"consider `{c_str} = {e_str}`")
      emit_error(
          con.loc,
          f"type mismatch: `{format_constraint(cx, con)}` does not always hold")

    cx.solver.pop()


def convert_to_solver_expr(cx: Typeck,
                           expr: IntParam | ConstraintFn | ast.Expr) -> z3.Ast:
  if isinstance(expr, IntParam):
    expr = cx.simplify_param(expr)

  if isinstance(expr, ConstIntParam):
    return z3.IntVal(expr.value)

  if isinstance(expr, (InferrableIntParam, FreeIntParam)):
    if expr not in expr.cx.solver_params:
      param = z3.FreshConst(z3.IntSort(), str(expr))
      expr.cx.solver.add(param >= 0)
      expr.cx.solver_params[expr] = param
    return expr.cx.solver_params[expr]

  if isinstance(expr, DerivedIntParam):
    return convert_to_solver_expr(expr.cx, expr.expr)

  if isinstance(expr, ConstraintFn):
    if expr.op == "clog2":
      arg = convert_to_solver_expr(cx, expr.arg)
      clog2 = z3.Function("clog2", z3.IntSort(), z3.IntSort())
      cx.solver.add(clog2(arg) < arg)
      return clog2(arg)

  if isinstance(expr, ast.IntLitExpr):
    return z3.IntVal(expr.value)

  if isinstance(expr, ast.IdentExpr):
    binding = expr.binding.get()
    if isinstance(binding, ast.FnParam):
      return convert_to_solver_expr(cx, cx.params[binding])
    # Otherwise create an opaque term in the solver.
    if expr not in cx.solver_nodes:
      opaque = z3.FreshConst(z3.IntSort())
      cx.solver_nodes[expr] = opaque
    return cx.solver_nodes[expr]

  if isinstance(expr, ast.ParenExpr):
    return convert_to_solver_expr(cx, expr.expr)

  if isinstance(expr, ast.BinaryExpr):
    lhs = convert_to_solver_expr(cx, expr.lhs)
    rhs = convert_to_solver_expr(cx, expr.rhs)
    if expr.op == ast.BinaryOp.EQ:
      return lhs == rhs
    if expr.op == ast.BinaryOp.NE:
      return lhs != rhs
    if expr.op == ast.BinaryOp.GT:
      return lhs > rhs
    if expr.op == ast.BinaryOp.GE:
      return lhs >= rhs
    if expr.op == ast.BinaryOp.LT:
      return lhs < rhs
    if expr.op == ast.BinaryOp.LE:
      return lhs <= rhs
    if expr.op == ast.BinaryOp.ADD:
      return lhs + rhs
    if expr.op == ast.BinaryOp.SUB:
      return lhs - rhs
    if expr.op == ast.BinaryOp.MUL:
      return lhs * rhs
    if expr.op == ast.BinaryOp.DIV:
      return lhs / rhs
    if expr.op == ast.BinaryOp.MOD:
      return lhs % rhs
    emit_error(expr.loc, "solving of operator `{expr.op}` not implemented")

  if isinstance(expr, ast.FieldCallExpr):
    name = expr.name.spelling()
    if name == "clog2":
      arg = convert_to_solver_expr(cx, expr.target)
      clog2 = z3.Function("clog2", z3.IntSort(), z3.IntSort())
      if isinstance(arg, z3.IntNumRef):
        return z3.IntVal(len(arg.as_binary_string()))
      cx.solver.add(clog2(arg) < z3.If(arg == 0, 1, arg))
      cx.solver.add(clog2(arg) >= 0)
      return clog2(arg)

  if isinstance(expr, ast.Expr):
    emit_error(expr.full_loc, "unsupported constraint expression")

  assert False, f"solving of `{expr}` not implemented ({expr.__class__.__name__})"


# Check whether two expressions are structurally equivalent.
def exprs_match(a: ast.Expr, b: ast.Expr) -> bool:
  if type(a) != type(b):
    return False

  if isinstance(a, ast.IdentExpr) and isinstance(b, ast.IdentExpr):
    return a.name.spelling() == b.name.spelling()

  if isinstance(a, ast.FieldCallExpr) and isinstance(b, ast.FieldCallExpr):
    if not exprs_match(a.target, b.target):
      return False
    if a.name.spelling() != b.name.spelling():
      return False
    for arg_a, arg_b in zip(a.args, b.args):
      if not exprs_match(arg_a, arg_b):
        return False
    return True

  assert False, f"cannot compare {type(a).__name__}"


def typeck(root: ast.Root):
  root_typecks = []

  # Type-check all signature and prepare the necessary free parameters.
  for item in root.items:
    if isinstance(item, (ast.ModItem, ast.FnItem)):
      cx = Typeck(item)
      cx.typeck_signature()
      root_typecks.append((item, cx))
    else:
      emit_error(item.loc, f"unsupported in typeck: {item.__class__.__name__}")
  assert len(root_typecks) == len(root.items)

  # Type-check the AST in full detail.
  for item, cx in root_typecks:
    cx.typeck()
