from __future__ import annotations

import z3
from math import log2, floor
from typing import Optional, List, Dict
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
    pass


@dataclass
class ConstIntParam(IntParam):
    value: int

    def __str__(self) -> str:
        return f"{self.value}"


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


#===------------------------------------------------------------------------===#
# Type Hierarchy
#===------------------------------------------------------------------------===#


@dataclass
class Type:
    pass


@dataclass
class UnitType(Type):

    def __str__(self) -> str:
        return "()"


@dataclass
class GenericIntType(Type):

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
class InferrableType(Type):
    # A unique integer identifier.
    num: int = field(default_factory=count().__next__)

    # The type this uint has been inferred to.
    inferred: Type | None = None

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    def __str__(self) -> str:
        return f"?{self.num}"


#===------------------------------------------------------------------------===#
# Type Checking and Inference
#===------------------------------------------------------------------------===#


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


class Typeck:
    root: ast.ModItem | ast.FnItem
    params: Dict[ast.AstNode, IntParam]
    solver: z3.Solver
    solver_params: Dict[ast.AstNode, z3.Int]

    def __init__(self, root: ast.ModItem | ast.FnItem) -> None:
        self.root = root
        self.params = dict()
        self.solver = z3.Solver()
        self.solver_params = dict()

    # Convert an AST type node into an actual type.
    def convert_ast_type(self, ty: ast.AstType) -> Type:
        if isinstance(ty, ast.UnitType):
            return UnitType()
        if isinstance(ty, ast.TupleType):
            return TupleType(
                [self.convert_ast_type(field) for field in ty.fields])
        if isinstance(ty, ast.UIntType):
            self.typeck_expr(ty.size)
            if isinstance(ty.size, ast.IntLitExpr):
                return UIntType(ConstIntParam(ty.size.value))
            if isinstance(ty.size, ast.IdentExpr):
                binding = ty.size.binding.get()
                param = self.params.get(binding)
                if not param:
                    emit_info(binding.loc,
                              f"`{ty.size.name.spelling()}` defined here")
                    emit_error(ty.size.loc,
                               f"invalid parameter for `uint` width")
                return UIntType(param)
            emit_error(ty.size.loc, f"unsupported expression for `uint` width")
        if isinstance(ty, ast.WireType):
            return WireType(self.convert_ast_type(ty.inner))
        if isinstance(ty, ast.RegType):
            return RegType(self.convert_ast_type(ty.inner))
        emit_error(ty.loc, f"unsupported type: `{ty.loc.spelling()}`")

    # Type-check just the signature of the `root` node.
    def typeck_signature(self):
        if isinstance(self.root, ast.FnItem):
            # Declare the parameters.
            for param in self.root.params:
                param.fty = GenericIntType()
                param.free_param = FreeIntParam(param.name.spelling())
                self.params[param] = param.free_param
                self.solver_params[param.free_param] = z3.Int(
                    param.name.spelling())

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
        solver_expr = self.convert_to_solver_expr(expr)

        # Check if the clause is trivially true.
        self.solver.push()
        self.solver.add(~solver_expr)
        if self.solver.check() == z3.unsat:
            emit_warning(expr.loc,
                         "constraint always holds and can be removed")
        self.solver.pop()

        # Check if the clause renders the constraints unsatisfiable.
        self.solver.add(solver_expr)
        if self.solver.check() == z3.unsat:
            emit_error(expr.loc, "constraint is unsatisfiable")

    # Type-check the entire AST starting at the `root` node.
    def typeck(self):
        for stmt in self.root.stmts:
            self.typeck_stmt(stmt)

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
            ty = self.convert_ast_type(
                stmt.ty) if stmt.ty else InferrableType()
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
            self.unify_types(self.root.return_fty, ty, stmt.loc, lhs_loc,
                             rhs_loc)
            return

        if isinstance(stmt, ast.AssignStmt):
            lhs = self.typeck_expr(stmt.lhs)
            rhs = self.typeck_expr(stmt.rhs)
            self.unify_types(lhs, rhs, stmt.loc)
            return

        emit_error(stmt.loc,
                   f"unsupported in typeck: {stmt.__class__.__name__}")

    def typeck_expr(self, expr: ast.Expr) -> Type:
        expr.fty = self.simplify_type(self.type_of_expr(expr))
        return expr.fty

    def type_of_expr(self, expr: ast.Expr) -> Type:
        if isinstance(expr, ast.IntLitExpr):
            if expr.width is None:
                return UIntType(
                    InferrableIntParam(
                        width_hint=min_bits_for_int(expr.value)))
            else:
                return UIntType(ConstIntParam(expr.width))

        if isinstance(expr, ast.UnitLitExpr):
            return UnitType()

        if isinstance(expr, ast.IdentExpr):
            target = expr.binding.get()
            if hasattr(target, "fty") and target.fty and isinstance(
                    target.fty, Type):
                return target.fty
            emit_info(target.loc,
                      f"name `{expr.name.spelling()}` defined here")
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
            return self.typeck_expr(expr.arg)

        if isinstance(expr, ast.BinaryExpr):
            lhs = self.typeck_expr(expr.lhs)
            rhs = self.typeck_expr(expr.rhs)
            self.unify_types(lhs, rhs, expr.loc, expr.lhs.full_loc,
                             expr.rhs.full_loc)
            return lhs

        if isinstance(expr, ast.CallExpr):
            return self.type_of_call_expr(expr)

        if isinstance(expr, ast.FieldCallExpr):
            return self.type_of_field_call_expr(expr)

        emit_error(expr.loc,
                   f"unsupported in typeck: {expr.__class__.__name__}")

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
                p = InferrableIntParam()
                target_typeck.params[param] = p
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
                self.unify_types(func_ty,
                                 call_ty,
                                 call_arg.loc,
                                 lhs_loc=target_arg.loc)

            # Annotate the call parameters.
            expr.call_params = [self.simplify_param(p) for p in call_params]

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
            return UIntType(ConstIntParam(width))

        if name == "wire":
            require_num_args(expr, 1)
            return WireType(self.typeck_expr(expr.args[0]))

        if name == "reg":
            require_num_args(expr, 2)
            clock = self.typeck_expr(expr.args[0])
            self.unify_types(UIntType(ConstIntParam(1)),
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
                good, counter = self.prove_greater_than(target.width, offset)
                if not good:
                    emit_info(
                        expr.target.loc,
                        f"the bit index may be outside the type `{target}` whose bit range is `0..{target.width}`"
                    )
                    for param, loc, value in counter:
                        emit_info(loc,
                                  f"for example consider {param} = {value}")
                    emit_error(expr.args[0].loc,
                               f"bit index out of bounds for `{target}`")

            return UIntType(ConstIntParam(1))

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
            return UIntType(ConstIntParam(width))

        if name == "mux":
            require_num_args(expr, 2)
            target = self.typeck_expr(expr.target)
            self.unify_types(UIntType(ConstIntParam(1)),
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
            self.unify_types(target,
                             WireType(inner),
                             expr.loc,
                             lhs_loc=expr.target.full_loc)
            arg = self.typeck_expr(expr.args[0])
            self.unify_types(inner,
                             arg,
                             expr.loc,
                             lhs_loc=expr.target.full_loc,
                             rhs_loc=expr.args[0].full_loc)
            return target

        if name == "get":
            require_num_args(expr, 0)
            target = self.typeck_expr(expr.target)
            inner = InferrableType()
            self.unify_types(target,
                             WireType(inner),
                             expr.loc,
                             lhs_loc=expr.target.full_loc)
            return inner

        if name == "next":
            require_num_args(expr, 1)
            target = self.typeck_expr(expr.target)
            inner = InferrableType()
            self.unify_types(target,
                             RegType(inner),
                             expr.loc,
                             lhs_loc=expr.target.full_loc)
            arg = self.typeck_expr(expr.args[0])
            self.unify_types(inner,
                             arg,
                             expr.loc,
                             lhs_loc=expr.target.full_loc,
                             rhs_loc=expr.args[0].full_loc)
            return target

        if name == "current":
            require_num_args(expr, 0)
            target = self.typeck_expr(expr.target)
            inner = InferrableType()
            self.unify_types(target,
                             RegType(inner),
                             expr.loc,
                             lhs_loc=expr.target.full_loc)
            return inner

        emit_error(expr.name.loc, f"unknown function `{name}`")

    # Finalize the types in the AST. This replaces all type variables with the
    # actual type that was inferred.
    def finalize(self, root: ast.AstNode):
        for node in root.walk(ast.WalkOrder.PostOrder):
            if hasattr(node, "fty"):
                if node.fty is None:
                    emit_error(node.loc,
                               f"uninferred type: {node.__class__.__name__}")
                node.fty = self.finalize_type(node.fty)
            self.finalize_node(node)

    # A final chance to modify the type of each node in the AST. This is useful
    # to do a final cleanup and get rid of type variables, or replace uninferred
    # types with some default.
    def finalize_type(self, ty: Type) -> Type:
        ty = self.simplify_type(ty)

        # Finalize parameters.
        if isinstance(ty, UIntType):
            ty.width = self.finalize_param(ty.width)

        # Finalize inner types.
        if isinstance(ty, (WireType, RegType)):
            ty.inner = self.finalize_type(ty.inner)

        return ty

    # A final chance to modify a parameter at the end of type-checking. This is
    # useful to do a final cleanup and get rid of parameters, or replace
    # uninferred parameters with some default.
    def finalize_param(self, param: IntParam) -> IntParam:
        param = self.simplify_param(param)

        # Make uninferred int parameters the minimum size for their literals.
        if isinstance(param,
                      InferrableIntParam) and param.width_hint is not None:
            param.inferred = ConstIntParam(param.width_hint)
            param = param.inferred

        return param

    def finalize_node(self, node: ast.AstNode):
        # Make sure that integer literals fit into their inferred type.
        if isinstance(node, ast.IntLitExpr) and node.width is None:
            width = min_bits_for_int(node.value)
            if isinstance(node.fty, UIntType) and (
                    not isinstance(node.fty.width, ConstIntParam)
                    or width > node.fty.width.value):
                emit_info(
                    node.loc,
                    f"the literal `{node.value}` does not fit into the type `{node.fty}` since it requires at least {width} bits"
                )
                emit_error(node.loc, f"literal out of range for `{node.fty}`")

        # Unary and binary operators must operate on integers.
        if isinstance(node, ast.UnaryExpr) or isinstance(node, ast.BinaryExpr):
            if not isinstance(node.fty, UIntType) and not isinstance(
                    node.fty, GenericIntType):
                word = {
                    ast.UnaryOp.NEG: "negate",
                    ast.UnaryOp.NOT: "invert",
                    ast.BinaryOp.ADD: "add",
                    ast.BinaryOp.SUB: "subtract",
                }.get(node.op, node.op.name)
                emit_error(node.loc, f"cannot {word} `{node.fty}`")

        # Finalize call parameters.
        if isinstance(node, ast.CallExpr):
            assert node.call_params is not None
            node.call_params = [
                self.finalize_param(p) for p in node.call_params
            ]

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

        # Unify fields of tuples.
        if isinstance(lhs, TupleType) and isinstance(rhs, TupleType):
            if len(lhs.fields) == len(rhs.fields):
                for a, b in zip(lhs.fields, rhs.fields):
                    self.unify_types(a, b, loc, lhs_loc, rhs_loc)
                return

        # unify(uint<?A>, uint<?B>) -> uint<?A>
        # unify(uint<?A>, uint<N>) -> uint<N>
        # unify(uint<N>, uint<?B>) -> uint<N>
        if isinstance(lhs, UIntType) and isinstance(rhs, UIntType):
            if isinstance(lhs.width, InferrableIntParam) and isinstance(
                    rhs.width, InferrableIntParam):
                rhs.width.inferred = lhs.width
                lhs.width.width_hint = max(lhs.width.width_hint,
                                           rhs.width.width_hint)
                return
            if isinstance(rhs.width, InferrableIntParam):
                rhs.width.inferred = lhs.width
                return
            if isinstance(lhs.width, InferrableIntParam):
                lhs.width.inferred = rhs.width
                return

        # unify({integer}, uint<_>) -> {integer}
        # unify(uint<_>, {integer}) -> {integer}
        if isinstance(lhs, GenericIntType) and isinstance(rhs, UIntType):
            return
        if isinstance(lhs, UIntType) and isinstance(rhs, GenericIntType):
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

        if isinstance(ty, UIntType):
            ty.width = self.simplify_param(ty.width)

        return ty

    # Simplify a parameter by replacing variables with their assigned value.
    def simplify_param(self, param: IntParam) -> IntParam:
        if isinstance(param, InferrableIntParam) and param.inferred:
            param.inferred = self.simplify_param(param.inferred)
            return param.inferred

        return param

    # Prove that the given parameter is greater than the given constant value.
    def prove_greater_than(
        self,
        param: IntParam,
        value: int,
    ) -> Tuple[bool, List[Tuple[IntParam, int]]]:
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

    # Convert an AST expression to a solver expression.
    def convert_to_solver_expr(self, expr: ast.Expr) -> z3.Ast:
        if isinstance(expr, ast.IntLitExpr):
            return z3.IntVal(expr.value)

        if isinstance(expr, ast.IdentExpr):
            binding = expr.binding.get()
            if isinstance(binding, ast.FnParam):
                param = self.params[binding]
                return self.solver_params[param]

        if isinstance(expr, ast.BinaryExpr):
            lhs = self.convert_to_solver_expr(expr.lhs)
            rhs = self.convert_to_solver_expr(expr.rhs)
            if expr.op == ast.BinaryOp.GT:
                return lhs > rhs

        emit_error(expr.loc, f"invalid constraint expression")


def typeck(root: ast.Root):
    root_typecks = []

    # Type-check all signature and prepare the necessary free parameters.
    for item in root.items:
        if isinstance(item, (ast.ModItem, ast.FnItem)):
            cx = Typeck(item)
            cx.typeck_signature()
            root_typecks.append((item, cx))
        else:
            emit_error(item.loc,
                       f"unsupported in typeck: {item.__class__.__name__}")
    assert len(root_typecks) == len(root.items)

    # Type-check the AST in full detail.
    for item, cx in root_typecks:
        cx.typeck()
