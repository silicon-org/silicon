from __future__ import annotations

from math import log2, floor
from typing import Optional
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
class UIntType(Type):
    width: int

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


@dataclass
class InferrableUIntType(Type):
    # A unique integer identifier.
    num: int = field(default_factory=count().__next__)

    # The type this uint has been inferred to.
    inferred: Type | None = None

    # The minimum number of bits needed by the literal if this type remains
    # uninferred.
    width_hint: int = 0

    def __eq__(self, other) -> bool:
        return id(self) == id(other)

    def __str__(self) -> str:
        return f"uint<?{self.num}>"


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


@dataclass
class Typeck:

    # Convert an AST type node into an actual type.
    def convert_ast_type(self, ty: ast.AstType) -> Type:
        if isinstance(ty, ast.UIntType):
            return UIntType(ty.size)
        if isinstance(ty, ast.WireType):
            return WireType(self.convert_ast_type(ty.inner))
        if isinstance(ty, ast.RegType):
            return RegType(self.convert_ast_type(ty.inner))
        emit_error(ty.loc, f"unsupported type: `{ty.loc.spelling()}`")

    def typeck_mod(self, mod: ast.ModItem):
        for stmt in mod.stmts:
            self.typeck_stmt(stmt)

        # Finalize the types in the AST. This replaces all type variables with
        # the actual type that was inferred.
        for node in mod.walk(ast.WalkOrder.PostOrder):
            if hasattr(node, "fty"):
                if node.fty is None:
                    emit_error(node.loc,
                               f"uninferred type: {node.__class__.__name__}")
                ty = self.simplify_type(node.fty)

                # Make uninferred uints the minimum size for their literals.
                if isinstance(ty, InferrableUIntType):
                    ty.inferred = UIntType(ty.width_hint)
                    ty = ty.inferred

                node.fty = ty
            self.finalize_node(node)

    def typeck_stmt(self, stmt: ast.Stmt):
        if isinstance(stmt, ast.ExprStmt):
            self.typeck_expr(stmt.expr)
            return

        if isinstance(stmt, ast.InputStmt):
            stmt.fty = self.convert_ast_type(stmt.ty)
            return

        if isinstance(stmt, ast.OutputStmt):
            stmt.fty = self.convert_ast_type(stmt.ty)
            if stmt.expr:
                self.unify_types(stmt.fty, self.typeck_expr(stmt.expr),
                                 stmt.loc)
            return

        if isinstance(stmt, ast.LetStmt):
            if stmt.ty:
                stmt.fty = self.convert_ast_type(stmt.ty)
            else:
                stmt.fty = InferrableType()
            if stmt.expr:
                self.unify_types(stmt.fty, self.typeck_expr(stmt.expr),
                                 stmt.loc)
            return

        if isinstance(stmt, ast.AssignStmt):
            lhs = self.typeck_expr(stmt.lhs)
            rhs = self.typeck_expr(stmt.rhs)
            self.unify_types(lhs, rhs, stmt.loc)
            return

        emit_error(stmt.loc,
                   f"unsupported in typeck: {stmt.__class__.__name__}")

    def typeck_expr(self, expr: ast.Expr) -> Type:
        expr.fty = self.type_of_expr(expr)
        return expr.fty

    def type_of_expr(self, expr: ast.Expr) -> Type:
        if isinstance(expr, ast.IntLitExpr):
            if expr.width is None:
                return InferrableUIntType(
                    width_hint=min_bits_for_int(expr.value))
            else:
                return UIntType(expr.width)

        if isinstance(expr, ast.IdentExpr):
            target = expr.binding.get()
            if hasattr(target, "fty") and target.fty and isinstance(
                    target.fty, Type):
                return target.fty
            emit_info(target.loc,
                      f"name `{expr.name.spelling()}` defined here")
            emit_error(expr.loc, f"`{expr.name.spelling()}` has no known type")

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

        if name == "concat":
            width = 0
            for arg in expr.args:
                ty = self.typeck_expr(arg)
                if not isinstance(ty, UIntType):
                    emit_error(arg.loc, f"cannot concatenate `{ty}`")
                width += ty.width
            return UIntType(width)

        if name == "wire":
            require_num_args(expr, 1)
            return WireType(self.typeck_expr(expr.args[0]))

        if name == "reg":
            require_num_args(expr, 2)
            clock = self.typeck_expr(expr.args[0])
            self.unify_types(UIntType(1),
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
            if offset < 0 or offset >= target.width:
                emit_info(
                    expr.target.loc,
                    f"the bit index `{offset}` is outside the type `{target}` whose bit range is `0..{target.width}`"
                )
                emit_error(expr.args[0].loc,
                           f"bit index out of bounds for `{target}`")
            return UIntType(1)

        if name == "slice":
            require_num_args(expr, 2)
            offset = require_int_lit(expr, 0)
            width = require_int_lit(expr, 1, min=0)
            target = self.typeck_expr(expr.target)
            self.typeck_expr(expr.args[0])
            self.typeck_expr(expr.args[1])
            if not isinstance(target, UIntType):
                emit_error(expr.loc, f"cannot slice `{target}`")
            if offset + width > target.width:
                emit_info(
                    expr.args[0].loc | expr.args[1].loc,
                    f"the slice `{offset}..{offset+width}` is outside the type `{target}` whose bit range is `0..{target.width}`"
                )
                emit_error(expr.args[0].loc | expr.args[1].loc,
                           f"slice out of bounds for `{target}`")
            return UIntType(width)

        if name == "mux":
            require_num_args(expr, 2)
            target = self.typeck_expr(expr.target)
            self.unify_types(UIntType(1),
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

    def finalize_node(self, node: ast.AstNode):
        # Make sure that integer literals fit into their inferred type.
        if isinstance(node, ast.IntLitExpr) and node.width is None:
            width = min_bits_for_int(node.value)
            if isinstance(node.fty, UIntType) and width > node.fty.width:
                emit_info(
                    node.loc,
                    f"the literal `{node.value}` does not fit into the type `{node.fty}` since it requires at least {width} bits"
                )
                emit_error(node.loc, f"literal out of range for `{node.fty}`")

        # Unary and binary operators must operate on integers.
        if isinstance(node, ast.UnaryExpr) or isinstance(node, ast.BinaryExpr):
            if not isinstance(node.fty, UIntType):
                word = {
                    ast.UnaryOp.NEG: "negate",
                    ast.UnaryOp.NOT: "invert",
                    ast.BinaryOp.ADD: "add",
                    ast.BinaryOp.SUB: "subtract",
                }.get(node.op, node.op.name)
                emit_error(node.loc, f"cannot {word} `{node.fty}`")

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

        # unify(uint<?A>, uint<?B>) -> uint<?A>
        # unify(uint<?A>, uint<N>) -> uint<N>
        # unify(uint<N>, uint<?B>) -> uint<N>
        if isinstance(lhs, InferrableUIntType) and isinstance(
                rhs, InferrableUIntType):
            rhs.inferred = lhs
            lhs.width_hint = max(lhs.width_hint, rhs.width_hint)
            return
        if isinstance(lhs, InferrableUIntType) and isinstance(rhs, UIntType):
            lhs.inferred = rhs
            return
        if isinstance(lhs, UIntType) and isinstance(rhs, InferrableUIntType):
            rhs.inferred = lhs
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
        if isinstance(ty, InferrableUIntType) and ty.inferred:
            ty.inferred = self.simplify_type(ty.inferred)
            return ty.inferred

        if isinstance(ty, InferrableType) and ty.inferred:
            ty.inferred = self.simplify_type(ty.inferred)
            return ty.inferred

        return ty


def typeck(root: ast.Root):
    for item in root.items:
        if isinstance(item, ast.ModItem):
            Typeck().typeck_mod(item)
        else:
            emit_error(item.loc,
                       f"unsupported in typeck: {item.__class__.__name__}")
