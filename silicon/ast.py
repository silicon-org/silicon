from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass
from typing import *
from silicon.lexer import Token, TokenKind
from silicon.source import Loc
from enum import IntEnum, auto


@dataclass
class AstNode:
    loc: Loc


@dataclass
class Root(AstNode):
    items: List[Item]


#===------------------------------------------------------------------------===#
# Items
#===------------------------------------------------------------------------===#


@dataclass
class Item(AstNode):
    pass


@dataclass
class ModItem(Item):
    full_loc: Loc
    name: Token
    stmts: List[Stmt]


#===------------------------------------------------------------------------===#
# Statements
#===------------------------------------------------------------------------===#


@dataclass
class Stmt(AstNode):
    full_loc: Loc


@dataclass
class InputStmt(Stmt):
    name: Token
    ty: AstType


@dataclass
class OutputStmt(Stmt):
    name: Token
    ty: AstType
    expr: Expr


#===------------------------------------------------------------------------===#
# Types
#===------------------------------------------------------------------------===#


@dataclass
class AstType(AstNode):
    pass


@dataclass
class UIntType(AstType):
    size: int
    size_loc: Loc


#===------------------------------------------------------------------------===#
# Expressions
#===------------------------------------------------------------------------===#


@dataclass
class Expr(AstNode):
    pass


@dataclass
class IntLitExpr(Expr):
    value: int
    width: int


@dataclass
class BinaryExpr(Expr):
    op: BinaryOp
    lhs: Expr
    rhs: Expr


class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()


# https://en.cppreference.com/w/c/language/operator_precedence
class Precedence(IntEnum):
    MIN = auto()
    OR = auto()
    XOR = auto()
    AND = auto()
    EQUALITY = auto()
    RELATIONAL = auto()
    ADD_SUB = auto()
    MUL_DIV = auto()
    PRIMARY = auto()
    MAX = auto()


BINARY_OPS: Dict[TokenKind, BinaryOp] = {
    TokenKind.ADD: BinaryOp.ADD,
    TokenKind.SUB: BinaryOp.SUB,
}

BINARY_PRECEDENCE: Dict[BinaryOp, Precedence] = {
    BinaryOp.ADD: Precedence.ADD_SUB,
    BinaryOp.SUB: Precedence.ADD_SUB,
}

#===------------------------------------------------------------------------===#
# Dumping
#===------------------------------------------------------------------------===#


def dump_ast(node: AstNode) -> str:

    def dump_field(name: str, value) -> List[str]:
        if isinstance(value, AstNode):
            return [dump_inner(value, name)]
        elif isinstance(value, list):
            fields = []
            for i, v in enumerate(value):
                fields += dump_field(f"{name}[{i}]", v)
            return fields
        return []

    def dump_inner(node: AstNode, field_prefix: str) -> str:
        line = ""
        if field_prefix:
            line += f"{field_prefix}: "
        line += node.__class__.__name__
        for name, value in node.__dict__.items():
            if isinstance(value, str):
                line += f" {name}=\"{value}\""
            elif isinstance(value, int):
                line += f" {name}={value}"
            elif isinstance(value, Token):
                line += f" \"{value.spelling()}\""
            elif isinstance(value, BinaryOp):
                line += f" {value.name}"
        fields = []
        for name, value in node.__dict__.items():
            fields += dump_field(name, value)
        for i, field in enumerate(fields):
            is_last = (i + 1 == len(fields))
            sep_first = "`-" if is_last else "|-"
            sep_rest = "  " if is_last else "| "
            line += "\n" + sep_first + field.replace("\n", "\n" + sep_rest)
        return line

    return dump_inner(node, "")
