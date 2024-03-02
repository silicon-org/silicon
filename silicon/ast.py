from __future__ import annotations

from enum import Enum, auto
from dataclasses import dataclass, field
from typing import *
from silicon.lexer import Token, TokenKind
from silicon.source import Loc
from enum import Enum, IntEnum, auto


class WalkOrder(Enum):
    PreOrder = auto()
    PostOrder = auto()


@dataclass
class AstNode:
    loc: Loc

    # Walk over this `AstNode` and all its children. If order is `PreOrder`,
    # parent nodes are visited before their child nodes. If order is
    # `PostOrder`, parent nodes are visited after their child nodes.
    def walk(
        self,
        order: WalkOrder = WalkOrder.PreOrder
    ) -> Generator[AstNode, None, None]:
        if order == WalkOrder.PreOrder:
            yield self
        for child in self.children():
            yield from child.walk(order)
        if order == WalkOrder.PostOrder:
            yield self

    # Get child `AstNode`s of this node. This visits all fields of the current
    # node, looking into nested lists, sets, and dictionaries, and produces all
    # `AstNode`s it finds.
    def children(self) -> Generator[AstNode, None, None]:
        for value in self.__dict__.values():
            yield from walk_nodes(value)


# Walk the `AstNode`s contained in a nested set of lists, sets, and
# dictionaries.
def walk_nodes(value) -> Generator[AstNode, None, None]:
    if isinstance(value, AstNode):
        yield value
    elif isinstance(value, list) or isinstance(value, set):
        for v in value:
            yield from walk_nodes(v)
    elif isinstance(value, dict):
        for v in value.values():
            yield from walk_nodes(v)


@dataclass
class Root(AstNode):
    items: List[Item]


# A binding from something like an identifier expression to the the declaration
# of that name.
@dataclass
class Binding:
    target: Optional[AstNode] = None

    # Get the target node of the binding, asserting that the binding has been
    # resolved.
    def get(self) -> AstNode:
        assert self.target is not None
        return self.target


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
    expr: Optional[Expr]


@dataclass
class LetStmt(Stmt):
    name: Token
    ty: AstType
    expr: Optional[Expr]


@dataclass
class ExprStmt(Stmt):
    expr: Expr


@dataclass
class AssignStmt(Stmt):
    lhs: Expr
    rhs: Expr


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
class IdentExpr(Expr):
    name: Token
    binding: Binding = field(default_factory=Binding)


@dataclass
class IntLitExpr(Expr):
    value: int
    width: Optional[int]


@dataclass
class UnaryExpr(Expr):
    op: UnaryOp
    arg: Expr


class UnaryOp(Enum):
    NEG = auto()
    NOT = auto()


UNARY_OPS: Dict[TokenKind, UnaryOp] = {
    TokenKind.SUB: UnaryOp.NEG,
    TokenKind.NOT: UnaryOp.NOT,
}


@dataclass
class BinaryExpr(Expr):
    op: BinaryOp
    lhs: Expr
    rhs: Expr


class BinaryOp(Enum):
    ADD = auto()
    SUB = auto()


BINARY_OPS: Dict[TokenKind, BinaryOp] = {
    TokenKind.ADD: BinaryOp.ADD,
    TokenKind.SUB: BinaryOp.SUB,
}


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


BINARY_PRECEDENCE: Dict[BinaryOp, Precedence] = {
    BinaryOp.ADD: Precedence.ADD_SUB,
    BinaryOp.SUB: Precedence.ADD_SUB,
}

#===------------------------------------------------------------------------===#
# Dumping
#===------------------------------------------------------------------------===#


def dump_ast(node: AstNode) -> str:
    ids: Dict[int, int] = {}

    def get_id(node: AstNode) -> int:
        if id(node) not in ids:
            ids[id(node)] = len(ids)
        return ids[id(node)]

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
            elif isinstance(value, Enum):
                line += f" {value.name}"
            elif isinstance(value, Binding) and value.target is not None:
                line += f" {name}={value.target.__class__.__name__}(@{get_id(value.get())})"
        line += f" @{get_id(node)}"
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
