from __future__ import annotations

import re
from dataclasses import dataclass
from typing import *
from silicon.lexer import Token, TokenKind
from silicon import ast
from silicon.diagnostics import *


def parse_tokens(tokens: List[Token]) -> ast.Root:
    p = Parser(tokens=tokens, last_loc=tokens[0].loc)
    return parse_root(p)


@dataclass
class Parser:
    tokens: List[Token]
    last_loc: Loc

    def loc(self) -> Loc:
        return self.tokens[0].loc

    def consume(self) -> Token:
        t = self.tokens[0]
        self.tokens = self.tokens[1:]
        self.last_loc = t.loc
        return t

    def consume_if(self, kind: TokenKind) -> Optional[Token]:
        if self.tokens[0].kind == kind:
            return self.consume()
        return None

    def require(self, kind: TokenKind, msg: Optional[str] = None) -> Token:
        if token := self.consume_if(kind):
            return token
        msg = msg or kind.name
        emit_error(self.loc(),
                   f"expected {msg}, found {self.tokens[0].kind.name}")

    def isa(self, kind: TokenKind) -> bool:
        return self.tokens[0].kind == kind

    def not_delim(self, *args: TokenKind) -> bool:
        return self.tokens[0].kind not in (TokenKind.EOF, *args)


def parse_root(p: Parser) -> ast.Root:
    loc = p.loc()
    items: List[ast.Item] = []
    while not p.isa(TokenKind.EOF):
        items.append(parse_item(p))
    return ast.Root(loc=loc | p.last_loc, items=items)


def parse_item(p: Parser) -> ast.Item:
    # Parse module definitions.
    if kw := p.consume_if(TokenKind.KW_MOD):
        name = p.require(TokenKind.IDENT, "module name")
        p.require(TokenKind.LCURLY)
        stmts: List[ast.Stmt] = []
        while p.not_delim(TokenKind.RCURLY):
            stmts.append(parse_stmt(p))
        p.require(TokenKind.RCURLY)
        return ast.ModItem(loc=name.loc,
                           full_loc=kw.loc | p.last_loc,
                           name=name,
                           stmts=stmts)

    emit_error(p.loc(), f"expected item, found {p.tokens[0].kind.name}")


def parse_stmt(p: Parser) -> ast.Stmt:
    if kw := p.consume_if(TokenKind.KW_INPUT):
        name = p.require(TokenKind.IDENT, "input name")
        p.require(TokenKind.COLON)
        ty = parse_type(p)
        p.require(TokenKind.SEMICOLON)
        return ast.InputStmt(loc=name.loc,
                             full_loc=kw.loc | p.last_loc,
                             name=name,
                             ty=ty)

    if kw := p.consume_if(TokenKind.KW_OUTPUT):
        name = p.require(TokenKind.IDENT, "output name")
        p.require(TokenKind.COLON)
        ty = parse_type(p)
        p.require(TokenKind.ASSIGN)
        expr = parse_expr(p)
        p.require(TokenKind.SEMICOLON)
        return ast.OutputStmt(loc=name.loc,
                              full_loc=kw.loc | p.last_loc,
                              name=name,
                              ty=ty,
                              expr=expr)

    emit_error(p.loc(), f"expected statement, found {p.tokens[0].kind.name}")


def parse_type(p: Parser) -> ast.AstType:
    loc = p.loc()

    if p.isa(TokenKind.IDENT):
        name = p.tokens[0].spelling()
        if name == "uint":
            token = p.consume()
            p.require(TokenKind.LT)
            size = p.require(TokenKind.NUM_LIT)
            try:
                size_value = int(size.spelling())
            except:
                emit_error(size.loc, "invalid size for `uint`")
            p.require(TokenKind.GT)
            return ast.UIntType(loc=loc | p.last_loc,
                                size=size_value,
                                size_loc=size.loc)

    emit_error(p.loc(), f"expected type, found {p.tokens[0].kind.name}")


def parse_expr(p: Parser,
               min_prec: ast.Precedence = ast.Precedence.MIN) -> ast.Expr:
    expr = parse_primary_expr(p)

    # Parse infix operators.
    while True:
        if op := ast.BINARY_OPS.get(p.tokens[0].kind):
            prec = ast.BINARY_PRECEDENCE[op]
            if prec > min_prec:
                p.consume()
                rhs = parse_expr(p, prec)
                expr = ast.BinaryExpr(loc=expr.loc | rhs.loc,
                                      op=op,
                                      lhs=expr,
                                      rhs=rhs)
                continue
        break

    return expr


def parse_primary_expr(p: Parser) -> ast.Expr:
    # Parse number literals.
    if lit := p.consume_if(TokenKind.NUM_LIT):
        m = re.match(r'(\d+)u(\d+)', lit.spelling())
        if not m:
            emit_error(lit.loc,
                       f"expected number literal of the form `<V>u<N>`")
        return ast.IntLitExpr(loc=lit.loc, value=int(m[1]), width=int(m[2]))

    # Parse identifiers.
    if token := p.consume_if(TokenKind.IDENT):
        return ast.IdentExpr(loc=token.loc, name=token)

    emit_error(p.loc(), f"expected expression, found {p.tokens[0].kind.name}")
