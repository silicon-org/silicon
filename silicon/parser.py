from __future__ import annotations

import re
from dataclasses import dataclass
from typing import *
from copy import copy

from silicon.lexer import Token, TokenKind
from silicon import ast
from silicon.diagnostics import *
from silicon.source import *


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

    # Parse function definitions.
    if kw := p.consume_if(TokenKind.KW_FN):
        name = p.require(TokenKind.IDENT, "function name")

        # Parse arguments.
        p.require(TokenKind.LPAREN)
        args: List[ast.FnArg] = []
        while p.not_delim(TokenKind.RPAREN):
            arg_name = p.require(TokenKind.IDENT, "argument name")
            p.require(TokenKind.COLON)
            arg_type = parse_type(p)
            args.append(
                ast.FnArg(loc=arg_name.loc,
                          full_loc=arg_name.loc | p.last_loc,
                          name=arg_name,
                          ty=arg_type))
            if not p.consume_if(TokenKind.COMMA):
                break
        p.require(TokenKind.RPAREN)

        # Parse return type.
        return_ty: Optional[ast.AstType] = None
        if p.consume_if(TokenKind.ARROW):
            return_ty = parse_type(p)

        # Parse body.
        p.require(TokenKind.LCURLY)
        stmts = []
        while p.not_delim(TokenKind.RCURLY):
            stmts.append(parse_stmt(p))
        p.require(TokenKind.RCURLY)
        return ast.FnItem(loc=name.loc,
                          full_loc=kw.loc | p.last_loc,
                          name=name,
                          args=args,
                          return_ty=return_ty,
                          stmts=stmts)

    emit_error(p.loc(), f"expected item, found {p.tokens[0].kind.name}")


def parse_stmt_or_expr(p: Parser) -> ast.Stmt | ast.Expr:
    loc = p.loc()

    # Parse stray semicolons.
    if p.consume_if(TokenKind.SEMICOLON):
        return ast.EmptyStmt(loc=loc | p.last_loc, full_loc=loc | p.last_loc)

    # Parse input port declarations.
    if kw := p.consume_if(TokenKind.KW_INPUT):
        name = p.require(TokenKind.IDENT, "input name")
        p.require(TokenKind.COLON)
        ty = parse_type(p)
        p.require(TokenKind.SEMICOLON)
        return ast.InputStmt(loc=name.loc,
                             full_loc=loc | p.last_loc,
                             name=name,
                             ty=ty)

    # Parse output port declarations.
    if kw := p.consume_if(TokenKind.KW_OUTPUT):
        name = p.require(TokenKind.IDENT, "output name")
        p.require(TokenKind.COLON)
        ty = parse_type(p)
        maybeExpr: Optional[ast.Expr] = None
        if assign := p.consume_if(TokenKind.ASSIGN):
            maybeExpr = parse_expr(p)
        p.require(TokenKind.SEMICOLON)
        return ast.OutputStmt(loc=name.loc,
                              full_loc=loc | p.last_loc,
                              name=name,
                              ty=ty,
                              expr=maybeExpr)

    # Parse let bindings.
    if kw := p.consume_if(TokenKind.KW_LET):
        name = p.require(TokenKind.IDENT, "output name")
        maybeTy = None
        if colon := p.consume_if(TokenKind.COLON):
            maybeTy = parse_type(p)
        maybeExpr = None
        if assign := p.consume_if(TokenKind.ASSIGN):
            maybeExpr = parse_expr(p)
        p.require(TokenKind.SEMICOLON)
        return ast.LetStmt(loc=name.loc,
                           full_loc=loc | p.last_loc,
                           name=name,
                           ty=maybeTy,
                           expr=maybeExpr)

    # Parse return statements.
    if kw := p.consume_if(TokenKind.KW_RETURN):
        maybeExpr = None
        if not p.isa(TokenKind.SEMICOLON):
            maybeExpr = parse_expr(p)
        p.require(TokenKind.SEMICOLON)
        return ast.ReturnStmt(loc=kw.loc,
                              full_loc=loc | p.last_loc,
                              expr=maybeExpr)

    # Otherwise this is a statement that starts with an expression.
    expr = parse_expr(p)
    if p.consume_if(TokenKind.SEMICOLON):
        return ast.ExprStmt(loc=expr.loc, full_loc=loc | p.last_loc, expr=expr)

    # Parse assignments.
    if assign := p.consume_if(TokenKind.ASSIGN):
        rhs = parse_expr(p)
        p.require(TokenKind.SEMICOLON)
        return ast.AssignStmt(loc=assign.loc,
                              full_loc=loc | p.last_loc,
                              lhs=expr,
                              rhs=rhs)

    return expr


def parse_stmt(p: Parser) -> ast.Stmt:
    stmt = parse_stmt_or_expr(p)
    if isinstance(stmt, ast.Stmt):
        return stmt
    if expr_requires_semicolon(stmt):
        p.require(TokenKind.SEMICOLON)
    return ast.ExprStmt(loc=stmt.loc,
                        full_loc=stmt.loc | p.last_loc,
                        expr=stmt)


# Check whether an expression requires a trailing semicolon if used as a
# statement in a block.
def expr_requires_semicolon(expr: ast.Expr) -> bool:
    return not isinstance(expr, ast.BlockExpr)


def parse_type(p: Parser) -> ast.AstType:
    loc = p.loc()

    # Parse tuples or the `()` unit type.
    if p.consume_if(TokenKind.LPAREN):
        if p.consume_if(TokenKind.RPAREN):
            return ast.UnitType(loc=loc | p.last_loc)
        fields: List[ast.AstType] = []
        while p.not_delim(TokenKind.RPAREN):
            fields.append(parse_type(p))
            if not p.consume_if(TokenKind.COMMA):
                break
        p.require(TokenKind.RPAREN)
        if len(fields) < 2:
            emit_error(loc | p.last_loc, "tuple requires at least two fields")
        return ast.TupleType(loc=loc | p.last_loc, fields=fields)

    # Parse types that start with a name.
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

        if name == "Wire":
            token = p.consume()
            p.require(TokenKind.LT)
            inner = parse_type(p)
            p.require(TokenKind.GT)
            return ast.WireType(loc=loc | p.last_loc, inner=inner)

        if name == "Reg":
            token = p.consume()
            p.require(TokenKind.LT)
            inner = parse_type(p)
            p.require(TokenKind.GT)
            return ast.RegType(loc=loc | p.last_loc, inner=inner)

    emit_error(p.loc(), f"expected type, found {p.tokens[0].kind.name}")


def parse_expr(
    p: Parser,
    min_prec: ast.Precedence = ast.Precedence.MIN,
) -> ast.Expr:
    expr = parse_prefix_expr(p)
    while infix := parse_infix_expr(p, expr, min_prec):
        expr = infix
    return expr


def parse_prefix_expr(p: Parser) -> ast.Expr:
    loc = p.loc()

    # Parse unary operators.
    if op := ast.UNARY_OPS.get(p.tokens[0].kind):
        op_loc = p.consume().loc
        arg = parse_prefix_expr(p)
        return ast.UnaryExpr(
            loc=op_loc,
            full_loc=loc | arg.loc,
            op=op,
            arg=arg,
        )

    # Otherwise parse a primary expression and optional suffix expressions.
    expr = parse_primary_expr(p)
    while suffix := parse_suffix_expr(p, expr):
        expr = suffix
    return expr


def parse_infix_expr(
    p: Parser,
    lhs: ast.Expr,
    min_prec: ast.Precedence,
) -> Optional[ast.Expr]:
    # Parse binary operators.
    if op := ast.BINARY_OPS.get(p.tokens[0].kind):
        prec = ast.BINARY_PRECEDENCE[op]
        if prec > min_prec:
            op_loc = p.consume().loc
            rhs = parse_expr(p, prec)
            return ast.BinaryExpr(
                loc=op_loc,
                full_loc=lhs.loc | rhs.loc,
                op=op,
                lhs=lhs,
                rhs=rhs,
            )

    return None


def parse_suffix_expr(p: Parser, expr: ast.Expr) -> Optional[ast.Expr]:
    # Parse field calls, such as `a.b(c)`, or tuple field accesses.
    if dot := p.consume_if(TokenKind.DOT):
        # a.b(c)
        if name := p.consume_if(TokenKind.IDENT):
            p.require(TokenKind.LPAREN)
            args: List[ast.Expr] = []
            while p.not_delim(TokenKind.RPAREN):
                args.append(parse_expr(p))
                if not p.consume_if(TokenKind.COMMA):
                    break
            p.require(TokenKind.RPAREN)
            return ast.FieldCallExpr(
                loc=name.loc,
                full_loc=expr.loc | p.last_loc,
                target=expr,
                name=name,
                args=args,
            )

        # a.42
        if index := p.consume_if(TokenKind.NUM_LIT):
            try:
                index_value = int(index.spelling())
            except:
                emit_error(index.loc, "invalid tuple index")
            return ast.TupleFieldExpr(loc=index.loc,
                                      full_loc=expr.loc | p.last_loc,
                                      target=expr,
                                      field=index_value)

        emit_error(dot.loc,
                   f"expected field name, found {p.tokens[0].kind.name}")

    return None


def parse_primary_expr(p: Parser) -> ast.Expr:
    loc = p.loc()

    # Parse number literals.
    if lit := p.consume_if(TokenKind.NUM_LIT):
        m = re.match(r'(\d+)(u(\d+))?', lit.spelling())
        if not m:
            emit_error(
                lit.loc,
                f"expected number literal of the form `<V>` or `<V>u<N>`")
        return ast.IntLitExpr(
            loc=lit.loc,
            full_loc=lit.loc,
            value=int(m[1]),
            width=int(m[3]) if m[2] else None,
        )

    # Parse identifiers and calls.
    if token := p.consume_if(TokenKind.IDENT):
        if paren := p.consume_if(TokenKind.LPAREN):
            args: List[ast.Expr] = []
            while p.not_delim(TokenKind.RPAREN):
                args.append(parse_expr(p))
                if not p.consume_if(TokenKind.COMMA):
                    break
            p.require(TokenKind.RPAREN)
            return ast.CallExpr(
                loc=token.loc,
                full_loc=token.loc | p.last_loc,
                name=token,
                args=args,
            )

        return ast.IdentExpr(loc=token.loc, full_loc=token.loc, name=token)

    # Parse parenthesized expressions, tuples, or `()` unit literals.
    if p.consume_if(TokenKind.LPAREN):
        if p.consume_if(TokenKind.RPAREN):
            return ast.UnitLitExpr(loc=loc | p.last_loc,
                                   full_loc=loc | p.last_loc)
        fields: List[ast.Expr] = []
        while p.not_delim(TokenKind.RPAREN):
            fields.append(parse_expr(p))
            if not p.consume_if(TokenKind.COMMA):
                break
        p.require(TokenKind.RPAREN)
        if len(fields) == 1:
            return ast.ParenExpr(loc=fields[0].loc,
                                 full_loc=loc | p.last_loc,
                                 expr=fields[0])
        else:
            return ast.TupleExpr(loc=loc | p.last_loc,
                                 full_loc=loc | p.last_loc,
                                 fields=fields)

    # Parse blocks.
    if p.isa(TokenKind.LCURLY):
        return parse_block_expr(p)

    emit_error(p.loc(), f"expected expression, found {p.tokens[0].kind.name}")


def parse_block_expr(p: Parser) -> ast.BlockExpr:
    loc = p.loc()
    stmts: List[ast.Stmt] = []
    result: Optional[ast.Expr] = None
    p.require(TokenKind.LCURLY)

    while p.not_delim(TokenKind.RCURLY):
        stmt = parse_stmt_or_expr(p)

        # If the statement was an expression, this may be the final expression
        # in the block which defines the block's result value.
        if isinstance(stmt, ast.Expr):
            # If this was an expression without subsequent semicolon, and the
            # next token is the closing brace `}`, this expression is the result
            # value of the block.
            if p.isa(TokenKind.RCURLY):
                result = stmt
                break

            # Otherwise the block isn't done yet. Complain about the missing
            # semicolon.
            if expr_requires_semicolon(stmt):
                after = copy(p.last_loc)
                after.offset += after.length
                after.length = 0
                emit_error(after, "expected `;` after expression")

            # Wrap this expression up as a statement, since there are more left
            # in the block.
            stmt = ast.ExprStmt(loc=stmt.loc,
                                full_loc=stmt.full_loc,
                                expr=stmt)

        stmts.append(stmt)

    p.require(TokenKind.RCURLY)
    return ast.BlockExpr(loc=loc | p.last_loc,
                         full_loc=loc | p.last_loc,
                         stmts=stmts,
                         result=result)
