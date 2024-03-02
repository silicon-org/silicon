from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from enum import Enum, auto
from silicon.diagnostics import *
from silicon.source import *
from typing import *

__all__ = [
    "Token",
    "TokenKind",
    "tokenize_file",
]


class TokenKind(Enum):
    IDENT = auto()
    NUM_LIT = auto()

    LCURLY = auto()
    RCURLY = auto()
    LPAREN = auto()
    RPAREN = auto()
    LBRACK = auto()
    RBRACK = auto()

    DOT = auto()
    COMMA = auto()
    COLON = auto()
    SEMICOLON = auto()
    NOT = auto()

    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()

    ADD = auto()
    SUB = auto()

    ASSIGN = auto()

    KW_ELSE = auto()
    KW_FOR = auto()
    KW_IF = auto()
    KW_INPUT = auto()
    KW_LET = auto()
    KW_MOD = auto()
    KW_OUTPUT = auto()
    KW_RETURN = auto()
    KW_WHILE = auto()

    EOF = auto()


@dataclass
class Token:
    loc: Loc
    kind: TokenKind

    def spelling(self) -> str:
        return self.loc.spelling()


SYMBOLS1: Dict[str, TokenKind] = {
    "{": TokenKind.LCURLY,
    "}": TokenKind.RCURLY,
    "(": TokenKind.LPAREN,
    ")": TokenKind.RPAREN,
    "[": TokenKind.LBRACK,
    "]": TokenKind.RBRACK,
    ".": TokenKind.DOT,
    ",": TokenKind.COMMA,
    ":": TokenKind.COLON,
    ";": TokenKind.SEMICOLON,
    "!": TokenKind.NOT,
    "<": TokenKind.LT,
    ">": TokenKind.GT,
    "=": TokenKind.ASSIGN,
    "+": TokenKind.ADD,
    "-": TokenKind.SUB,
}

SYMBOLS2: Dict[str, TokenKind] = {
    "==": TokenKind.EQ,
    "!=": TokenKind.NE,
    "<=": TokenKind.LE,
    ">=": TokenKind.GE,
}

KEYWORDS: Dict[str, TokenKind] = {
    "else": TokenKind.KW_ELSE,
    "for": TokenKind.KW_FOR,
    "if": TokenKind.KW_IF,
    "input": TokenKind.KW_INPUT,
    "let": TokenKind.KW_LET,
    "mod": TokenKind.KW_MOD,
    "output": TokenKind.KW_OUTPUT,
    "return": TokenKind.KW_RETURN,
    "while": TokenKind.KW_WHILE,
}


def tokenize_file(path: str) -> List[Token]:
    with open(path, "r") as f:
        file = SourceFile(path, f.read())
    return tokenize(file)


def tokenize(file: SourceFile) -> List[Token]:
    lexer = Lexer(loc=Loc(file, 0, 0),
                  text=file.contents,
                  total_length=len(file.contents),
                  tokens=[])
    while len(lexer.text) > 0:
        tokenize_next(lexer)
    lexer.reset_loc()
    lexer.emit(TokenKind.EOF)
    return lexer.tokens


@dataclass
class Lexer:
    loc: Loc
    text: str
    total_length: int
    tokens: List[Token]

    def reset_loc(self):
        self.loc.offset = self.total_length - len(self.text)
        self.loc.length = 0

    def get_loc(self) -> Loc:
        loc = copy(self.loc)
        loc.length = self.total_length - len(self.text) - loc.offset
        return loc

    def consume(self, num: int = 1):
        self.text = self.text[num:]

    def consume_while(self, predicate: Callable[[str], bool]) -> bool:
        if len(self.text) == 0 or not predicate(self.text[0]):
            return False
        self.consume()
        while len(self.text) > 0 and predicate(self.text[0]):
            self.consume()
        return True

    def emit(self, kind: TokenKind):
        self.tokens.append(Token(loc=self.get_loc(), kind=kind))


def is_whitespace(c: str) -> bool:
    return c in " \t\n\r"


def is_digit(c: str) -> bool:
    return c >= "0" and c <= "9"


def is_ident_start(c: str) -> bool:
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z") or c == "_"


def is_ident(c: str) -> bool:
    return is_ident_start(c) or is_digit(c)


def tokenize_next(lex: Lexer):
    # Skip whitespace.
    if lex.consume_while(is_whitespace):
        return

    # Skip single-line comments.
    if lex.text[:2] == "//":
        lex.consume_while(lambda x: x != "\n")
        return

    # Skip multi-line comments.
    if lex.text[:2] == "/*":
        lex.reset_loc()
        lex.consume(2)
        while len(lex.text) > 0 and lex.text[:2] != "*/":
            lex.consume()
        if lex.text[:2] != "*/":
            emit_error(lex.get_loc(), "unclosed comment; missing `*/`")
        lex.consume(2)
        return

    lex.reset_loc()

    # Parse symbols.
    if kind := SYMBOLS2.get(lex.text[:2]):
        lex.consume(2)
        lex.emit(kind)
        return

    if kind := SYMBOLS1.get(lex.text[:1]):
        lex.consume(1)
        lex.emit(kind)
        return

    # Parse identifiers.
    if is_ident_start(lex.text[0]):
        lex.consume_while(is_ident)
        kind = KEYWORDS.get(lex.get_loc().spelling()) or TokenKind.IDENT
        lex.emit(kind)
        return

    # Parse number literals.
    if is_digit(lex.text[0]):
        lex.consume_while(is_digit)
        if lex.text[0] == "u":
            lex.consume(1)
            lex.consume_while(is_digit)
        lex.emit(TokenKind.NUM_LIT)
        return

    # If we get here, this character is not supported.
    emit_error(lex.get_loc(), f"unknown character `{lex.text[0]}`")
