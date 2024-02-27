from typing import NoReturn
from sys import stderr
from silicon.source import *
from termcolor import colored

__all__ = [
    "emit_error",
    "emit_info",
    "emit_diagnostic",
]


def emit_error(loc: Loc, msg: str) -> NoReturn:
    emit_diagnostic("error", "red", loc, msg)
    raise DiagnosticException()


def emit_info(loc: Loc, msg: str):
    emit_diagnostic("info", "cyan", loc, msg)


def emit_diagnostic(severity: str, color: str, loc: Loc, msg: str):
    text = colored(severity + ":", color, attrs=["bold"])
    text += " "
    text += colored(msg, attrs=["bold"])
    print(text, file=stderr)
    print(f"{loc}:", file=stderr)

    src_before = loc.file.contents[:loc.offset].split("\n")[-1]
    src_within = loc.file.contents[loc.offset:loc.offset + loc.length]
    src_after = loc.file.contents[loc.offset + loc.length:].split("\n")[0]

    text = "  | " + src_before
    text += colored(src_within, color, attrs=["bold"])
    text += src_after
    print(text, file=stderr)

    text = "  | " + " " * len(src_before)
    text += colored("^" * max(len(src_within), 1), color, attrs=["bold"])
    print(text, file=stderr)


class DiagnosticException(Exception):
    pass
