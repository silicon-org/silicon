import argparse
from silicon import ast
from silicon.ast import dump_ast
from silicon.codegen import codegen
from silicon.lexer import tokenize_file
from silicon.names import resolve_names
from silicon.parser import parse_tokens
from sys import stdout


def main():
    # Parse command line arguments.
    parser = argparse.ArgumentParser(
        description="An experimental hardware compiler.")

    parser.add_argument("input",
                        metavar="INPUT",
                        help="Source file to compile")

    parser.add_argument("--dump-tokens",
                        action="store_true",
                        help="Dump lexed tokens and exit")

    parser.add_argument("--dump-ast",
                        action="store_true",
                        help="Dump parsed syntax and exit")

    parser.add_argument("--dump-resolved",
                        action="store_true",
                        help="Dump syntax after name resolution and exit")

    args = parser.parse_args()

    # Tokenize the input.
    tokens = tokenize_file(args.input)
    if args.dump_tokens:
        for token in tokens:
            print(f"- {token.kind.name}: `{token.loc.spelling()}`")
        return

    # Parse the tokens into an AST.
    root = parse_tokens(tokens)
    if args.dump_ast:
        print(dump_ast(root))
        return

    # Resolve the names in the AST.
    resolve_names(root)
    if args.dump_resolved:
        print(dump_ast(root))
        return

    # Emit CIRCT code.
    codegen(root)
