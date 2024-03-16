import argparse
import re
from sys import exit, stdout

from silicon import ast
from silicon.transforms.unroll import UnrollPass
from silicon.ast import dump_ast
from silicon.codegen import codegen
from silicon.lexer import tokenize
from silicon.names import resolve_names
from silicon.parser import parse_tokens
from silicon.source import SourceFile
from silicon.diagnostics import DiagnosticException
from silicon.ir import convert_ast_to_ir
from silicon.ty import typeck

from xdsl.ir import MLContext


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

    parser.add_argument("--dump-types",
                        action="store_true",
                        help="Dump syntax after type checking and exit")

    parser.add_argument("--dump-ir",
                        action="store_true",
                        help="Dump IR and exit")

    parser.add_argument("--dump-final-ir",
                        action="store_true",
                        help="Dump IR before codegeneration and exit")

    parser.add_argument("--split-input-file",
                        action="store_true",
                        help="Split input by `// -----` delimiter")

    args = parser.parse_args()

    # If the user requested the input file to be split along a delimiter, read
    # the file, split it into chunks, and run each through the compiler
    # separately.
    if args.split_input_file:
        with open(args.input, "r") as f:
            splits = re.split(r'[ ]*//[ ]*-----.*$',
                              f.read(),
                              flags=re.MULTILINE)
        any_failed = False
        padding = ""
        for i, split in enumerate(splits):
            try:
                if i > 0:
                    print("// -----")
                file = SourceFile(args.input, padding + split)
                process_input(args, file)
            except DiagnosticException:
                any_failed = True
            padding += "\n" * split.count("\n")
        if any_failed:
            exit(1)
    else:
        with open(args.input, "r") as f:
            file = SourceFile(args.input, f.read())
        try:
            process_input(args, file)
        except DiagnosticException:
            exit(1)


def process_input(args: argparse.Namespace, input: SourceFile):
    # Tokenize the input.
    tokens = tokenize(input)
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

    # Infer and check types.
    typeck(root)
    if args.dump_types:
        print(dump_ast(root))
        return

    # Convert the AST to our IR.
    ir = convert_ast_to_ir(root)
    if args.dump_ir:
        print(ir)
        return

    # Run the lowering pipeline.
    ctx = MLContext()
    UnrollPass().apply(ctx, ir)
    if args.dump_final_ir:
        print(ir)
        return

    # Emit CIRCT code.
    codegen(root)
