#!/usr/bin/env python3

# # Collect Documentation
#
# Copies a documentation file from a source to a destination, applying
# transformations needed for Hugo along the way:
# - Strips `[TOC]` markers (Hugo Book renders TOC automatically).
# - Injects YAML front matter if missing. The title is extracted from the first
#   `# Heading` line, falling back to the filename stem.
# - Optionally checks ```silicon code blocks by compiling them with `silc`.

import argparse
import re
import subprocess
import sys
from pathlib import Path
from highlight import highlight_blocks

parser = argparse.ArgumentParser()
parser.add_argument("src", type=Path)
parser.add_argument("dst", type=Path)
parser.add_argument("--silc", type=Path, default=None,
                    help="Path to silc binary; enables silicon code block checking")
args = parser.parse_args()

text = args.src.read_text()

# Strip [TOC] markers. Hugo Book renders the table of contents automatically.
text = re.sub(r"^\[TOC\]\s*$", "", text, flags=re.MULTILINE)

# Ensure YAML front matter with a title. If front matter already exists, strip
# it off; otherwise start with an empty front matter. Then ensure a title line
# is present, guessing from the first `# Heading` or the filename stem. Finally
# reassemble the document with the front matter.
match = re.match(r"^---\n(.+?)^---\n", text, re.MULTILINE | re.DOTALL)
if match:
    front = match.group(1)
    text = text[len(match.group(0)):]
else:
    front = ""
if not re.search(r"^title\s*:", front, re.MULTILINE):
    title = args.dst.stem
    heading = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if heading:
        title = heading.group(1).strip()
    front += f"title: \"{title}\"\n"
text = f"---\n{front}---\n{text}"

# If `--silc` is provided and the source file is not under a `design/`
# directory, extract all ```silicon fenced code blocks from the text and
# compile each one with `silc`. Collect all failures and report them at the end.
# This runs before highlighting so the fenced blocks are still intact.

if args.silc and "design/" not in str(args.src):
    failures = []
    for m in re.finditer(
        r"^```silicon\s*\n(.*?)^```\s*$", text, re.MULTILINE | re.DOTALL
    ):
        code = m.group(1)
        line = text[:m.start()].count("\n") + 1
        result = subprocess.run(
            [str(args.silc), "--format=silicon", "-o", "/dev/null", "-"],
            input=code,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            failures.append((line, code, result.stderr))

    if failures:
        for line, code, stderr in failures:
            print(f"error: {args.src}:{line}: silicon code block failed to compile",
                  file=sys.stderr)
            for codeline in code.rstrip("\n").split("\n"):
                print(f"  | {codeline}", file=sys.stderr)
            if stderr.strip():
                print(f"  silc: {stderr.strip()}", file=sys.stderr)
            print(file=sys.stderr)
        sys.exit(1)

# Replace ```silicon code blocks with syntax-highlighted HTML and write the
# output file.
text = highlight_blocks(text)
args.dst.parent.mkdir(parents=True, exist_ok=True)
args.dst.write_text(text)
