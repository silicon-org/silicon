#!/usr/bin/env python3

# # Collect Documentation
#
# Copies a documentation file from a source to a destination, applying
# transformations needed for Hugo along the way:
# - Strips `[TOC]` markers (Hugo Book renders TOC automatically).
# - Injects YAML front matter if missing. The title is extracted from the first
#   `# Heading` line, falling back to the filename stem.

import argparse
import re
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("src", type=Path)
parser.add_argument("dst", type=Path)
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

args.dst.parent.mkdir(parents=True, exist_ok=True)
args.dst.write_text(text)
