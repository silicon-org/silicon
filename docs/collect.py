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
parser.add_argument("-I", dest="search_paths", action="append", type=Path, default=[])
args = parser.parse_args()

# Resolve src against the search paths.
for path in args.search_paths:
    candidate = path / args.src
    if candidate.exists():
        src = candidate
        break
else:
    print(f"error: {args.src} not found in search paths:", file=sys.stderr)
    for path in args.search_paths:
        print(f"  {path}", file=sys.stderr)
    sys.exit(1)

text = src.read_text()

# Strip [TOC] markers. Hugo Book renders the table of contents automatically.
text = re.sub(r"^\[TOC\]\s*$", "", text, flags=re.MULTILINE)

# Inject YAML front matter if missing.
if not text.startswith("---"):
    title = args.dst.stem
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        title = match.group(1).strip()
    text = f"---\ntitle: \"{title}\"\n---\n\n{text}"

args.dst.parent.mkdir(parents=True, exist_ok=True)
args.dst.write_text(text)
