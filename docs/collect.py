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

# Inject YAML front matter if missing.
if not text.startswith("---"):
    title = args.dst.stem
    match = re.search(r"^#\s+(.+)$", text, re.MULTILINE)
    if match:
        title = match.group(1).strip()
    text = f"---\ntitle: \"{title}\"\n---\n\n{text}"

args.dst.parent.mkdir(parents=True, exist_ok=True)
args.dst.write_text(text)
