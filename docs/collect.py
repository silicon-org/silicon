#!/usr/bin/env python3
import argparse
import shutil
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

args.dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy2(src, args.dst)
