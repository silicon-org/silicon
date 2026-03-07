#!/usr/bin/env python3

# # Silicon Syntax Highlighter
#
# Hugo uses Chroma for syntax highlighting, but Chroma has no extension
# mechanism for custom lexers — you can't add a new language without forking
# Chroma and rebuilding Hugo. This was requested upstream (gohugoio/hugo#10421)
# but never implemented. So we do our own highlighting as a pre-processing step
# in collect.py, before Hugo ever sees the markdown.
#
# The tokenizer is a simple regex-based scanner, similar to how Chroma's XML
# lexer definitions work. Tokens are matched greedily in priority order. The
# output matches what Hugo's built-in Chroma highlighter produces for known
# languages (inline styles, same DOM structure), so Silicon code blocks blend
# in with the rest of the docs. Colors are from the catppuccin-frappe theme.
#
# Keep this file in sync with `include/silicon/Syntax/Tokens.def` whenever
# keywords or operators change. The colors below must also match the Chroma
# style configured in hugo.toml (`markup.highlight.style`); if the theme
# changes, regenerate the color map with `hugo gen chromastyles --style=<name>`
# and update `_COLORS` accordingly.

import re
from html import escape

# Catppuccin Frappe colors, matching `hugo gen chromastyles --style=catppuccin-frappe`.
_COLORS = {
    "base_fg": "#c6d0f5",
    "base_bg": "#303446",
    "keyword": "#ca9ee6",         # .k
    "keyword_type": "#e78284",    # .kt
    "keyword_const": "#ef9f76",   # .kc
    "comment": "#737994",         # .c1 (italic)
    "number": "#ef9f76",          # .mi/.mh/.mb
    "operator": "#99d1db",        # .o (bold)
    "function": "#8caaee",        # .nf
}

# Token definitions in priority order. Each entry is (name, pattern). The first
# match wins, so more specific patterns (like keywords) must come before the
# generic identifier pattern.
_TOKENS = [
    ("whitespace", r"\s+"),
    ("comment",    r"//[^\n]*"),
    ("keyword",    r"\b(?:const|dyn|else|fn|for|if|let|loop|match|pub|return|while)\b"),
    ("type",       r"\b(?:int|uint)\b"),
    ("bool",       r"\b(?:true|false)\b"),
    ("hex",        r"0x[0-9a-fA-F_]+"),
    ("bin",        r"0b[01_]+"),
    ("number",     r"[0-9][0-9_]*"),
    ("arrow",      r"->"),
    ("op2",        r"==|!=|<=|>=|<<|>>"),
    ("op1",        r"[+\-*/%&|^!<>=]"),
    ("punct",      r"[{}()\[\].,;:?]"),
    ("ident",      r"[a-zA-Z_]\w*"),
]

_TOKEN_RE = re.compile("|".join(f"(?P<{name}>{pat})" for name, pat in _TOKENS))


def _span(style: str, text: str) -> str:
    return f'<span style="{style}">{escape(text)}</span>'


def _colorize_token(name: str, text: str, next_char: str) -> str:
    """Wrap a token in a styled span. Returns raw escaped text for tokens that
    use the default text color (whitespace, punctuation, plain identifiers)."""
    if name == "whitespace":
        return escape(text)
    if name == "comment":
        return _span(f"color:{_COLORS['comment']};font-style:italic", text)
    if name == "keyword":
        return _span(f"color:{_COLORS['keyword']}", text)
    if name == "type":
        return _span(f"color:{_COLORS['keyword_type']}", text)
    if name == "bool":
        return _span(f"color:{_COLORS['keyword_const']}", text)
    if name in ("hex", "bin", "number"):
        return _span(f"color:{_COLORS['number']}", text)
    if name in ("arrow", "op2", "op1"):
        return _span(f"color:{_COLORS['operator']};font-weight:bold", text)
    if name == "ident" and next_char == "(":
        return _span(f"color:{_COLORS['function']}", text)
    # punct and plain ident: default text color, no span needed.
    return escape(text)


def highlight(code: str) -> str:
    """Highlight Silicon source code and return a complete HTML block matching
    Chroma's inline-style output format."""
    fg = _COLORS["base_fg"]
    bg = _COLORS["base_bg"]

    # Tokenize and colorize.
    parts = []
    pos = 0
    for m in _TOKEN_RE.finditer(code):
        # Any unmatched characters between tokens get escaped as-is.
        if m.start() > pos:
            parts.append(escape(code[pos:m.start()]))
        name = m.lastgroup
        text = m.group()
        # Peek at the next non-whitespace character to detect function calls.
        rest = code[m.end():]
        next_char = rest.lstrip()[0] if rest.lstrip() else ""
        parts.append(_colorize_token(name, text, next_char))
        pos = m.end()
    if pos < len(code):
        parts.append(escape(code[pos:]))

    # Wrap each line in Chroma's flex-line structure. Strip any trailing
    # newline from the code to avoid an empty trailing line.
    raw = "".join(parts).rstrip("\n")
    lines = raw.split("\n")
    wrapped = "\n".join(
        f'<span style="display:flex;"><span>{line}\n</span></span>'
        for line in lines
    )

    return (
        f'<div class="highlight">'
        f'<pre tabindex="0" style="color:{fg};background-color:{bg};'
        f'-moz-tab-size:4;-o-tab-size:4;tab-size:4;">'
        f'<code class="language-silicon" data-lang="silicon">'
        f"{wrapped}"
        f"</code></pre></div>"
    )


def highlight_blocks(text: str) -> str:
    """Find all ```silicon fenced code blocks in Markdown text and replace them
    with highlighted HTML."""
    def _replace(m):
        return highlight(m.group(1))
    return re.sub(
        r"^```silicon\s*\n(.*?)^```\s*$",
        _replace,
        text,
        flags=re.MULTILINE | re.DOTALL,
    )
