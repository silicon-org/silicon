---
title: Operators and Symbols
weight: 2
---

# Appendix: Operators and Symbols

This appendix contains a glossary of Silicon's operators, symbols, and other syntax elements.

## Operators

The following table contains the operators in Silicon, an example of how the operator would appear in context, and a short explanation.

| Operator | Example | Explanation |
|----------|---------|-------------|
| `!` | `!expr` | Logical or bitwise complement (unary) |
| `&` | `&expr` | Reference (unary) |
| `&` | `expr & expr` | Bitwise AND |
| `*` | `*expr` | Dereference (unary) |
| `*` | `expr * expr` | Multiplication |
| `/` | `expr / expr` | Division |
| `%` | `expr % expr` | Remainder |
| `+` | `expr + expr` | Addition |
| `-` | `-expr` | Negation (unary) |
| `-` | `expr - expr` | Subtraction |
| `<<` | `expr << expr` | Left shift |
| `>>` | `expr >> expr` | Right shift |
| `<` | `expr < expr` | Less than |
| `>` | `expr > expr` | Greater than |
| `<=` | `expr <= expr` | Less than or equal |
| `>=` | `expr >= expr` | Greater than or equal |
| `==` | `expr == expr` | Equality |
| `!=` | `expr != expr` | Inequality |
| `^` | `expr ^ expr` | Bitwise XOR |
| `\|` | `expr \| expr` | Bitwise OR |

### Operator Precedence

Binary operators are listed here from highest to lowest precedence.
All binary operators are left-associative.

| Precedence | Operators |
|------------|-----------|
| Highest | `*` `/` `%` |
| | `+` `-` |
| | `<<` `>>` |
| | `<` `>` `<=` `>=` |
| | `==` `!=` |
| | `&` |
| | `^` |
| Lowest | `\|` |

## Non-Operator Symbols

### Punctuation

The following table contains symbols that appear on their own and are not used as operators.

| Symbol | Explanation |
|--------|-------------|
| `->` | Function return type |
| `.` | Field access |
| `,` | Separator in argument lists and parameters |
| `:` | Type annotation |
| `;` | Statement terminator |
| `=` | Assignment in `let` bindings |

### Delimiters

The following table contains symbols that are used as matching pairs of delimiters.

| Symbol | Explanation |
|--------|-------------|
| `(` `)` | Function call arguments, function parameter list |
| `{` `}` | Block expressions, function bodies |
| `[` `]` | Index expression (`a[i]`), slice expression (`a[i, len]`) |
| `<` `>` | Type parameters, as in `uint<32>` |

### Comments

The following table shows the comment syntax in Silicon.

| Symbol | Explanation |
|--------|-------------|
| `// ...` | Line comment |
| `/* ... */` | Block comment, can be nested |

### Number Literal Prefixes

The following table shows the prefixes for number literals in different bases.
Underscores (`_`) can be used as visual separators in any number literal.

| Prefix | Example | Base |
|--------|---------|------|
| *(none)* | `42` | Decimal |
| `0b` | `0b1010` | Binary |
| `0o` | `0o755` | Octal |
| `0x` | `0xff` | Hexadecimal |
