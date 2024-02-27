from __future__ import annotations

from typing import *
from silicon import ast
from silicon.diagnostics import *
from dataclasses import dataclass, field

__all__ = ["resolve_names"]


@dataclass
class Scope:
    parent: Optional[Scope]
    names: Dict[str, ast.AstNode] = field(default_factory=dict)

    # Resolve a `name` to an AST node in this scope or one of its parents.
    # Reports an error diagnostic for the given `loc` if the name is not found.
    def resolve(self, name: str, loc: Loc) -> ast.AstNode:
        scope: Optional[Scope] = self
        while scope:
            if node := scope.names.get(name):
                return node
            scope = scope.parent
        emit_error(loc, f"unknown name `{name}`")

    # Declare a name in this scope.
    def declare(self, name: str, node: ast.AstNode):
        if existing_node := self.names.get(name):
            if cannot_be_shadowed(existing_node):
                emit_info(existing_node.loc,
                          f"previous definition of `{name}` was here")
                emit_error(node.loc, f"name `{name}` already defined")
        self.names[name] = node


# Returns true if the given node's name cannot be shadowed in a scope, and a
# "name already defined" error should be thrown.
def cannot_be_shadowed(node: ast.AstNode) -> bool:
    if isinstance(node, ast.InputStmt) or isinstance(node, ast.OutputStmt):
        return True
    if isinstance(node, ast.ModItem):
        return True
    return False


# Resolve names in an AST.
def resolve_names(root: ast.Root):
    resolve_node(root, Scope(parent=None))

    # Ensure that no unresolved names remain in the AST.
    for child in root.walk():
        for name, value in child.__dict__.items():
            if isinstance(value, ast.Binding) and value.target is None:
                emit_error(child.loc,
                           f"unresolved {name} in {child.__class__.__name__}")


# Resolve the names within the given node.
def resolve_node(node: ast.AstNode, scope: Scope):
    # Root items in the AST and declarations in a module do not have to be
    # declared before they can be used. We therefore declare all names upfront
    # before we resolve them.
    if isinstance(node, ast.Root) or isinstance(node, ast.ModItem):
        subscope = Scope(parent=scope)
        for child in node.children():
            declare_node(child, subscope)
        for child in node.children():
            resolve_node(child, subscope)
        return

    # Resolve children.
    for child in node.children():
        resolve_node(child, scope)

    # Handle specific nodes.
    if isinstance(node, ast.IdentExpr):
        node.binding.target = scope.resolve(node.name.spelling(), node.loc)


# Declare the names of a given node.
def declare_node(node: ast.AstNode, scope: Scope):
    if isinstance(node, ast.ModItem):
        scope.declare(node.name.spelling(), node)
        return
    if isinstance(node, ast.InputStmt) or isinstance(node, ast.OutputStmt):
        scope.declare(node.name.spelling(), node)
        return