from __future__ import annotations
from typing import Type

from silicon.ir import SiliconDialect
from silicon.transforms.unroll import UnrollPass
from silicon.diagnostics import DiagnosticException

import xdsl
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.passes import ModulePass


def main():
    try:
        SiliconOptMain().run()
    except DiagnosticException:
        exit(1)


class SiliconOptMain(xDSLOptMain):

    def __init__(self):
        super().__init__(description="Silicon modular optimizer driver")

    def register_all_dialects(self):
        self.ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        self.ctx.load_dialect(SiliconDialect)

    def register_all_passes(self):
        self.register_module_pass(UnrollPass)

    def register_module_pass(self, p: Type[ModulePass]):
        self.register_pass(p.name, lambda: p)
