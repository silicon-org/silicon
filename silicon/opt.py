from __future__ import annotations
from typing import Type

from silicon.ir import SiliconDialect
from silicon.transforms.unroll import UnrollPass
from silicon.diagnostics import DiagnosticException

import xdsl
from xdsl.dialects.builtin import ModuleOp
from xdsl.xdsl_opt_main import xDSLOptMain
from xdsl.passes import ModulePass


def main():
    SiliconOptMain().run()


class SiliconOptMain(xDSLOptMain):

    def __init__(self):
        super().__init__(description="Silicon modular optimizer driver")
        self.any_failed = False

    def register_all_dialects(self):
        self.ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        self.ctx.load_dialect(SiliconDialect)

    def register_all_passes(self):
        self.register_module_pass(UnrollPass)

    def register_module_pass(self, p: Type[ModulePass]):
        self.register_pass(p.name, lambda: p)

    def run(self):
        super().run()
        if self.any_failed:
            exit(1)

    def apply_passes(self, prog: ModuleOp) -> bool:
        try:
            return super().apply_passes(prog)
        except DiagnosticException:
            self.any_failed = True
            return False
