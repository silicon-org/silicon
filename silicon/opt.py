from __future__ import annotations
from silicon.ir import SiliconDialect

import xdsl
from xdsl.xdsl_opt_main import xDSLOptMain


def main():
    SiliconOptMain().run()


class SiliconOptMain(xDSLOptMain):

    def __init__(self):
        super().__init__(description="Silicon modular optimizer driver")

    def register_all_dialects(self):
        self.ctx.load_dialect(xdsl.dialects.builtin.Builtin)
        self.ctx.load_dialect(SiliconDialect)
