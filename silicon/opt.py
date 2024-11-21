from __future__ import annotations
from silicon.codegen import codegen
from silicon.diagnostics import DiagnosticException, Loc, emit_error
from silicon.dialect.hir import HIRDialect
from silicon.ir import SiliconDialect
from silicon.transforms.mem2reg import Mem2RegPass
from silicon.transforms.unroll import UnrollPass
from typing import Type, IO
from xdsl.dialects.builtin import ModuleOp
from xdsl.passes import ModulePass
from xdsl.transforms.canonicalize import CanonicalizePass
from xdsl.transforms.common_subexpression_elimination import CommonSubexpressionElimination
from xdsl.transforms.dead_code_elimination import DeadCodeElimination
from xdsl.xdsl_opt_main import xDSLOptMain
import xdsl
import xdsl.utils.exceptions


def main():
  SiliconOptMain().run()


class SiliconOptMain(xDSLOptMain):

  def __init__(self):
    super().__init__(description="Silicon modular optimizer driver")
    self.any_failed = False

  def register_all_dialects(self):
    self.ctx.load_dialect(xdsl.dialects.builtin.Builtin)
    self.ctx.load_dialect(SiliconDialect)
    self.ctx.load_dialect(HIRDialect)

  def register_all_passes(self):
    self.register_module_pass(CanonicalizePass)
    self.register_module_pass(DeadCodeElimination)
    self.register_module_pass(CommonSubexpressionElimination)
    self.register_module_pass(UnrollPass)
    self.register_module_pass(Mem2RegPass)

  def register_module_pass(self, p: Type[ModulePass]):
    self.register_pass(p.name, lambda: p)

  def register_all_targets(self):

    def _output_codegen(prog: ModuleOp, output: IO[str]):
      try:
        mlir_module = codegen(prog)
        print(mlir_module, file=output)
      except DiagnosticException:
        self.any_failed = True

    super().register_all_targets()
    self.available_targets = {
        "mlir": self.available_targets["mlir"],
        "codegen": _output_codegen,
    }

  def run(self):
    super().run()
    if self.any_failed:
      exit(1)

  def apply_passes(self, prog: ModuleOp) -> bool:
    try:
      try:
        return super().apply_passes(prog)
      except xdsl.utils.exceptions.DiagnosticException as e:
        emit_error(Loc.unknown(), str(e))
    except DiagnosticException:
      self.any_failed = True
      return False
