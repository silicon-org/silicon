from __future__ import annotations
from typing import Dict, List

from silicon import ir
from silicon.diagnostics import *

from xdsl.builder import Builder
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import (
    MLContext,
    Operation,
    SSAValue,
)
from xdsl.passes import ModulePass


class UnrollPass(ModulePass):
    """
    Unroll all control flow and resolve declarations and assignments to their
    actual values at the respective point in the program.
    """
    name = "unroll"

    decls: List[Operation]
    reads: List[Operation]
    assignments: Dict[SSAValue, SSAValue]

    def apply(self, ctx: MLContext, module: ModuleOp):
        for op in module.ops:
            if not isinstance(op, ir.SiModuleOp):
                continue

            self.decls = []
            self.reads = []
            self.assignments = {}
            self.unroll(op)

            # Replace all `RegDeclOp`s with a corresponding `RegOp` that has its
            # input connected to the final assigned "next" value.
            for decl in self.decls:
                if not isinstance(decl, ir.RegDeclOp):
                    continue
                if not decl.result.uses:
                    continue
                builder = Builder.before(decl)
                value = self.assignments[decl.result]
                reg = builder.insert(
                    ir.RegOp(decl.clock, value, ir.get_loc(decl))).current
                self.assignments[decl.result] = reg

            # Replace all `OutputDeclOp`s with a corresponding `OutputOp` that
            # has the final value assigned to the port as its operand.
            for decl in self.decls:
                if not isinstance(decl, ir.OutputDeclOp):
                    continue
                assignment = self.assignments.get(decl.result)
                if not assignment:
                    emit_error(
                        ir.get_loc(decl),
                        f"output `{decl.decl_name.data}` has not been assigned"
                    )
                builder = Builder.before(decl)
                builder.insert(
                    ir.OutputPortOp(decl.decl_name, assignment,
                                    ir.get_loc(decl)))

            # Replace all `WireGetOp`s and `RegCurrentOp`s with the final
            # assigned value.
            for read in self.reads:
                value = self.assignments[read.operands[0]]
                while isinstance(value.owner, ir.WireGetOp) or isinstance(
                        value.owner, ir.RegCurrentOp):
                    value = self.assignments[value.owner.operands[0]]
                read.results[0].replace_by(value)
                read.detach()
                read.erase()

            # Delete all variable, wire, and register declarations.
            for decl in self.decls:
                decl.detach()
                decl.erase()

    def unroll(self, op: Operation):
        # Replace all variable uses with the current assignment.
        for idx in range(len(op.operands)):
            if idx == 0 and isinstance(op, ir.AssignOp):
                continue
            operand = op.operands[idx]
            if not isinstance(operand.owner, ir.VarDeclOp) and not isinstance(
                    operand.owner, ir.OutputDeclOp):
                continue
            assignment = self.assignments.get(operand)
            if not assignment:
                emit_error(
                    ir.get_loc(op),
                    f"`{operand.owner.decl_name.data}` is unassigned at this point"
                )
            op.operands[idx] = assignment

        # Keep track of the declarations for later post-processing.
        if isinstance(op, ir.VarDeclOp) or isinstance(
                op, ir.WireDeclOp) or isinstance(
                    op, ir.RegDeclOp) or isinstance(op, ir.OutputDeclOp):
            self.decls.append(op)

        # Handle assignments.
        if isinstance(op, ir.AssignOp):
            if not isinstance(op.dst.owner, ir.VarDeclOp) and not isinstance(
                    op.dst.owner, ir.OutputDeclOp):
                emit_error(
                    ir.get_loc(op),
                    f"expression `{ir.get_loc(op.dst).spelling()}` cannot appear on left-hand side of `=`"
                )
            self.assignments[op.dst] = op.src
            op.detach()
            op.erase()
            return
        if isinstance(op, ir.WireSetOp):
            self.assignments[op.wire] = op.value
            op.detach()
            op.erase()
            return
        if isinstance(op, ir.RegNextOp):
            self.assignments[op.reg] = op.value
            op.detach()
            op.erase()
            return
        if isinstance(op, ir.WireGetOp) or isinstance(op, ir.RegCurrentOp):
            self.reads.append(op)
            return

        # Process the body.
        for region in op.regions:
            for block in region.blocks:
                for inner_op in block.ops:
                    self.unroll(inner_op)
