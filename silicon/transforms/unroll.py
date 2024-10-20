from __future__ import annotations
from typing import Dict, List, Set

from silicon import ir
from silicon.diagnostics import *

from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import (
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

    worklist: List[Operation]
    decls: List[Operation]
    reads: List[Operation]
    assignments: Dict[SSAValue, SSAValue]
    called_funcs: Set[ir.FuncOp]

    def apply(self, ctx: MLContext, module: ModuleOp):
        self.called_funcs = set()

        for op in module.ops:
            if not isinstance(op, ir.SiModuleOp):
                continue

            self.decls = []
            self.reads = []
            self.assignments = {}

            # Unroll the module body using a worklist.
            self.worklist = list(op.body.ops)
            while len(self.worklist) > 0:
                self.unroll(self.worklist.pop(0))

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

        # Remove functions.
        for op in module.ops:
            if not isinstance(op, ir.FuncOp):
                continue
            if op not in self.called_funcs:
                emit_warning(op.loc,
                             f"function `{op.sym_name.data}` is never used")
            op.detach()
            op.erase()

    def unroll(self, op: Operation):
        # Replace all derefs with the current assignment.
        if isinstance(op, ir.DerefOp):
            assignment = self.assignments.get(op.arg)
            if not assignment:
                name = "value"
                if isinstance(op.arg.owner, (ir.VarDeclOp, ir.OutputDeclOp)):
                    name = f"`{op.arg.owner.decl_name.data}`"
                emit_error(ir.get_loc(op),
                           f"{name} is unassigned at this point")
            op.result.replace_by(assignment)
            op.detach()
            op.erase()
            return

        # Inline calls.
        if isinstance(op, ir.CallOp):
            func = op.get_called_func()
            self.called_funcs.add(func)
            mapping: Dict[SSAValue, SSAValue] = dict()
            for call_arg, func_arg in zip(op.args, func.body.blocks[0].args):
                mapping[func_arg] = call_arg
            builder = Builder.before(op)
            inlined_ops = []
            for body_op in func.body.blocks[0].ops:
                if isinstance(body_op, ir.ReturnOp):
                    for call_result, return_arg in zip(op.res, body_op.args):
                        call_result.replace_by(mapping[return_arg])
                    continue
                inlined_ops.append(builder.insert(body_op.clone(mapping)))
            op.detach()
            op.erase()
            self.worklist = inlined_ops + self.worklist
            return

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

        # Add any nested ops to the worklist.
        for region in op.regions:
            for block in region.blocks:
                self.worklist = list(block.ops) + self.worklist
