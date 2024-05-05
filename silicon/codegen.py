from __future__ import annotations

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from silicon import ast, ir
from silicon.diagnostics import *
from silicon.source import Loc
from silicon.ty import Type, UIntType, WireType, RegType

import circt
from circt.ir import Context, InsertionPoint, IntegerType, Location, Module as MLIRModuleOp
from circt.ir import Value as CirctValue
from circt.dialects import hw, comb, seq
from circt.support import BackedgeBuilder, connect

__all__ = ["codegen"]


def codegen(root: ir.ModuleOp) -> MLIRModuleOp:
    with Context() as ctx, Location.unknown():
        circt.register_dialects(ctx)
        module = MLIRModuleOp.create()
        with InsertionPoint(module.body):
            for op in root.ops:
                if isinstance(op, ir.SiModuleOp):
                    ModuleCodegen().codegen_module(op)
                else:
                    emit_error(ir.get_loc(op), "not supported for codegen")
        module.operation.verify()
        return module


# Convert a Silicon IR type to a CIRCT IR type.
def codegen_type(ty: ir.Attribute, loc: Loc) -> circt.ir.Type:
    if isinstance(ty, ir.IntegerType):
        return IntegerType.get_signless(ty.width.data)
    if isinstance(ty, ir.WireType) or isinstance(ty, ir.RegType):
        return codegen_type(ty.inner, loc)
    if isinstance(ty, ir.TupleType):
        fields = [(f"_{i}", codegen_type(field, loc))
                  for i, field in enumerate(ty.fields)]
        return hw.StructType.get(fields)

    emit_error(loc, f"type `{ty}` not supported for codegen")


# Code generation facilities for an `si.module`.
@dataclass
class ModuleCodegen:
    values: Dict[ir.SSAValue, CirctValue] = field(default_factory=dict)
    backedges: List[Tuple[CirctValue,
                          ir.SSAValue]] = field(default_factory=list)

    def codegen_module(self, mod: ir.SiModuleOp):
        # Collect the input and output ports.
        input_ports: List[ir.InputPortOp] = []
        output_ports: List[ir.OutputPortOp] = []
        used_names: Dict[str, Loc] = dict()

        def check_port_name(name: str, loc: Loc):
            if existing := used_names.get(name):
                emit_info(existing,
                          f"previous definition of port `{name}` was here")
                emit_error(loc, f"port `{name}` already defined")
            used_names[name] = loc

        for op in mod.body.ops:
            if isinstance(op, ir.InputPortOp):
                check_port_name(op.port_name.data, ir.get_loc(op))
                input_ports.append(op)
            if isinstance(op, ir.OutputPortOp):
                check_port_name(op.port_name.data, ir.get_loc(op))
                output_ports.append(op)

        # Create the module and entry block.
        hwmod = hw.HWModuleOp(
            name=mod.sym_name.data,
            input_ports=[(op.port_name.data,
                          codegen_type(op.value.type, ir.get_loc(op)))
                         for op in input_ports],
            output_ports=[(op.port_name.data,
                           codegen_type(op.value.type, ir.get_loc(op)))
                          for op in output_ports],
        )
        block = hwmod.add_entry_block()

        # Associate the CIRCT block arguments with the input ports. This will
        # allow us to resolve references to IR ops to the corresponding CIRCT
        # value.
        for port, arg in zip(input_ports, block.arguments):
            self.values[port.value] = arg

        # Convert the module body.
        with InsertionPoint(block):
            for op in mod.body.ops:
                self.codegen_op(op)
            hw.OutputOp([self.values[port.value] for port in output_ports])

        # Patch up all backedges.
        for backedge, silicon_ir_value in self.backedges:
            backedge.replace_all_uses_with(self.values[silicon_ir_value])
            backedge.owner.erase()

    def codegen_op(self, op: ir.Operation):
        if isinstance(op, ir.InputPortOp) or isinstance(op, ir.OutputPortOp):
            return

        # Lookup the operands or create dummy placeholder values for them.
        operands = []
        for operand in op.operands:
            if value := self.values.get(operand):
                operands.append(value)
            else:
                backedge = circt.ir.Operation.create(
                    "builtin.unrealized_conversion_cast",
                    [codegen_type(operand.type, ir.get_loc(operand))],
                ).result
                self.values[operand] = backedge
                self.backedges.append((backedge, operand))
                operands.append(backedge)

        # Handle operations that produce a single result.
        if len(op.results) == 1:
            if value := self.codegen_single_result_op(op, operands):
                self.values[op.results[0]] = value
                return

        emit_error(ir.get_loc(op), f"op `{op.name}` not supported for codegen")

    def codegen_single_result_op(
        self,
        op: ir.Operation,
        operands: List[CirctValue],
    ) -> CirctValue | None:
        if isinstance(op, ir.ConstantOp):
            return hw.ConstantOp.create(
                IntegerType.get_signless(op.value.type.width.data),
                op.value.value.data,
            ).result
        if isinstance(op, ir.ConstantUnitOp):
            return hw.ConstantOp.create(IntegerType.get_signless(1), 0).result
        if isinstance(op, ir.NegOp):
            zero = hw.ConstantOp.create(operands[0].type, 0)
            return comb.SubOp(zero, operands[0]).result
        if isinstance(op, ir.NotOp):
            ones = hw.ConstantOp.create(operands[0].type, -1)
            return comb.XorOp([ones, operands[0]]).result
        if isinstance(op, ir.AddOp):
            return comb.AddOp(operands).result
        if isinstance(op, ir.SubOp):
            return comb.SubOp(*operands).result
        if isinstance(op, ir.ConcatOp):
            return comb.ConcatOp(operands).result
        if isinstance(op, ir.ExtractOp):
            assert isinstance(op.result.type, ir.IntegerType)
            return comb.ExtractOp(
                IntegerType.get_signless(op.result.type.width.data),
                operands[0],
                op.offset.data,
            ).result
        if isinstance(op, ir.MuxOp):
            return comb.MuxOp(*operands).result
        if isinstance(op, ir.RegOp):
            next = operands[1]
            clock = seq.ToClockOp(operands[0]).result
            return seq.CompRegOp(next.type, next, clock).result
        if isinstance(op, ir.TupleCreateOp):
            ty = codegen_type(op.res.type, ir.get_loc(op))
            return hw.StructCreateOp(ty, operands).result
        if isinstance(op, ir.TupleGetOp):
            ty = codegen_type(op.res.type, ir.get_loc(op))
            return hw.StructExtractOp(ty, operands[0], op.field.data).result
        return None
