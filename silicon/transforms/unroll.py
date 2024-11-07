from __future__ import annotations

import sys
import io
from typing import Dict, List, Set
from dataclasses import dataclass, field

from silicon import ir
from silicon.diagnostics import *
from silicon.source import Loc
from silicon.consteval import const_eval_ir

from xdsl.builder import Builder
from xdsl.printer import Printer
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
  reads: List[Operation]
  erase_later: List[Operation]
  assignments: Dict[SSAValue, Value]
  called_funcs: Set[ir.FuncOp]

  def apply(self, ctx: MLContext, module: ModuleOp):
    self.called_funcs = set()

    # Handle modules.
    for op in module.ops:
      if isinstance(op, ir.SiModuleOp):
        unroll_module(self, op)

    # Remove functions.
    for op in module.ops:
      if not isinstance(op, ir.FuncOp):
        continue
      if op not in self.called_funcs:
        emit_warning(op.loc, f"function `{op.sym_name.data}` is never used")
      op.detach()
      op.erase()


def unroll_module(cx: UnrollPass, op: ir.SiModuleOp):
  cx.reads = []
  cx.assignments = {}
  cx.erase_later = []

  # Unroll the module body using a worklist.
  cx.worklist = list(op.body.ops)
  while len(cx.worklist) > 0:
    unroll_op(cx, cx.worklist.pop(0))

  # Replace all `RegDeclOp`s with a corresponding `RegOp` that has its
  # input connected to the final assigned "next" value.
  for decl in op.body.ops:
    if not isinstance(decl, ir.RegDeclOp):
      continue
    if not decl.result.uses:
      continue
    builder = Builder.before(decl)
    value = cx.assignments[decl.result]
    assert isinstance(value, SSAValue)
    reg = builder.insert(ir.RegOp(decl.clock, value, ir.get_loc(decl))).current
    cx.assignments[decl.result] = reg

  # Replace all `OutputDeclOp`s with a corresponding `OutputOp` that
  # has the final value assigned to the port as its operand.
  for decl in op.body.ops:
    if not isinstance(decl, ir.OutputDeclOp):
      continue
    ref = cx.assignments[decl.result]
    assert isinstance(ref, RefValue)
    assert isinstance(ref.target, ScalarSlot)
    loc = ir.get_loc(decl)
    if ref.target.value is None:
      emit_error(loc, f"output `{decl.decl_name.data}` is unassigned")
    assert isinstance(ref.target.value, SSAValue)
    builder = Builder.before(decl)
    builder.insert(ir.OutputPortOp(decl.decl_name, ref.target.value, loc))

  # Replace all `WireGetOp`s and `RegCurrentOp`s with the final
  # assigned value.
  for read in cx.reads:
    value = cx.assignments[read.operands[0]]
    assert isinstance(value, SSAValue)
    while isinstance(value.owner, ir.WireGetOp) or isinstance(
        value.owner, ir.RegCurrentOp):
      value = cx.assignments[value.owner.operands[0]]
      assert isinstance(value, SSAValue)
    read.results[0].replace_by(value)
    read.detach()
    read.erase()

  # Delete all variable, wire, and register declarations.
  for candidate in reversed(cx.erase_later):
    any_uses = False
    for result in candidate.results:
      if len(result.uses) > 0:
        any_uses = True
        break
    if not any_uses:
      candidate.detach()
      candidate.erase()


def unroll_op(cx: UnrollPass, op: Operation):
  # Inline calls.
  if isinstance(op, ir.CallOp):
    func = op.get_called_func()
    cx.called_funcs.add(func)
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
      cloned_op = body_op.clone(mapping)
      if hasattr(body_op, "loc"):
        cloned_op.loc = body_op.loc
      inlined_ops.append(builder.insert(cloned_op))
    op.detach()
    op.erase()
    cx.worklist = inlined_ops + cx.worklist
    return

  # Inline ifs.
  if isinstance(op, ir.IfOp):
    cond_attr = const_eval_ir(op.cond)
    assert cond_attr.type == ir.IntegerType(1)
    cond = bool(cond_attr.value.data)
    body_to_inline = op.then_body if cond else op.else_body
    builder = Builder.before(op)
    inlined_ops = []
    mapping = dict()
    for body_op in body_to_inline.blocks[0].ops:
      cloned_op = body_op.clone(mapping)
      if hasattr(body_op, "loc"):
        cloned_op.loc = body_op.loc
      inlined_ops.append(builder.insert(cloned_op))
    op.detach()
    op.erase()
    cx.worklist = inlined_ops + cx.worklist
    return

  # Create allocation slots for declaration ops. These can then be assigned and
  # dereferenced.
  if isinstance(op, (ir.VarDeclOp, ir.OutputDeclOp)):
    slot = alloc_slot(op.get_declared_type())
    slot.name = op.decl_name.data
    ref = RefValue(slot)
    cx.assignments[op.result] = ref
    cx.erase_later.append(op)
    return

  # Assignments directly change the value in the `Slot` on their left-hand side.
  if isinstance(op, ir.AssignOp):
    dst = cx.assignments[op.dst]
    if not isinstance(dst, RefValue):
      emit_error(ir.get_loc(op), "cannot assign to left-hand")
    src = cx.assignments.get(op.src, op.src)
    assign_slot(dst.target, src)
    op.detach()
    op.erase()
    return

  # Handle dereferencing and creating a temporary reference.
  if isinstance(op, ir.DerefOp):
    return unroll_deref(cx, op)
  if isinstance(op, ir.RefOp):
    return unroll_ref(cx, op)

  if isinstance(op, ir.TupleCreateOp):
    agg = AggregateValue([cx.assignments.get(arg, arg) for arg in op.args])
    cx.assignments[op.res] = agg
    cx.erase_later.append(op)
    return
  if isinstance(op, ir.TupleGetRefOp):
    return unroll_tuple_get_ref(cx, op)
  if isinstance(op, ir.TupleGetOp):
    return unroll_tuple_get(cx, op)

  # Handle wire and register declarations.
  # TODO: Get rid of these in favor of something more robust.
  if isinstance(op, (ir.WireDeclOp, ir.RegDeclOp)):
    cx.erase_later.append(op)
    return
  if isinstance(op, ir.WireSetOp):
    cx.assignments[op.wire] = op.value
    op.detach()
    op.erase()
    return
  if isinstance(op, ir.RegNextOp):
    cx.assignments[op.reg] = op.value
    op.detach()
    op.erase()
    return
  if isinstance(op, (ir.WireGetOp, ir.RegCurrentOp)):
    cx.reads.append(op)
    return

  # Add any nested ops to the worklist.
  for region in op.regions:
    for block in region.blocks:
      cx.worklist = list(block.ops) + cx.worklist


def unroll_tuple_get(cx: UnrollPass, op: ir.TupleGetOp):
  aggregate = cx.assignments[op.arg]
  assert isinstance(aggregate, AggregateValue)
  value = aggregate.fields[op.field.data]
  cx.assignments[op.res] = value
  if isinstance(value, SSAValue):
    op.res.replace_by(value)
  cx.erase_later.append(op)


def unroll_tuple_get_ref(cx: UnrollPass, op: ir.TupleGetRefOp):
  ref = cx.assignments[op.arg]
  assert isinstance(ref, RefValue)
  assert isinstance(ref.target, AggregateSlot)
  subref = RefValue(ref.target.fields[op.field.data])
  cx.assignments[op.res] = subref
  cx.erase_later.append(op)


def unroll_deref(cx: UnrollPass, op: ir.DerefOp):
  ref = cx.assignments[op.arg]
  assert isinstance(ref, RefValue)
  value = deref_slot(ref.target, ir.get_loc(op))
  cx.assignments[op.result] = value
  if not value_contains_refs(value):
    assert isinstance(op.result.type, ir.TypeAttribute)
    op.result.replace_by(
        materialize_value(cx, value, op.result.type, Builder.before(op),
                          ir.get_loc(op)))
  cx.erase_later.append(op)


def unroll_ref(cx: UnrollPass, op: ir.RefOp):
  assert isinstance(op.arg.type, ir.TypeAttribute)
  slot = alloc_slot(op.arg.type)
  assign_slot(slot, cx.assignments.get(op.arg, op.arg))
  ref = RefValue(slot)
  cx.assignments[op.result] = ref
  cx.erase_later.append(op)


#===------------------------------------------------------------------------===#
# Utilities
#===------------------------------------------------------------------------===#


def format_value(value: Value | None) -> str:
  if isinstance(value, SSAValue):
    return ir.format(value)
  return f"{value}"


#===------------------------------------------------------------------------===#
# Values
#===------------------------------------------------------------------------===#


class AggregateValue:
  fields: List[Value]

  def __init__(self, fields: List[Value]):
    self.fields = fields

  def __repr__(self) -> str:
    fs = [format_value(v) for v in self.fields]
    return f"Aggregate({fs})"


class RefValue:
  target: Slot

  def __init__(self, target: Slot):
    self.target = target

  def __repr__(self) -> str:
    return f"Ref({self.target})"


Value = RefValue | AggregateValue | SSAValue


def value_contains_refs(value: Value) -> bool:
  if isinstance(value, AggregateValue):
    for field in value.fields:
      if value_contains_refs(field):
        return True
    return False
  return isinstance(value, RefValue)


def materialize_value(cx: UnrollPass, value: Value, type: ir.TypeAttribute,
                      builder: Builder, loc: Loc) -> SSAValue:
  if isinstance(value, SSAValue):
    return value

  if isinstance(type, ir.TupleType) and isinstance(value, AggregateValue):
    fields = []
    for field_value, field_type in zip(value.fields, type.fields):
      fields.append(
          materialize_value(cx, field_value, field_type, builder, loc))
    op = builder.insert(ir.TupleCreateOp(fields, loc))
    cx.erase_later.append(op)
    return op.res

  assert False, f"cannot materialize value of type {type}"


#===------------------------------------------------------------------------===#
# Allocation Slots
#===------------------------------------------------------------------------===#


class Slot:
  parent: Slot | None
  name: str | None

  def __init__(self):
    self.parent = None
    self.name = None


class ScalarSlot(Slot):
  value: Value | None

  def __init__(self, value: Value | None):
    super().__init__()
    self.value = value

  def __repr__(self) -> str:
    return f"Scalar({format_value(self.value)})"


class AggregateSlot(Slot):
  fields: List[Slot]

  def __init__(self, fields: List[Slot]):
    super().__init__()
    self.fields = fields
    for field in self.fields:
      field.parent = self

  def __repr__(self) -> str:
    return f"Aggregate({self.fields})"


def get_slot_name(slot: Slot) -> str:
  name = slot.name or "?"
  if slot.parent is not None:
    return get_slot_name(slot.parent) + "." + name
  return name


def alloc_slot(type: ir.TypeAttribute) -> Slot:
  if isinstance(type, ir.TupleType):
    a = AggregateSlot([alloc_slot(fty) for fty in type.fields])
    for i, f in enumerate(a.fields):
      f.name = str(i)
    return a
  return ScalarSlot(None)


def deref_slot(slot: Slot, loc: Loc) -> Value:
  if isinstance(slot, ScalarSlot):
    if slot.value is None:
      name = get_slot_name(slot)
      emit_error(loc, f"`{name}` is unassigned at this point")
    return slot.value
  elif isinstance(slot, AggregateSlot):
    return AggregateValue([deref_slot(f, loc) for f in slot.fields])
  else:
    assert False


def assign_slot(dst: Slot, src: Value):
  if isinstance(dst, AggregateSlot) and isinstance(src, AggregateValue):
    assert len(dst.fields) == len(src.fields)
    for dst_field, src_field in zip(dst.fields, src.fields):
      assign_slot(dst_field, src_field)
  elif isinstance(dst, ScalarSlot) and isinstance(src, (SSAValue, RefValue)):
    dst.value = src
  else:
    assert False, f"invalid assignment {dst} = {src}"
