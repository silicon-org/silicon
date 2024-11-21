from __future__ import annotations
from dataclasses import dataclass, field
from silicon.dialect import hir
from silicon.source import Loc
from typing import *
from xdsl.builder import Builder
from xdsl.context import MLContext
from xdsl.dialects.builtin import ModuleOp
from xdsl.ir import Block
from xdsl.ir import Operation
from xdsl.ir import OpResult
from xdsl.ir import Region
from xdsl.ir import SSAValue
from xdsl.passes import ModulePass
from xdsl.printer import Printer
from xdsl.traits import MemoryEffectKind
from xdsl.traits import only_has_effect


class Mem2RegPass(ModulePass):
  """
  Forward values from stores to subsequent loads.
  """
  name = "mem2reg"

  def apply(self, ctx: MLContext, module: ModuleOp):
    worklist: List[Operation] = [module]
    while len(worklist) > 0:
      op = worklist.pop()
      for region in op.regions:
        for block in region.blocks:
          worklist += reversed(block.ops)
        handle(Context(region))


@dataclass
class Context:
  region: Region
  blocks: Dict[Block, BlockInfo] = field(default_factory=dict)


@dataclass
class BlockInfo:
  block: Block
  predecessors: List[BlockInfo] = field(default_factory=list)
  successors: List[BlockInfo] = field(default_factory=list)
  changed: bool = True
  stores: Dict[SSAValue, List[SSAValue]] = field(default_factory=dict)
  loads: List[SSAValue] = field(default_factory=list)


def handle(cx: Context):
  # Create a lattice of blocks.
  for block in cx.region.blocks:
    cx.blocks[block] = BlockInfo(block)
  for block, info in cx.blocks.items():
    info.predecessors += (cx.blocks[p] for p in block.predecessors())
    for p in info.predecessors:
      p.successors += [info]

  # Propagate the loaded and stored values across the lattice.
  any_changed = True
  while any_changed:
    any_changed = False
    for block_info in cx.blocks.values():
      if block_info.changed:
        block_info.changed = False
        update_block(block_info)
        if block_info.changed:
          any_changed = True

  # Make a list of references for which we can forward stores to subequent
  # loads. We conservatively only do this if there is no aliasing or any other
  # funny business.
  checked_refs: Dict[SSAValue, bool] = {}
  for block_info in cx.blocks.values():
    for ref in block_info.stores:
      if ref in checked_refs:
        continue
      checked_refs[ref] = all(
          isinstance(use.operation, (hir.LoadOp, hir.StoreOp))
          for use in ref.uses)
  candidates = set(ref for ref, amenable in checked_refs.items() if amenable)

  # Forward the stored values to corresponding loads.
  for block_info in cx.blocks.values():
    replace_loads(block_info, candidates)

  # Delete stores and allocations that are no longer read.
  for ref in candidates:
    if not all(isinstance(use.operation, hir.StoreOp) for use in ref.uses):
      continue
    for use in list(ref.uses):
      use.operation.detach()
      use.operation.erase()
    if isinstance(ref, OpResult) and only_has_effect(ref.owner,
                                                     MemoryEffectKind.ALLOC):
      ref.owner.detach()
      ref.owner.erase()


def update_block(block: BlockInfo):
  # Combine the incoming stores from the predecessor blocks.
  stores: Dict[SSAValue, List[SSAValue]] = dict()
  for predecessor in block.predecessors:
    for slot, values in predecessor.stores.items():
      vs = stores.setdefault(slot, list())
      for v in values:
        if v not in vs:
          vs.append(v)

  # Combine the incoming loads from the successor blocks.
  loads = list()
  for successor in block.successors:
    for v in successor.loads:
      if v not in loads:
        loads.append(v)

  # Add this block's stores.
  for op in block.block.ops:
    if isinstance(op, hir.StoreOp):
      stores[op.target] = [op.value]
    if isinstance(op, hir.LoadOp):
      if op.target not in loads:
        loads.append(op.target)

  # Add this block's loads.
  for op in reversed(block.block.ops):
    if isinstance(op, hir.StoreOp):
      if op.target in loads:
        loads.remove(op.target)
    if isinstance(op, hir.LoadOp):
      if op.target not in loads:
        loads.append(op.target)

  if block.stores != stores:
    block.stores = stores
    block.changed = True

  if block.loads != loads:
    block.loads = loads
    block.changed = True


def replace_loads(block: BlockInfo, candidates: Set[SSAValue]):
  stores = dict(block.stores)
  for op in block.block.ops:
    if isinstance(op, hir.StoreOp):
      stores[op.target] = [op.value]
    if isinstance(op, hir.LoadOp) and op.target in candidates:
      vs = stores.get(op.target, [])
      if len(vs) == 1:
        op.result.replace_by(vs[0])
        op.detach()
        op.erase()
