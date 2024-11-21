from __future__ import annotations
from typing import *
from xdsl.ir import Block
from xdsl.ir import Operation
from xdsl.ir import Region
from xdsl.ir import SSAValue
import xdsl


# Try to move operations around such that a value becomes defined before another
# operation.
def try_move_before(move: SSAValue, before: SSAValue) -> bool:
  if isinstance(move.owner, Block):
    return False
  if isinstance(before.owner, Block):
    return False

  move_op: Operation = move.owner
  before_op: Operation = before.owner

  # The ops must be in the same block.
  move_block = move_op.parent_block()
  before_block = before_op.parent_block()
  assert move_block
  assert before_block
  if move_block != before_block:
    return False
  block = move_block

  # If the `move` operation already dominates the `before` operation, we don't
  # need to do anything.
  if block.get_operation_index(move_op) < block.get_operation_index(before_op):
    return True

  # Move all of our operands before the `before` operation.
  for operand in move_op.operands:
    if not try_move_before(operand, before):
      return False

  # We can only move the `move` operation if it has no side-effects, or if there
  # are no side-effecting ops that we would move across.
  if not xdsl.traits.is_side_effect_free(move_op):
    op = before_op
    while op is not move_op:
      if not xdsl.traits.is_side_effect_free(op):
        print(f"cannot move {move_op} across {op}")
        return False
      assert op.next_op
      op = op.next_op

  # At this point we know that the move is legal.
  move_op.detach()
  block.insert_op_before(move_op, before_op)
  return True

  # if isinstance(value, Operation):
  #   value = value.owner

  # value_block: Block
  # if isinstance(value, Operation):
  #   value_block = value.parent_block()
  #   assert value_block is not None

  print(f"cannot move {move_op} before {before_op}")
  return False
