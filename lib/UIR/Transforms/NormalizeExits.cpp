//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # Normalize Early Function Exits into Single-Exit Value Form
//
// This pass rewrites every `uir.return` that is not a function body's
// terminator into choice/pick value form, bottom-up per region, leaving one
// `uir.return` as the body terminator. It is phase-blind — no phase
// information is consulted — and runs before phase analysis, so analysis and
// sorting never see an early exit.
//
// The rewrite applies two rules uniformly at every nesting level:
//
// - **Encode**: a structured CF op (`uir.if`, `uir.expr`, `uir.loop`) whose
//   regions hold a function exit is rebuilt with one extra trailing result,
//   a choice tagging how the region ended. Exit paths build tag 0 with the
//   returned values and their type operands as payload; every other path
//   builds the payload-free tag 1. On ifs and exprs the choice rides the
//   yield; on loops it rides `uir.break` in the loop's trailing result slot
//   (existing breaks gain the tag-1 slot, and exit paths become breaks).
//   Result slots an exit path does not provide are `uir.unreachable_value`,
//   never omitted, matching the exact-arity yield/break invariant.
//
// - **Dispatch**: right after the rebuilt op, an `hir.pick` on its choice
//   result branches between the exit and the continuation: the tag-0
//   alternative re-encodes the exit for the enclosing block, and the tag-1
//   alternative receives the rest of the block, where the same rules recurse.
//   Multiple exits in one block right-nest into a linear chain of binary
//   picks, so no continuation is ever duplicated.
//
// How an exit is encoded depends only on the block's parent op — its
// `ExitPlan`. The function body threads the returned values and their type
// operands through valued picks into the single trailing `uir.return`.
// If/expr regions yield their result slots plus the choice. Loop bodies
// break out of the loop directly, with a resultless dispatch pick and
// `uir.unreachable` after it, since break/continue are ordinary loop-real
// control flow. Dead code after exhaustive exits lands in tag-1 alternatives
// terminated by `uir.unreachable` and is never picked.
//
// The choices and picks this pass creates are executable single-phase
// control flow, not phase-crossing vocabulary: the pick's choice operand is
// decided in the same phase that runs the alternative.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/IR/Builders.h"

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_NORMALIZEEXITSPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

using namespace mlir;
using namespace silicon;
using namespace silicon::uir;

//===----------------------------------------------------------------------===//
// Exit Normalizer
//===----------------------------------------------------------------------===//

namespace {
/// How a block hands a function exit to its surroundings, determined solely
/// by the block's parent op.
struct ExitPlan {
  enum Kind {
    /// The function body block: exits thread through valued dispatch picks
    /// as (values..., type operands...) and feed the single trailing
    /// `uir.return`.
    FuncBody,
    /// A region block of an exit-carrying `uir.if` or `uir.expr`: exits ride
    /// the block's yield as a trailing choice slot.
    Arm,
    /// The body block of an exit-carrying `uir.loop`: exits leave the loop
    /// as breaks carrying the choice in the loop's trailing result slot.
    LoopBody,
  } kind;
  /// Type operands of the enclosing op's original result slots (empty for
  /// the function body).
  SmallVector<Value> slotTypes;
};

/// Rewrites all early `uir.return` ops out of one `uir.func`.
class ExitNormalizer {
public:
  ExitNormalizer(FuncOp funcOp) : funcOp(funcOp) {}
  LogicalResult run();

private:
  LogicalResult normalizeBlock(Block &block, const ExitPlan &plan,
                               bool threading);
  LogicalResult finishEnd(Block &block, const ExitPlan &plan, bool threading);
  FailureOr<Value> normalizeExitOp(Operation *op);
  FailureOr<Value> normalizeIf(IfOp ifOp);
  FailureOr<Value> normalizeExpr(ExprOp exprOp);
  FailureOr<Value> normalizeLoop(LoopOp loopOp);
  void buildExitEndpoint(OpBuilder &builder, Location loc, const ExitPlan &plan,
                         ValueRange values, ValueRange typeOfValues,
                         bool terminatorFlavor);

  FuncOp funcOp;
  /// The one `hir.choice_type` shared by all encodings in this function.
  Value choiceType;
  /// The function's declared result arity; every exit payload carries this
  /// many values plus as many type operands.
  unsigned numResults = 0;
  Type anyTy;
};
} // namespace

/// Whether `op` transitively contains a function exit.
static bool containsReturn(Operation *op) {
  return op->walk([](ReturnOp) { return WalkResult::interrupt(); })
      .wasInterrupted();
}

/// The nearest enclosing `uir.loop` of `op`, or null.
static LoopOp enclosingLoop(Operation *op) {
  for (auto *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto loopOp = dyn_cast<LoopOp>(parent))
      return loopOp;
  return {};
}

LogicalResult ExitNormalizer::run() {
  // The FuncOp verifier keeps signature regions free of `uir.return`, so
  // only the body region can hold exits. Nothing to do unless some return
  // is not the body terminator.
  auto &body = funcOp.getBody().front();
  auto *bodyTerminator = body.getTerminator();
  auto walkResult = funcOp.getBody().walk([&](ReturnOp op) {
    return op == bodyTerminator ? WalkResult::advance()
                                : WalkResult::interrupt();
  });
  if (!walkResult.wasInterrupted())
    return success();

  numResults = funcOp.getResultPhases().size();
  anyTy = hir::AnyType::get(funcOp.getContext());
  OpBuilder builder(&body, body.begin());
  choiceType = hir::ChoiceTypeOp::create(builder, funcOp.getLoc()).getResult();
  return normalizeBlock(body, ExitPlan{ExitPlan::FuncBody, {}},
                        /*threading=*/false);
}

/// Build the encoding of a function exit that returns `values` with
/// `typeOfValues`, per the enclosing block's plan. With `terminatorFlavor`
/// the encoding ends in the enclosing region's real terminator (`uir.yield`
/// with type operands); without, it ends in the `hir.yield` that threads
/// through a dispatch pick. Loop bodies always break — their dispatch never
/// threads.
void ExitNormalizer::buildExitEndpoint(OpBuilder &builder, Location loc,
                                       const ExitPlan &plan, ValueRange values,
                                       ValueRange typeOfValues,
                                       bool terminatorFlavor) {
  switch (plan.kind) {
  case ExitPlan::FuncBody: {
    assert(!terminatorFlavor && "the function body's exits always thread");
    SmallVector<Value> operands(values);
    llvm::append_range(operands, typeOfValues);
    hir::YieldOp::create(builder, loc, operands);
    return;
  }
  case ExitPlan::Arm: {
    SmallVector<Value> payload(values);
    llvm::append_range(payload, typeOfValues);
    auto choice = hir::ChoiceOp::create(builder, loc, anyTy, 0, payload);
    SmallVector<Value> slotValues, slotTypes;
    for (Value slotType : plan.slotTypes) {
      slotValues.push_back(
          UnreachableValueOp::create(builder, loc, anyTy).getResult());
      slotTypes.push_back(slotType);
    }
    slotValues.push_back(choice.getResult());
    slotTypes.push_back(choiceType);
    if (terminatorFlavor)
      YieldOp::create(builder, loc, slotValues, slotTypes);
    else
      hir::YieldOp::create(builder, loc, slotValues);
    return;
  }
  case ExitPlan::LoopBody: {
    SmallVector<Value> payload(values);
    llvm::append_range(payload, typeOfValues);
    auto choice = hir::ChoiceOp::create(builder, loc, anyTy, 0, payload);
    SmallVector<Value> slotValues, slotTypes;
    for (Value slotType : plan.slotTypes) {
      slotValues.push_back(
          UnreachableValueOp::create(builder, loc, anyTy).getResult());
      slotTypes.push_back(slotType);
    }
    slotValues.push_back(choice.getResult());
    slotTypes.push_back(choiceType);
    BreakOp::create(builder, loc, slotValues, slotTypes);
    return;
  }
  }
  llvm_unreachable("covered switch");
}

/// Normalize one block: recurse into the first op holding a function exit,
/// rebuild it into choice-yielding form, and dispatch on the choice right
/// after it, moving the rest of the block into the continuation alternative.
/// `threading` marks blocks that are dispatch-pick alternatives created by
/// this pass, whose endpoints thread via `hir.yield` instead of rebuilding
/// the enclosing region's terminator.
LogicalResult ExitNormalizer::normalizeBlock(Block &block, const ExitPlan &plan,
                                             bool threading) {
  Operation *exitOp = nullptr;
  for (auto &op : block.without_terminator())
    if (containsReturn(&op)) {
      exitOp = &op;
      break;
    }
  if (!exitOp)
    return finishEnd(block, plan, threading);

  auto loc = exitOp->getLoc();
  auto choice = normalizeExitOp(exitOp);
  if (failed(choice))
    return failure();

  // Dispatch on the choice right after the rebuilt op.
  OpBuilder builder(&block,
                    std::next(Block::iterator(choice->getDefiningOp())));
  unsigned numPickResults;
  switch (plan.kind) {
  case ExitPlan::FuncBody:
    numPickResults = 2 * numResults;
    break;
  case ExitPlan::Arm:
    numPickResults = plan.slotTypes.size() + 1;
    break;
  case ExitPlan::LoopBody:
    numPickResults = 0;
    break;
  }
  auto pick = hir::PickOp::create(builder, loc,
                                  SmallVector<Type>(numPickResults, anyTy),
                                  *choice, /*alternativesCount=*/2);

  // Tag 0: the exit, re-encoded for this block. The payload is always the
  // returned values plus their type operands.
  auto &exitAlt = pick.getAlternatives()[0].emplaceBlock();
  exitAlt.addArguments(SmallVector<Type>(2 * numResults, anyTy),
                       SmallVector<Location>(2 * numResults, loc));
  {
    OpBuilder altBuilder(&exitAlt, exitAlt.end());
    buildExitEndpoint(altBuilder, loc, plan,
                      exitAlt.getArguments().take_front(numResults),
                      exitAlt.getArguments().drop_front(numResults),
                      /*terminatorFlavor=*/false);
  }

  // Tag 1: the continuation. The rest of the block, terminator included,
  // moves in and normalizes recursively in threading mode.
  auto &contAlt = pick.getAlternatives()[1].emplaceBlock();
  contAlt.getOperations().splice(contAlt.end(), block.getOperations(),
                                 std::next(Block::iterator(pick)), block.end());
  if (failed(normalizeBlock(contAlt, plan, /*threading=*/true)))
    return failure();

  // Re-terminate the block after the dispatch pick.
  OpBuilder endBuilder(&block, block.end());
  switch (plan.kind) {
  case ExitPlan::FuncBody:
    if (threading) {
      hir::YieldOp::create(endBuilder, loc, pick.getResults());
    } else {
      ReturnOp::create(endBuilder, loc,
                       pick.getResults().take_front(numResults),
                       pick.getResults().drop_front(numResults));
    }
    break;
  case ExitPlan::Arm:
    if (threading) {
      hir::YieldOp::create(endBuilder, loc, pick.getResults());
    } else {
      SmallVector<Value> slotTypes(plan.slotTypes);
      slotTypes.push_back(choiceType);
      YieldOp::create(endBuilder, loc, pick.getResults(), slotTypes);
    }
    break;
  case ExitPlan::LoopBody:
    UnreachableOp::create(endBuilder, loc);
    break;
  }
  return success();
}

/// Rewrite the terminator of an exit-free block per the block's plan: a
/// terminating return becomes the exit encoding, a yield of a rebuilt
/// if/expr gains the tag-1 choice slot, and loop terminators, unreachables,
/// and loop exits passing through an arm stay untouched.
LogicalResult ExitNormalizer::finishEnd(Block &block, const ExitPlan &plan,
                                        bool threading) {
  auto *terminator = block.getTerminator();
  if (auto returnOp = dyn_cast<ReturnOp>(terminator)) {
    // The single trailing return of the function body is the normal form.
    if (plan.kind == ExitPlan::FuncBody && !threading)
      return success();
    OpBuilder builder(returnOp);
    buildExitEndpoint(builder, returnOp.getLoc(), plan, returnOp.getValues(),
                      returnOp.getTypeOfValues(),
                      /*terminatorFlavor=*/!threading);
    returnOp.erase();
    return success();
  }
  if (auto yieldOp = dyn_cast<YieldOp>(terminator);
      yieldOp && plan.kind == ExitPlan::Arm) {
    // The enclosing op gains a choice slot; this path continues normally.
    OpBuilder builder(yieldOp);
    auto choice = hir::ChoiceOp::create(builder, yieldOp.getLoc(), anyTy, 1,
                                        ValueRange{});
    SmallVector<Value> values(yieldOp.getValues());
    values.push_back(choice.getResult());
    if (threading) {
      hir::YieldOp::create(builder, yieldOp.getLoc(), values);
    } else {
      SmallVector<Value> types(yieldOp.getTypeOfValues());
      types.push_back(choiceType);
      YieldOp::create(builder, yieldOp.getLoc(), values, types);
    }
    yieldOp.erase();
    return success();
  }
  // Break, continue, and unreachable terminators provide nothing to the
  // encoding and stay untouched: breaks were already extended when their
  // loop was rebuilt, and unreachable tails are never picked.
  return success();
}

/// Rebuild one exit-carrying op into choice-yielding form and return its
/// trailing choice result.
FailureOr<Value> ExitNormalizer::normalizeExitOp(Operation *op) {
  if (auto ifOp = dyn_cast<IfOp>(op))
    return normalizeIf(ifOp);
  if (auto exprOp = dyn_cast<ExprOp>(op))
    return normalizeExpr(exprOp);
  if (auto loopOp = dyn_cast<LoopOp>(op))
    return normalizeLoop(loopOp);
  return op->emitOpError()
         << "holds a function exit but is not a structured control flow op "
            "that exit normalization can encode through";
}

FailureOr<Value> ExitNormalizer::normalizeIf(IfOp ifOp) {
  // Materialize an absent else region: it is the empty fall-through path and
  // must produce the tag-1 choice.
  if (ifOp.getElseRegion().empty()) {
    auto &elseBlock = ifOp.getElseRegion().emplaceBlock();
    OpBuilder builder(&elseBlock, elseBlock.end());
    YieldOp::create(builder, ifOp.getLoc(), ValueRange{}, ValueRange{});
  }
  ExitPlan plan{ExitPlan::Arm, SmallVector<Value>(ifOp.getResultTypes())};
  for (auto &region : ifOp->getRegions())
    if (failed(normalizeBlock(region.front(), plan, /*threading=*/false)))
      return failure();

  OpBuilder builder(ifOp);
  SmallVector<Value> typeOperands(ifOp.getResultTypes());
  typeOperands.push_back(choiceType);
  auto newIf = IfOp::create(builder, ifOp.getLoc(),
                            SmallVector<Type>(ifOp.getNumResults() + 1, anyTy),
                            ifOp.getCondition(), typeOperands);
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIf.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  ifOp.erase();
  return newIf.getResults().back();
}

FailureOr<Value> ExitNormalizer::normalizeExpr(ExprOp exprOp) {
  ExitPlan plan{ExitPlan::Arm, SmallVector<Value>(exprOp.getResultTypes())};
  if (failed(normalizeBlock(exprOp.getBody().front(), plan,
                            /*threading=*/false)))
    return failure();

  OpBuilder builder(exprOp);
  SmallVector<Value> typeOperands(exprOp.getResultTypes());
  typeOperands.push_back(choiceType);
  auto newExpr = ExprOp::create(
      builder, exprOp.getLoc(),
      SmallVector<Type>(exprOp.getNumResults() + 1, anyTy), exprOp.getPinAttr(),
      exprOp.getPhaseShiftAttr(), typeOperands);
  newExpr.getBody().takeBody(exprOp.getBody());
  for (auto [oldResult, newResult] :
       llvm::zip(exprOp.getResults(), newExpr.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  exprOp.erase();
  return newExpr.getResults().back();
}

FailureOr<Value> ExitNormalizer::normalizeLoop(LoopOp loopOp) {
  // Existing breaks mean ordinary loop completion: they gain the trailing
  // tag-1 choice slot before exits are rewritten, so only exit paths carry
  // tag 0.
  SmallVector<BreakOp> breaks;
  loopOp.getBody().walk([&](BreakOp breakOp) {
    if (enclosingLoop(breakOp) == loopOp)
      breaks.push_back(breakOp);
  });
  for (auto breakOp : breaks) {
    OpBuilder builder(breakOp);
    auto choice = hir::ChoiceOp::create(builder, breakOp.getLoc(), anyTy, 1,
                                        ValueRange{});
    SmallVector<Value> values(breakOp.getValues());
    values.push_back(choice.getResult());
    SmallVector<Value> types(breakOp.getTypeOfValues());
    types.push_back(choiceType);
    BreakOp::create(builder, breakOp.getLoc(), values, types);
    breakOp.erase();
  }

  ExitPlan plan{ExitPlan::LoopBody,
                SmallVector<Value>(loopOp.getResultTypes())};
  if (failed(normalizeBlock(loopOp.getBody().front(), plan,
                            /*threading=*/false)))
    return failure();

  OpBuilder builder(loopOp);
  SmallVector<Value> typeOperands(loopOp.getResultTypes());
  typeOperands.push_back(choiceType);
  auto newLoop =
      LoopOp::create(builder, loopOp.getLoc(),
                     SmallVector<Type>(loopOp.getNumResults() + 1, anyTy),
                     loopOp.getInits(), loopOp.getInitTypes(), typeOperands);
  newLoop.getBody().takeBody(loopOp.getBody());
  for (auto [oldResult, newResult] :
       llvm::zip(loopOp.getResults(), newLoop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  loopOp.erase();
  return newLoop.getResults().back();
}

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

namespace {
struct NormalizeExitsPass
    : public uir::impl::NormalizeExitsPassBase<NormalizeExitsPass> {
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FuncOp>())
      if (failed(ExitNormalizer(funcOp).run()))
        return signalPassFailure();
  }
};
} // namespace
