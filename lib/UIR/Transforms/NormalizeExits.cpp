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
// terminator into value form, bottom-up per region, leaving one `uir.return`
// as the body terminator. It is phase-blind — no phase information is
// consulted — and runs before phase analysis, so analysis and sorting never
// see an early exit.
//
// The encoding keeps every SSA value at a single phase: the exit *decision*
// and the exit *values* have different phases in general, so they travel as
// separate values and are never fused into one.
//
// - **Encode**: a structured CF op (`uir.if`, `uir.expr`, `uir.loop`) whose
//   regions hold a function exit is rebuilt with extra trailing results: one
//   slot per function result carrying the exit path's returned value, plus a
//   payload-free choice tagging how the region ended (tag 0 = exited, tag 1
//   = fell through). The tag lives at the op's own phase; the value slots
//   are ordinary CF results that the standard floor, forwarding, and
//   choice-ification rules deliver at whatever phase their providers have.
//   Paths that do not provide a slot fill it with `uir.unreachable_value`,
//   never omit it, matching the exact-arity yield/break invariant. On ifs
//   and exprs the slots and tag ride the yield; on loops they ride
//   `uir.break` (existing breaks gain the tag-1 slots, and exit paths become
//   breaks). The value slots' result types are one shared `hir.inferrable`
//   per function result, created once at the body top: every exit site's own
//   type operands unify into it, and being floating it satisfies the type
//   operand rule's one-phase-earlier demand at any phase.
//
// - **Dispatch**: right after the rebuilt op, an `hir.pick` on its tag
//   branches between the exit and the continuation: the tag-0 alternative
//   consumes the op's exit-value slots directly (alternatives carry no
//   block arguments), and the tag-1 alternative receives the rest of the
//   block, where the same rules recurse. Multiple exits in one block
//   right-nest into a linear chain of binary picks, so no continuation is
//   ever duplicated.
//
// How an exit is encoded depends only on the block's parent op — its
// `ExitPlan`. The function body threads the exit values through valued picks
// into the single trailing `uir.return`, whose type operands are the shared
// result-type inferrables. If/expr regions yield their original slots, the
// exit-value slots, and the tag. Loop bodies break out of the loop directly,
// with a resultless dispatch pick and `uir.unreachable` after it, since
// break/continue are ordinary loop-real control flow. Dead code after
// exhaustive exits lands in tag-1 alternatives terminated by
// `uir.unreachable` and is never picked.
//
// The choices and picks this pass creates are executable single-phase
// control flow, not phase-crossing vocabulary: the pick's tag operand is
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
    /// The function body block: exit values thread through valued dispatch
    /// picks and feed the single trailing `uir.return`.
    FuncBody,
    /// A region block of an exit-carrying `uir.if` or `uir.expr`: exit
    /// values and the tag ride the block's yield as trailing slots.
    Arm,
    /// The body block of an exit-carrying `uir.loop`: exit values and the
    /// tag leave the loop as trailing `uir.break` slots.
    LoopBody,
  } kind;
  /// Type operands of the enclosing op's original result slots (empty for
  /// the function body).
  SmallVector<Value> slotTypes;
};

/// An op rebuilt into exit-encoding form: its trailing results are the
/// exit-value slots followed by the tag choice.
struct RebuiltExit {
  Operation *op;

  /// The exit-value slot results.
  ValueRange exitValues(unsigned numResults) const {
    return op->getResults().slice(op->getNumResults() - numResults - 1,
                                  numResults);
  }
  /// The tag choice result.
  Value tag() const { return op->getResults().back(); }
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
  FailureOr<RebuiltExit> normalizeExitOp(Operation *op);
  FailureOr<RebuiltExit> normalizeIf(IfOp ifOp);
  FailureOr<RebuiltExit> normalizeExpr(ExprOp exprOp);
  FailureOr<RebuiltExit> normalizeLoop(LoopOp loopOp);
  void buildExitEndpoint(OpBuilder &builder, Location loc, const ExitPlan &plan,
                         ValueRange values, ValueRange typeOfValues,
                         bool terminatorFlavor);

  FuncOp funcOp;
  /// The one `hir.choice_type` shared by all encodings in this function.
  Value choiceType;
  /// One shared `hir.inferrable` per function result, created at the body
  /// top: the exit-value slot type of every rebuilt op, the unification
  /// target of every exit site's type operands, and the final return's type
  /// operands. Floating, so it satisfies the type operand rule's
  /// one-phase-earlier demand at any phase.
  SmallVector<Value> resultTypeVars;
  /// The function's declared result arity; every exit path provides this
  /// many values.
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
  for (unsigned i = 0; i < numResults; ++i)
    resultTypeVars.push_back(
        hir::InferrableOp::create(builder, funcOp.getLoc()).getResult());
  return normalizeBlock(body, ExitPlan{ExitPlan::FuncBody, {}},
                        /*threading=*/false);
}

/// Build the encoding of a function exit that delivers `values` with
/// `typeOfValues`, per the enclosing block's plan: unify each value's type
/// with the enclosing op's exit slot type, then provide all slots — original
/// slots as `uir.unreachable_value`, exit slots as the values, and the
/// payload-free tag-0 choice. With `terminatorFlavor` the encoding ends in
/// the enclosing region's real terminator (`uir.yield` with type operands);
/// without, it ends in the `hir.yield` that threads through a dispatch pick.
/// Loop bodies always break — their dispatch never threads. The function
/// body threads the bare values.
void ExitNormalizer::buildExitEndpoint(OpBuilder &builder, Location loc,
                                       const ExitPlan &plan, ValueRange values,
                                       ValueRange typeOfValues,
                                       bool terminatorFlavor) {
  for (auto [slotType, valueType] : llvm::zip(resultTypeVars, typeOfValues))
    if (slotType != valueType)
      builder.createOrFold<hir::UnifyOp>(loc, anyTy, slotType, valueType);

  if (plan.kind == ExitPlan::FuncBody) {
    assert(!terminatorFlavor && "the function body's exits always thread");
    hir::YieldOp::create(builder, loc, values);
    return;
  }

  SmallVector<Value> slotValues, slotTypes;
  for (Value slotType : plan.slotTypes) {
    slotValues.push_back(
        UnreachableValueOp::create(builder, loc, anyTy).getResult());
    slotTypes.push_back(slotType);
  }
  llvm::append_range(slotValues, values);
  llvm::append_range(slotTypes, typeOfValues);
  auto choice = hir::ChoiceOp::create(builder, loc, anyTy, 0, ValueRange{});
  slotValues.push_back(choice.getResult());
  slotTypes.push_back(choiceType);

  if (plan.kind == ExitPlan::LoopBody)
    BreakOp::create(builder, loc, slotValues, slotTypes);
  else if (terminatorFlavor)
    YieldOp::create(builder, loc, slotValues, slotTypes);
  else
    hir::YieldOp::create(builder, loc, slotValues);
}

/// Normalize one block: recurse into the first op holding a function exit,
/// rebuild it into exit-encoding form, and dispatch on its tag right after
/// it, moving the rest of the block into the continuation alternative.
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
  auto rebuilt = normalizeExitOp(exitOp);
  if (failed(rebuilt))
    return failure();

  // Dispatch on the tag right after the rebuilt op.
  OpBuilder builder(&block, std::next(Block::iterator(rebuilt->op)));
  unsigned numPickResults;
  switch (plan.kind) {
  case ExitPlan::FuncBody:
    numPickResults = numResults;
    break;
  case ExitPlan::Arm:
    numPickResults = plan.slotTypes.size() + numResults + 1;
    break;
  case ExitPlan::LoopBody:
    numPickResults = 0;
    break;
  }
  auto pick = hir::PickOp::create(builder, loc,
                                  SmallVector<Type>(numPickResults, anyTy),
                                  rebuilt->tag(), /*alternativesCount=*/2);

  // Tag 0: the exit, re-encoded for this block. The exit values are the
  // rebuilt op's exit slots, consumed directly — no block arguments.
  auto &exitAlt = pick.getAlternatives()[0].emplaceBlock();
  {
    OpBuilder altBuilder(&exitAlt, exitAlt.end());
    buildExitEndpoint(altBuilder, loc, plan, rebuilt->exitValues(numResults),
                      resultTypeVars, /*terminatorFlavor=*/false);
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
      ReturnOp::create(endBuilder, loc, pick.getResults(), resultTypeVars);
    }
    break;
  case ExitPlan::Arm:
    if (threading) {
      hir::YieldOp::create(endBuilder, loc, pick.getResults());
    } else {
      SmallVector<Value> slotTypes(plan.slotTypes);
      llvm::append_range(slotTypes, resultTypeVars);
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
/// if/expr gains the unreachable exit slots and the tag-1 choice, and loop
/// terminators, unreachables, and loop exits passing through an arm stay
/// untouched.
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
    // The enclosing op gains the exit slots and the tag; this path continues
    // normally, providing no exit values.
    OpBuilder builder(yieldOp);
    SmallVector<Value> values(yieldOp.getValues());
    SmallVector<Value> types(yieldOp.getTypeOfValues());
    for (Value slotType : resultTypeVars) {
      values.push_back(
          UnreachableValueOp::create(builder, yieldOp.getLoc(), anyTy)
              .getResult());
      types.push_back(slotType);
    }
    auto choice = hir::ChoiceOp::create(builder, yieldOp.getLoc(), anyTy, 1,
                                        ValueRange{});
    values.push_back(choice.getResult());
    types.push_back(choiceType);
    if (threading)
      hir::YieldOp::create(builder, yieldOp.getLoc(), values);
    else
      YieldOp::create(builder, yieldOp.getLoc(), values, types);
    yieldOp.erase();
    return success();
  }
  // Break, continue, and unreachable terminators provide nothing to the
  // encoding and stay untouched: breaks were already extended when their
  // loop was rebuilt, and unreachable tails are never picked.
  return success();
}

/// Rebuild one exit-carrying op into exit-encoding form.
FailureOr<RebuiltExit> ExitNormalizer::normalizeExitOp(Operation *op) {
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

FailureOr<RebuiltExit> ExitNormalizer::normalizeIf(IfOp ifOp) {
  // Materialize an absent else region: it is the empty fall-through path and
  // must provide the unreachable exit slots and the tag-1 choice.
  if (ifOp.getElseRegion().empty()) {
    auto &elseBlock = ifOp.getElseRegion().emplaceBlock();
    OpBuilder builder(&elseBlock, elseBlock.end());
    YieldOp::create(builder, ifOp.getLoc(), ValueRange{}, ValueRange{});
  }
  OpBuilder builder(ifOp);
  ExitPlan plan{ExitPlan::Arm, SmallVector<Value>(ifOp.getResultTypes())};
  for (auto &region : ifOp->getRegions())
    if (failed(normalizeBlock(region.front(), plan, /*threading=*/false)))
      return failure();

  SmallVector<Value> typeOperands(ifOp.getResultTypes());
  llvm::append_range(typeOperands, resultTypeVars);
  typeOperands.push_back(choiceType);
  auto newIf = IfOp::create(
      builder, ifOp.getLoc(),
      SmallVector<Type>(ifOp.getNumResults() + numResults + 1, anyTy),
      ifOp.getCondition(), typeOperands);
  newIf.getThenRegion().takeBody(ifOp.getThenRegion());
  newIf.getElseRegion().takeBody(ifOp.getElseRegion());
  for (auto [oldResult, newResult] :
       llvm::zip(ifOp.getResults(), newIf.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  ifOp.erase();
  return RebuiltExit{newIf};
}

FailureOr<RebuiltExit> ExitNormalizer::normalizeExpr(ExprOp exprOp) {
  OpBuilder builder(exprOp);
  ExitPlan plan{ExitPlan::Arm, SmallVector<Value>(exprOp.getResultTypes())};
  if (failed(normalizeBlock(exprOp.getBody().front(), plan,
                            /*threading=*/false)))
    return failure();

  SmallVector<Value> typeOperands(exprOp.getResultTypes());
  llvm::append_range(typeOperands, resultTypeVars);
  typeOperands.push_back(choiceType);
  auto newExpr = ExprOp::create(
      builder, exprOp.getLoc(),
      SmallVector<Type>(exprOp.getNumResults() + numResults + 1, anyTy),
      exprOp.getPinAttr(), exprOp.getPhaseShiftAttr(), typeOperands);
  newExpr.getBody().takeBody(exprOp.getBody());
  for (auto [oldResult, newResult] :
       llvm::zip(exprOp.getResults(), newExpr.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  exprOp.erase();
  return RebuiltExit{newExpr};
}

FailureOr<RebuiltExit> ExitNormalizer::normalizeLoop(LoopOp loopOp) {
  OpBuilder builder(loopOp);
  ExitPlan plan{ExitPlan::LoopBody,
                SmallVector<Value>(loopOp.getResultTypes())};

  // Existing breaks mean ordinary loop completion: they gain the trailing
  // unreachable exit slots and the tag-1 choice before exits are rewritten,
  // so only exit paths carry tag 0.
  SmallVector<BreakOp> breaks;
  loopOp.getBody().walk([&](BreakOp breakOp) {
    if (enclosingLoop(breakOp) == loopOp)
      breaks.push_back(breakOp);
  });
  for (auto breakOp : breaks) {
    OpBuilder breakBuilder(breakOp);
    SmallVector<Value> values(breakOp.getValues());
    SmallVector<Value> types(breakOp.getTypeOfValues());
    for (Value slotType : resultTypeVars) {
      values.push_back(
          UnreachableValueOp::create(breakBuilder, breakOp.getLoc(), anyTy)
              .getResult());
      types.push_back(slotType);
    }
    auto choice = hir::ChoiceOp::create(breakBuilder, breakOp.getLoc(), anyTy,
                                        1, ValueRange{});
    values.push_back(choice.getResult());
    types.push_back(choiceType);
    BreakOp::create(breakBuilder, breakOp.getLoc(), values, types);
    breakOp.erase();
  }

  if (failed(normalizeBlock(loopOp.getBody().front(), plan,
                            /*threading=*/false)))
    return failure();

  SmallVector<Value> typeOperands(loopOp.getResultTypes());
  llvm::append_range(typeOperands, resultTypeVars);
  typeOperands.push_back(choiceType);
  auto newLoop = LoopOp::create(
      builder, loopOp.getLoc(),
      SmallVector<Type>(loopOp.getNumResults() + numResults + 1, anyTy),
      loopOp.getInits(), loopOp.getInitTypes(), typeOperands);
  newLoop.getBody().takeBody(loopOp.getBody());
  for (auto [oldResult, newResult] :
       llvm::zip(loopOp.getResults(), newLoop.getResults()))
    oldResult.replaceAllUsesWith(newResult);
  loopOp.erase();
  return RebuiltExit{newLoop};
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
