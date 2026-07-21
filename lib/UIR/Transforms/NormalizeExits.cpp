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
// A `uir.func` has two regions that follow one set of rules: the body, whose
// return-like exit is `uir.return`, and the signature, whose return-like
// exit is `uir.signature`. This pass rewrites every exit op that is not its
// region's terminator into value form, bottom-up per region, leaving one
// exit op as the region terminator. It is phase-blind — no phase information
// is consulted — and runs before phase analysis, so analysis and sorting
// never see an early exit.
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
//   per exit slot, created once at the region top: every exit site's own
//   type operands unify into it (a `uir.signature` site carries none, so its
//   unify set is empty and the slots use the shared inferrables directly),
//   and being floating it satisfies the type operand rule's one-phase-
//   earlier demand at any phase.
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
// `ExitPlan`. The root region block threads the exit values through valued
// picks into the single trailing exit op: for the body a `uir.return` whose
// type operands are the shared slot-type inferrables, for the signature a
// `uir.signature` whose operands are the delivered arg- and result-type
// slots. If/expr regions yield their original slots, the
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
    /// The root region block (function body or signature): exit values
    /// thread through valued dispatch picks and feed the single trailing
    /// exit op (`uir.return` or `uir.signature`).
    Root,
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

/// Rewrites all early exit ops out of one region of a `uir.func`: the body
/// (exit op `uir.return`) or the signature (exit op `uir.signature`). Both
/// regions follow one set of rules; only what an exit site delivers and how
/// the final root terminator is spelled differ.
class ExitNormalizer {
public:
  ExitNormalizer(FuncOp funcOp, Region &region)
      : funcOp(funcOp), region(region) {}
  LogicalResult run();

private:
  bool isSignature() { return &region == &funcOp.getSignature(); }
  bool isExit(Operation *op) {
    return isSignature() ? isa<SignatureOp>(op) : isa<ReturnOp>(op);
  }
  bool containsExit(Operation *op);
  SmallVector<Value> exitSiteValues(Operation *op);
  SmallVector<Value> exitSiteTypes(Operation *op);
  void buildRootTerminator(OpBuilder &builder, Location loc, ValueRange values);
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
  /// The region being normalized: the function's body or signature.
  Region &region;
  /// The one `hir.choice_type` shared by all encodings in this region.
  Value choiceType;
  /// One shared `hir.inferrable` per exit slot, created at the region top:
  /// the exit-value slot type of every rebuilt op, the unification target of
  /// every exit site's type operands, and the final return's type operands.
  /// Floating, so it satisfies the type operand rule's one-phase-earlier
  /// demand at any phase.
  SmallVector<Value> resultTypeVars;
  /// The number of exit slots every exit path provides: the function's
  /// declared result arity for the body, its argument plus result arity for
  /// the signature (one type value per argument and per result).
  unsigned numResults = 0;
  Type anyTy;
};
} // namespace

/// Whether `op` transitively contains this region's exit op.
bool ExitNormalizer::containsExit(Operation *op) {
  return op
      ->walk([&](Operation *inner) {
        return isExit(inner) ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
}

/// The values an exit site delivers, one per exit slot: a `uir.return`'s
/// return values, or a `uir.signature`'s argument and result type values.
SmallVector<Value> ExitNormalizer::exitSiteValues(Operation *op) {
  if (auto sigOp = dyn_cast<SignatureOp>(op)) {
    SmallVector<Value> values(sigOp.getTypeOfArgs());
    llvm::append_range(values, sigOp.getTypeOfResults());
    return values;
  }
  return SmallVector<Value>(cast<ReturnOp>(op).getValues());
}

/// The per-slot type operands an exit site carries: a `uir.return`'s type
/// operands, which unify into the shared slot-type inferrables. A
/// `uir.signature` delivers type values and carries no type operands of its
/// own, so its slots are typed by the shared inferrables directly and the
/// unify set is empty.
SmallVector<Value> ExitNormalizer::exitSiteTypes(Operation *op) {
  if (isa<SignatureOp>(op))
    return resultTypeVars;
  return SmallVector<Value>(cast<ReturnOp>(op).getTypeOfValues());
}

/// Build the region's single trailing exit op from the threaded slot values.
void ExitNormalizer::buildRootTerminator(OpBuilder &builder, Location loc,
                                         ValueRange values) {
  if (isSignature()) {
    unsigned numArgs = funcOp.getArgPhases().size();
    SignatureOp::create(builder, loc, values.take_front(numArgs),
                        values.drop_front(numArgs));
  } else {
    ReturnOp::create(builder, loc, values, resultTypeVars);
  }
}

/// The nearest enclosing `uir.loop` of `op`, or null.
static LoopOp enclosingLoop(Operation *op) {
  for (auto *parent = op->getParentOp(); parent; parent = parent->getParentOp())
    if (auto loopOp = dyn_cast<LoopOp>(parent))
      return loopOp;
  return {};
}

LogicalResult ExitNormalizer::run() {
  // Nothing to do unless some exit op is not the region terminator. The
  // verifier keeps each region free of the other region's exit kind.
  auto &block = region.front();
  auto *terminator = block.getTerminator();
  auto walkResult = region.walk([&](Operation *op) {
    return isExit(op) && op != terminator ? WalkResult::interrupt()
                                          : WalkResult::advance();
  });
  if (!walkResult.wasInterrupted())
    return success();

  numResults = funcOp.getResultPhases().size();
  if (isSignature())
    numResults += funcOp.getArgPhases().size();
  anyTy = hir::AnyType::get(funcOp.getContext());
  OpBuilder builder(&block, block.begin());
  choiceType = hir::ChoiceTypeOp::create(builder, funcOp.getLoc()).getResult();
  for (unsigned i = 0; i < numResults; ++i)
    resultTypeVars.push_back(
        hir::InferrableOp::create(builder, funcOp.getLoc()).getResult());
  return normalizeBlock(block, ExitPlan{ExitPlan::Root, {}},
                        /*threading=*/false);
}

/// Build the encoding of a function exit that delivers `values` with
/// `typeOfValues`, per the enclosing block's plan: unify each value's type
/// with the enclosing op's exit slot type, then provide all slots — original
/// slots as `uir.unreachable_value`, exit slots as the values, and the
/// payload-free tag-0 choice. With `terminatorFlavor` the encoding ends in
/// the enclosing region's real terminator (`uir.yield` with type operands);
/// without, it ends in the `hir.yield` that threads through a dispatch pick.
/// Loop bodies always break — their dispatch never threads. The root region
/// block threads the bare values.
void ExitNormalizer::buildExitEndpoint(OpBuilder &builder, Location loc,
                                       const ExitPlan &plan, ValueRange values,
                                       ValueRange typeOfValues,
                                       bool terminatorFlavor) {
  for (auto [slotType, valueType] : llvm::zip(resultTypeVars, typeOfValues))
    if (slotType != valueType)
      builder.createOrFold<hir::UnifyOp>(loc, anyTy, slotType, valueType);

  if (plan.kind == ExitPlan::Root) {
    assert(!terminatorFlavor && "the root region block's exits always thread");
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
    if (containsExit(&op)) {
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
  case ExitPlan::Root:
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
  case ExitPlan::Root:
    if (threading) {
      hir::YieldOp::create(endBuilder, loc, pick.getResults());
    } else {
      buildRootTerminator(endBuilder, loc, pick.getResults());
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
/// terminating exit op becomes the exit encoding, a yield of a rebuilt
/// if/expr gains the unreachable exit slots and the tag-1 choice, and loop
/// terminators, unreachables, and loop exits passing through an arm stay
/// untouched.
LogicalResult ExitNormalizer::finishEnd(Block &block, const ExitPlan &plan,
                                        bool threading) {
  auto *terminator = block.getTerminator();
  if (isExit(terminator)) {
    // The single trailing exit op of the root region block is the normal
    // form.
    if (plan.kind == ExitPlan::Root && !threading)
      return success();
    OpBuilder builder(terminator);
    buildExitEndpoint(builder, terminator->getLoc(), plan,
                      exitSiteValues(terminator), exitSiteTypes(terminator),
                      /*terminatorFlavor=*/!threading);
    terminator->erase();
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
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      if (failed(ExitNormalizer(funcOp, funcOp.getSignature()).run()))
        return signalPassFailure();
      if (failed(ExitNormalizer(funcOp, funcOp.getBody()).run()))
        return signalPassFailure();
    }
  }
};
} // namespace
