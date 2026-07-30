//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # Thread Sorted Phase Content into Fully Threaded Form
//
// This pass is a cheap, purely structural normalization between phase sorting
// and the phase-blind function splitter. It reads phase numbers off the
// `uir.phase` op attributes and the function's declared argument and result
// phases; it runs no phase analysis. The body and signature regions of every
// `uir.func` follow one set of rules.
//
// Sorted form lets crossings *jump*: a value yielded by phase P may be consumed
// directly by any later phase, function arguments and floating ops are consumed
// at any phase, and externally visible phases without content have no
// `uir.phase` op. Threading removes all of that slack so the splitter sees a
// dense chain of adjacent phases:
//
// - **Materialize** one `uir.phase` op per phase in the contiguous range from
//   the function's earliest phase to its latest — spanning the declared
//   argument and result phases (phase 0 always included), every existing phase
//   op (pre-span and internal alike), and every phase in between. Gap and
//   contentless phases become empty phase ops holding just a
//   `uir.phase_return`.
//
// - **Thread** every crossing hop by hop: a value a phase op at phase Q
//   consumes from outside comes from the immediately preceding phase op (Q-1)
//   as one of its results. Intermediate phases yield the value and pass it on;
//   a function argument enters the chain at its declared phase and is consumed
//   directly only there.
//
// - **Sink floats**: a region-free pure op at the region top level is cloned
//   into each phase region that uses it, before its first use; its operands
//   (necessarily other floats or arguments — a float consuming a phase result
//   is a hard error) are cloned or threaded along the same rules. No floating
//   op survives at the region top level.
//
// - **Signature interface**: the `uir.signature` terminator's operands are the
//   argument and result type values; each is delivered as a result of the
//   phase op at its declared phase, so a phase-P argument's type is available
//   at phase P. Floating type values sink into that phase and thread as above.
//
// Function results already ride the `uir.phase_return` of their declared phase
// in sorted form; the pass asserts this and does not fix violations.
//
// ## Algorithm
//
// Each region is threaded independently. Threading is a two-pass affair because
// a consumer discovered inside a late phase can demand a new result from an
// earlier one. First a *planning* walk records, per phase, the ordered list of
// extra results each phase must yield (the `appends`), keyed so identical
// demands coalesce. Then a *build* pass rebuilds the phase ops in increasing
// phase order — each phase now has its predecessor's final results available —
// remapping every external operand to the preceding phase's result, cloning
// sunk floats, and appending the planned yields. A final signature-remap step
// wires the `uir.signature` operands to their declared phase's results, and the
// obsolete original phase ops and floats are erased.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/MapVector.h"
#include <map>

using namespace mlir;
using namespace silicon;
using namespace uir;

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_THREADPHASESPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

//===----------------------------------------------------------------------===//
// RegionThreader
//===----------------------------------------------------------------------===//

namespace {

/// How a phase op produces one of its appended (threading) results.
struct Producer {
  enum Kind {
    /// Yield the preceding phase op's result at `index` (a threaded crossing).
    Thread,
    /// Yield the function argument at `index` (an argument entering the chain).
    Arg,
    /// Yield a clone of the specific floating value `floatValue` (a sunk
    /// signature type). Cloning the value's whole op delivers exactly the
    /// referenced result, so a multi-result float keeps each result distinct.
    Float,
  } kind;
  unsigned index = 0;
  Value floatValue;
};

/// The origin category of a value referenced across a phase boundary.
enum class Origin {
  /// A function argument (entry block argument of the region).
  Arg,
  /// A result of a `uir.phase` op.
  PhaseResult,
  /// A region-top-level floating pure op.
  Float,
  /// A value defined inside a phase op (an interior value).
  Interior,
};

/// The classification of a referenced value: its origin and the coordinates
/// threading needs (argument index and declared phase, or producing phase op
/// and result index, or the floating op).
struct Classified {
  Origin origin;
  unsigned argIndex = 0;
  int argPhase = 0;
  PhaseOp phaseOp;
  unsigned resultIndex = 0;
  Operation *floatOp = nullptr;
};

/// Threads a single region (body or signature) of a `uir.func` into the fully
/// threaded phase form. Owns the plan and build state for that region.
struct RegionThreader {
  RegionThreader(FuncOp funcOp, Region &region)
      : funcOp(funcOp), region(region), argPhases(funcOp.getArgPhases()),
        resultPhases(funcOp.getResultPhases()) {}

  LogicalResult run();

private:
  FuncOp funcOp;
  Region &region;
  ArrayRef<int32_t> argPhases;
  ArrayRef<int32_t> resultPhases;

  Block *rootBlock = nullptr;
  Operation *terminator = nullptr;
  SmallVector<Operation *> floats;

  int minPhase = 0;
  int maxPhase = 0;

  /// The original `uir.phase` op for each phase, before rebuilding.
  std::map<int, PhaseOp> oldPhase;
  /// The number of results each phase op had before threading.
  std::map<int, unsigned> oldNumResults;
  /// The current phase op for each phase after building (original or rebuilt).
  std::map<int, PhaseOp> newPhase;

  /// The extra results each phase must yield, in order.
  std::map<int, SmallVector<Producer>> appends;
  /// Result index within phase P that carries a threaded crossing. The key is
  /// (phase, kind, argIndex-or-originPhase, resultIndex); `kind` is 'a' for a
  /// function argument and 'p' for a phase-op result.
  std::map<std::tuple<int, char, int, unsigned>, unsigned> threadIndex;
  /// Result index within phase P that carries a specific sunk signature float
  /// value, keyed by (phase, defining op, result number) so distinct results of
  /// one float op get distinct slots.
  std::map<std::tuple<int, Operation *, unsigned>, unsigned> floatResultIndex;

  SmallVector<Operation *> toErase;

  Classified classify(Value value);
  bool isExternal(Value value, PhaseOp phaseOp);

  LogicalResult collect();
  LogicalResult validateResults();
  void computeRange();
  void materialize();

  unsigned ensureThreaded(int phase, char kind, int key, unsigned index);
  LogicalResult planInside(int phase, Value value, bool insideFloat);
  LogicalResult planAsResult(int phase, Value value);
  LogicalResult plan();

  Value resultOf(int phase, unsigned index) {
    return newPhase.at(phase).getResult(index);
  }
  Value resolveInside(int phase, Value value, OpBuilder &builder,
                      DenseMap<Value, Value> &floatMemo);
  Value resolveAsResult(int phase, Value value);
  void remapBody(PhaseOp phaseOp, int phase, OpBuilder &builder,
                 DenseMap<Value, Value> &floatMemo);
  LogicalResult buildPhase(int phase);
  void build();
  LogicalResult remapSignature();
};
} // namespace

//===----------------------------------------------------------------------===//
// Classification
//===----------------------------------------------------------------------===//

/// Classify a value by where it originates. Region-top block arguments are
/// function arguments; `uir.phase` results are phase crossings; region-top pure
/// ops are floats; anything defined inside a phase op is interior.
Classified RegionThreader::classify(Value value) {
  Classified c;
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner()->getParentOp() == funcOp) {
      c.origin = Origin::Arg;
      c.argIndex = blockArg.getArgNumber();
      c.argPhase = argPhases[c.argIndex];
      return c;
    }
    c.origin = Origin::Interior;
    return c;
  }
  Operation *defOp = value.getDefiningOp();
  if (auto phaseOp = dyn_cast<PhaseOp>(defOp)) {
    c.origin = Origin::PhaseResult;
    c.phaseOp = phaseOp;
    c.resultIndex = cast<OpResult>(value).getResultNumber();
    return c;
  }
  if (defOp->getParentOp() == funcOp) {
    c.origin = Origin::Float;
    c.floatOp = defOp;
    return c;
  }
  c.origin = Origin::Interior;
  return c;
}

/// Whether a value used inside `phaseOp` is defined outside it.
bool RegionThreader::isExternal(Value value, PhaseOp phaseOp) {
  Operation *scope = isa<BlockArgument>(value)
                         ? cast<BlockArgument>(value).getOwner()->getParentOp()
                         : value.getDefiningOp();
  return scope != phaseOp.getOperation() && !phaseOp->isProperAncestor(scope);
}

//===----------------------------------------------------------------------===//
// Collection and Range
//===----------------------------------------------------------------------===//

/// Gather the region's top-level phase ops, floats, and terminator, rejecting
/// anything else. The sorted-form input places exactly these at the region top
/// level; any other op kind means the input is not sorted form.
LogicalResult RegionThreader::collect() {
  rootBlock = &region.front();
  terminator = rootBlock->getTerminator();
  if (!isa<UnreachableOp, SignatureOp>(terminator))
    return terminator->emitOpError()
           << "cannot terminate a threaded region; expected 'uir.unreachable' "
              "or 'uir.signature'";

  for (auto &op : rootBlock->without_terminator()) {
    if (auto phaseOp = dyn_cast<PhaseOp>(&op)) {
      int phase = phaseOp.getPhase();
      if (oldPhase.count(phase))
        return phaseOp.emitOpError()
               << "duplicate 'uir.phase' op for phase " << phase;
      oldPhase[phase] = phaseOp;
      oldNumResults[phase] = phaseOp.getNumResults();
      continue;
    }
    // Everything else at the region top level must be a floating op: a
    // region-free value sorting never anchored to a phase. These are pure
    // trees, plus `uir.unreachable_value` — a region-free poison that, like
    // `hir.inferrable`, is deliberately non-`Pure` yet floats like any other.
    bool isFloat = op.getNumRegions() == 0 &&
                   (hir::isEffectivelyPure(&op) || isa<UnreachableValueOp>(op));
    if (!isFloat)
      return op.emitOpError()
             << "is not a 'uir.phase' op, the region terminator, or a floating "
                "op; the input is not in sorted phase form";
    floats.push_back(&op);
  }
  return success();
}

/// Assert the sorted-form guarantee that every function result rides the
/// `uir.phase_return` of its declared phase: every result phase has a phase op,
/// and every phase return there carries exactly the function results declared
/// at that phase. Only the body region carries function results; the signature
/// region has none.
LogicalResult RegionThreader::validateResults() {
  if (isa<SignatureOp>(terminator))
    return success();
  for (auto phase : resultPhases) {
    unsigned expected = llvm::count(resultPhases, phase);
    auto it = oldPhase.find(phase);
    if (it == oldPhase.end())
      return funcOp.emitOpError()
             << "declares a result at phase " << phase
             << " but has no 'uir.phase' op there to carry it";
    auto walk = it->second.getBody().walk([&](PhaseReturnOp ret) {
      if (ret.getValues().size() != expected) {
        ret.emitOpError() << "carries " << ret.getValues().size()
                          << " function results but phase " << phase
                          << " declares " << expected;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walk.wasInterrupted())
      return failure();
  }
  return success();
}

/// Compute the contiguous phase range: the declared span (argument and result
/// phases plus phase 0) widened to cover every existing phase op.
void RegionThreader::computeRange() {
  minPhase = 0;
  maxPhase = 0;
  for (auto phase : argPhases) {
    minPhase = std::min<int>(minPhase, phase);
    maxPhase = std::max<int>(maxPhase, phase);
  }
  for (auto phase : resultPhases) {
    minPhase = std::min<int>(minPhase, phase);
    maxPhase = std::max<int>(maxPhase, phase);
  }
  for (auto &[phase, op] : oldPhase) {
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  }
}

/// Create an empty `uir.phase` op for every phase in the range that has none.
void RegionThreader::materialize() {
  OpBuilder builder(terminator);
  auto loc = terminator->getLoc();
  for (int phase = minPhase; phase <= maxPhase; ++phase) {
    if (oldPhase.count(phase))
      continue;
    auto phaseOp = PhaseOp::create(builder, loc, TypeRange{}, phase);
    auto &block = phaseOp.getBody().emplaceBlock();
    OpBuilder returnBuilder = OpBuilder::atBlockEnd(&block);
    PhaseReturnOp::create(returnBuilder, loc, ValueRange{}, ValueRange{},
                          ValueRange{});
    oldPhase[phase] = phaseOp;
    oldNumResults[phase] = 0;
  }
}

//===----------------------------------------------------------------------===//
// Planning
//===----------------------------------------------------------------------===//

/// Ensure phase `phase` exposes the crossing identified by (kind, key, index)
/// as a result, threading it from its origin phase, and return the result
/// index. `kind` is 'a' for a function argument (key is its index, origin its
/// declared phase) or 'p' for a phase-op result (key is the origin phase).
unsigned RegionThreader::ensureThreaded(int phase, char kind, int key,
                                        unsigned index) {
  auto cacheKey = std::make_tuple(phase, kind, key, index);
  if (auto it = threadIndex.find(cacheKey); it != threadIndex.end())
    return it->second;

  int origin = kind == 'a' ? argPhases[key] : key;
  assert(phase >= origin && "threading a crossing before it exists");

  unsigned result;
  if (phase == origin) {
    if (kind == 'p') {
      // A phase-op result already sits at index `index` of its own phase.
      result = index;
    } else {
      // An argument enters the chain by being yielded from its declared phase.
      result = oldNumResults[phase] + appends[phase].size();
      appends[phase].push_back({Producer::Arg, (unsigned)key, nullptr});
    }
  } else {
    unsigned src = ensureThreaded(phase - 1, kind, key, index);
    result = oldNumResults[phase] + appends[phase].size();
    appends[phase].push_back({Producer::Thread, src, nullptr});
  }
  threadIndex[cacheKey] = result;
  return result;
}

/// Register the demand that `value` be available for use inside phase `phase`.
/// A crossing must be provided by the preceding phase; an argument at this very
/// phase is used directly; a float is cloned in, and its operands are recursed.
/// A float consuming a phase result is a hard error.
LogicalResult RegionThreader::planInside(int phase, Value value,
                                         bool insideFloat) {
  auto c = classify(value);
  switch (c.origin) {
  case Origin::Arg:
    if (c.argPhase > phase)
      return funcOp.emitOpError() << "argument declared at phase " << c.argPhase
                                  << " is used at earlier phase " << phase;
    if (c.argPhase < phase)
      ensureThreaded(phase - 1, 'a', c.argIndex, 0);
    return success();
  case Origin::PhaseResult:
    if (insideFloat)
      return c.phaseOp.emitOpError()
             << "result is consumed by a floating op; floating ops must not "
                "consume phase-op results";
    if (c.phaseOp.getPhase() >= phase)
      return c.phaseOp.emitOpError()
             << "result at phase " << c.phaseOp.getPhase()
             << " is consumed at earlier or equal phase " << phase;
    ensureThreaded(phase - 1, 'p', c.phaseOp.getPhase(), c.resultIndex);
    return success();
  case Origin::Float:
    for (auto operand : c.floatOp->getOperands())
      if (failed(planInside(phase, operand, /*insideFloat=*/true)))
        return failure();
    return success();
  case Origin::Interior:
    // An interior value of another phase never dominates this use, so a
    // verifier-valid input cannot reach here.
    llvm_unreachable("external operand is not arg, phase result, or float");
  }
  llvm_unreachable("all origins handled");
}

/// Register the demand that `value` be a result of phase `phase` (a signature
/// type value at its declared phase). A float is sunk into the phase as an
/// extra result; a crossing or argument is threaded up to the phase.
LogicalResult RegionThreader::planAsResult(int phase, Value value) {
  auto c = classify(value);
  switch (c.origin) {
  case Origin::Arg:
    if (c.argPhase > phase)
      return funcOp.emitOpError()
             << "argument declared at phase " << c.argPhase
             << " types a value at earlier phase " << phase;
    ensureThreaded(phase, 'a', c.argIndex, 0);
    return success();
  case Origin::PhaseResult:
    if (c.phaseOp.getPhase() > phase)
      return c.phaseOp.emitOpError()
             << "result at phase " << c.phaseOp.getPhase()
             << " types a value at earlier phase " << phase;
    ensureThreaded(phase, 'p', c.phaseOp.getPhase(), c.resultIndex);
    return success();
  case Origin::Float: {
    auto key = std::make_tuple(phase, c.floatOp,
                               cast<OpResult>(value).getResultNumber());
    if (floatResultIndex.count(key))
      return success();
    floatResultIndex[key] = oldNumResults[phase] + appends[phase].size();
    appends[phase].push_back({Producer::Float, 0, value});
    for (auto operand : c.floatOp->getOperands())
      if (failed(planInside(phase, operand, /*insideFloat=*/true)))
        return failure();
    return success();
  }
  case Origin::Interior:
    // A phase-interior value never dominates the region terminator.
    llvm_unreachable("signature operand is not arg, phase result, or float");
  }
  llvm_unreachable("all origins handled");
}

/// Walk every phase op and the signature terminator, registering all threading
/// demands into `appends` and `threadIndex`.
LogicalResult RegionThreader::plan() {
  for (int phase = minPhase; phase <= maxPhase; ++phase) {
    auto phaseOp = oldPhase.at(phase);
    LogicalResult status = success();
    phaseOp.getBody().walk([&](Operation *op) {
      for (auto operand : op->getOperands()) {
        if (!isExternal(operand, phaseOp))
          continue;
        if (failed(planInside(phase, operand, /*insideFloat=*/false))) {
          status = failure();
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
    if (failed(status))
      return failure();
  }

  if (auto sigOp = dyn_cast<SignatureOp>(terminator)) {
    for (auto [i, value] : llvm::enumerate(sigOp.getTypeOfArgs()))
      if (failed(planAsResult(argPhases[i], value)))
        return failure();
    for (auto [j, value] : llvm::enumerate(sigOp.getTypeOfResults()))
      if (failed(planAsResult(resultPhases[j], value)))
        return failure();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Building
//===----------------------------------------------------------------------===//

/// The value to use inside phase `phase` for an external `value`: a crossing
/// resolves to the preceding phase's result, an argument at this phase stays,
/// and a float is cloned once (memoized in `floatMemo`) at the builder's
/// insertion point, which advances so the clone tree lands in dependency order.
Value RegionThreader::resolveInside(int phase, Value value, OpBuilder &builder,
                                    DenseMap<Value, Value> &floatMemo) {
  auto c = classify(value);
  switch (c.origin) {
  case Origin::Arg:
    if (c.argPhase == phase)
      return value;
    return resultOf(phase - 1,
                    threadIndex.at({phase - 1, 'a', (int)c.argIndex, 0}));
  case Origin::PhaseResult:
    return resultOf(
        phase - 1,
        threadIndex.at({phase - 1, 'p', c.phaseOp.getPhase(), c.resultIndex}));
  case Origin::Float: {
    if (auto it = floatMemo.find(value); it != floatMemo.end())
      return it->second;
    IRMapping map;
    for (auto operand : c.floatOp->getOperands())
      map.map(operand, resolveInside(phase, operand, builder, floatMemo));
    Operation *clone = builder.clone(*c.floatOp, map);
    builder.setInsertionPointAfter(clone);
    for (auto [orig, cloned] :
         llvm::zip(c.floatOp->getResults(), clone->getResults()))
      floatMemo[orig] = cloned;
    return floatMemo[value];
  }
  case Origin::Interior:
    break;
  }
  llvm_unreachable("interior values are not resolved");
}

/// The result value of phase `phase` that delivers an external signature
/// `value`.
Value RegionThreader::resolveAsResult(int phase, Value value) {
  auto c = classify(value);
  switch (c.origin) {
  case Origin::Arg:
    return resultOf(phase, threadIndex.at({phase, 'a', (int)c.argIndex, 0}));
  case Origin::PhaseResult:
    return resultOf(phase, threadIndex.at({phase, 'p', c.phaseOp.getPhase(),
                                           c.resultIndex}));
  case Origin::Float:
    return resultOf(
        phase, floatResultIndex.at({phase, c.floatOp,
                                    cast<OpResult>(value).getResultNumber()}));
  case Origin::Interior:
    break;
  }
  llvm_unreachable("interior values are not resolved");
}

/// Redirect every external operand inside `phaseOp` to its threaded source: the
/// preceding phase's result, the argument itself, or a cloned float. Floats
/// cloned here land, via `builder` and `floatMemo`, at the entry block front so
/// they dominate every use in the phase.
void RegionThreader::remapBody(PhaseOp phaseOp, int phase, OpBuilder &builder,
                               DenseMap<Value, Value> &floatMemo) {
  SmallVector<Operation *> ops;
  phaseOp.getBody().walk([&](Operation *op) { ops.push_back(op); });
  for (auto *op : ops)
    for (auto &operand : op->getOpOperands())
      if (isExternal(operand.get(), phaseOp))
        operand.set(resolveInside(phase, operand.get(), builder, floatMemo));
}

/// Rebuild phase `phase` with its planned appended results, or keep it as is if
/// nothing threads through it. Assumes all earlier phases are already built.
LogicalResult RegionThreader::buildPhase(int phase) {
  auto oldOp = oldPhase.at(phase);
  Block &entry = oldOp.getBody().front();

  // All float clones — for both body uses and sunk signature types — land at
  // the entry block front, in dependency order, so they dominate every use and
  // ride into the rebuilt op unchanged.
  OpBuilder floatBuilder = OpBuilder::atBlockBegin(&entry);
  DenseMap<Value, Value> floatMemo;
  remapBody(oldOp, phase, floatBuilder, floatMemo);

  auto &producers = appends[phase];
  if (producers.empty()) {
    newPhase[phase] = oldOp;
    oldOp->moveBefore(terminator);
    return success();
  }

  // Materialize each appended value before rebuilding the op: threaded
  // crossings and entering arguments are defined outside and dominate every
  // phase return; sunk floats are cloned at the entry front like any other.
  SmallVector<Value> appendedValues;
  for (auto &producer : producers) {
    switch (producer.kind) {
    case Producer::Thread:
      appendedValues.push_back(resultOf(phase - 1, producer.index));
      break;
    case Producer::Arg:
      appendedValues.push_back(rootBlock->getArgument(producer.index));
      break;
    case Producer::Float:
      appendedValues.push_back(
          resolveInside(phase, producer.floatValue, floatBuilder, floatMemo));
      break;
    }
  }

  // Create the replacement phase op with one extra result per appended
  // producer, and move the (already remapped) body into it.
  OpBuilder builder(terminator);
  auto anyTy = hir::AnyType::get(funcOp.getContext());
  SmallVector<Type> resultTypes(oldNumResults[phase] + producers.size(), anyTy);
  auto newOp = PhaseOp::create(builder, oldOp.getLoc(), resultTypes, phase);
  newOp.getBody().takeBody(oldOp.getBody());
  newPhase[phase] = newOp;

  // Append the values to every phase return's yield list. A phase receiving a
  // crossing must have somewhere to carry it: a fully divergent phase op
  // (terminated by a bare `uir.unreachable` with no `uir.phase_return`
  // anywhere) can never re-yield, so a crossing threaded through it is a
  // sorted-form contradiction the sorter never produces.
  SmallVector<PhaseReturnOp> returns;
  newOp.getBody().walk([&](PhaseReturnOp ret) { returns.push_back(ret); });
  assert(!returns.empty() &&
         "crossing threaded through a phase op with no 'uir.phase_return'");
  for (auto ret : returns) {
    OpBuilder returnBuilder(ret);
    SmallVector<Value> yields(ret.getYields());
    yields.append(appendedValues.begin(), appendedValues.end());
    PhaseReturnOp::create(returnBuilder, ret.getLoc(), ret.getValues(),
                          ret.getTypeOfValues(), yields);
    ret.erase();
  }

  toErase.push_back(oldOp);
  return success();
}

/// Build every phase in increasing order.
void RegionThreader::build() {
  for (int phase = minPhase; phase <= maxPhase; ++phase)
    // Building only fails on malformed threadIndex lookups, which planning
    // rules out; treat a failure as a compiler bug.
    if (failed(buildPhase(phase)))
      llvm_unreachable("phase build failed after successful planning");
}

/// Wire the `uir.signature` terminator's operands to their declared phase's
/// results.
LogicalResult RegionThreader::remapSignature() {
  auto sigOp = dyn_cast<SignatureOp>(terminator);
  if (!sigOp)
    return success();
  for (auto [i, operand] : llvm::enumerate(sigOp.getTypeOfArgsMutable()))
    operand.set(resolveAsResult(argPhases[i], operand.get()));
  for (auto [j, operand] : llvm::enumerate(sigOp.getTypeOfResultsMutable()))
    operand.set(resolveAsResult(resultPhases[j], operand.get()));
  return success();
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

LogicalResult RegionThreader::run() {
  if (failed(collect()) || failed(validateResults()))
    return failure();
  computeRange();
  materialize();
  if (failed(plan()))
    return failure();
  build();
  if (failed(remapSignature()))
    return failure();

  // Erase the replaced phase ops and now-dead floating ops. Floats are erased
  // in reverse program order so a float feeding another float is removed after
  // its user.
  for (auto *op : toErase)
    op->erase();
  for (auto *floatOp : llvm::reverse(floats)) {
    assert(floatOp->use_empty() && "floating op still used after threading");
    floatOp->erase();
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct ThreadPhasesPass
    : public uir::impl::ThreadPhasesPassBase<ThreadPhasesPass> {
  void runOnOperation() override {
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      if (failed(RegionThreader(funcOp, funcOp.getSignature()).run()) ||
          failed(RegionThreader(funcOp, funcOp.getBody()).run()))
        return signalPassFailure();
    }
  }
};
} // namespace
