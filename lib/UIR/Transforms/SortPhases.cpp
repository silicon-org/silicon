//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # Sort Function Content Across Phase Boundaries
//
// This is the sorting stage of the SplitPhases pass. It rewrites a `uir.func`
// such that no control flow region contains content of a phase different
// from the region's own phase, which reduces the subsequent phase splitting
// to distributing straight-line ops.
//
// The phase map computed by PhaseAnalysis is the single source of truth for
// op and value phases throughout: every rewrite step erases the map entries
// of the ops it destroys and records entries for the ops it creates.
//
// The stage runs these steps in order:
//
// 1. **Dissolve phase grouping.** `uir.expr` bodies are inlined into their
//    parent block and `uir.pin` outputs are replaced by their inputs. The
//    grouping ops exist to communicate phase constraints to PhaseAnalysis;
//    once the phase map is computed they only stand in the way of moving
//    individual ops across region boundaries.
//
// 2. **Decompose calls.** Each `uir.call` becomes one `hir.call` per entry
//    of the callee's `uir.split_func` witness, placed at the call site with
//    opaque context values chaining consecutive entries. Sorting requires
//    this: the callee's per-phase halves are separate ops afterwards, so an
//    earlier-phase half under later-phase control flow hoists like any
//    other earlier-phase op. Callees must already be split when a caller
//    is sorted (the pass processes functions callees-first), so a missing
//    witness is a hard error.
//
// 3. **Sort per boundary.** For each phase boundary p (earliest first),
//    hoisting runs before extraction:
//
//    - **Hoist up**: ops at phase <= p transitively inside regions of CF
//      ops at phase > p move, in program order, to immediately before the
//      outermost such CF op. Per the D-rule, earlier-phase execution is
//      governed solely by earlier-phase control flow, so the hoisted ops
//      execute unconditionally and their results reach back into the
//      region as ordinary SSA uses. Hoisting must precede extraction:
//      extracting first would drag earlier-phase ops into a replicate body
//      and change their multiplicity.
//
//    - **Extract down**: each CF region op at phase p that still contains
//      later-phase content moves that content, in program order, into a
//      single receiving construct placed immediately after it. Loop-shaped
//      regions (uir.loop bodies, hir.replicate bodies) get a hit list
//      created before the op, a record pack appended per activation, and
//      an hir.replicate; branch-shaped regions (uir.if arms, hir.pick
//      alternatives) get a per-arm hir.choice yielded as an extra result
//      and an hir.pick. Values defined inside the region and used by the
//      moved content become record fields / choice payloads; values
//      defined outside are used directly; floating constants are cloned.
//      Later-phase results of a branch are removed from the CF op and
//      reconstructed as pick results. Receiving constructs count as CF
//      regions for the next boundary, which is where distance-2 nesting
//      comes from.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/SortPhases.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

#define DEBUG_TYPE "sort-phases"

//===----------------------------------------------------------------------===//
// Sorter
//===----------------------------------------------------------------------===//

namespace {
/// Sorts the content of a single `uir.func` across phase boundaries and
/// maintains the phase map for every op it erases or creates.
struct Sorter {
  Sorter(FuncOp funcOp, PhaseAnalysis &analysis, SymbolTable &symbolTable)
      : funcOp(funcOp), analysis(analysis), symbolTable(symbolTable) {}

  LogicalResult run();

private:
  FuncOp funcOp;
  PhaseAnalysis &analysis;
  SymbolTable &symbolTable;

  LogicalResult dissolveExprsAndPins();
  LogicalResult decomposeCalls();
  LogicalResult decomposeCall(CallOp callOp);

  std::pair<int16_t, int16_t> computePhaseRange();
  void hoistBoundary(int16_t p);
  LogicalResult extractBoundary(int16_t p);
  LogicalResult extractFromBranch(Operation *op, int16_t p);
  LogicalResult extractFromLoop(Operation *op, int16_t p);

  bool hasLaterContent(Operation *op, int16_t p);
  SmallVector<Operation *> collectLaterContent(Block &block, int16_t p);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp,
                                               ArrayRef<Operation *> movedOps);
  void remapIntoReceivingBlock(Operation *regionOp, Block &receivingBlock,
                               const llvm::SetVector<Value> &fields);

  /// Whether a phase is a real phase (not the float/never sentinel).
  static bool isRealPhase(int16_t phase) {
    return phase != PhaseAnalysis::kFloatPhase &&
           phase != PhaseAnalysis::kNeverPhase;
  }

  /// The real phase of an op, or nullopt for sentinels and unmapped ops.
  std::optional<int16_t> realPhaseOf(Operation *op) {
    auto it = analysis.opPhases.find(op);
    if (it == analysis.opPhases.end() || !isRealPhase(it->second))
      return std::nullopt;
    return it->second;
  }

  /// Record the phase of a created op and all of its results.
  void setPhase(Operation *op, int16_t phase) {
    analysis.opPhases[op] = phase;
    for (auto result : op->getResults())
      analysis.actualPhase[result] = phase;
  }

  /// Drop the phase map entries of an op about to be erased. Op and value
  /// pointers can be reused by later allocations, so stale entries must
  /// never linger.
  void forget(Operation *op) {
    analysis.opPhases.erase(op);
    for (auto result : op->getResults())
      analysis.actualPhase.erase(result);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Step 1: Dissolve Phase Grouping
//===----------------------------------------------------------------------===//

/// Inline all `uir.expr` bodies and replace all `uir.pin` outputs with their
/// inputs, in both the signature and body regions. Walks post-order so inner
/// exprs dissolve before their parents.
LogicalResult Sorter::dissolveExprsAndPins() {
  // Collect first: splicing while walking invalidates the walk.
  SmallVector<ExprOp> exprs;
  SmallVector<PinOp> pins;
  funcOp->walk([&](Operation *op) {
    if (auto exprOp = dyn_cast<ExprOp>(op))
      exprs.push_back(exprOp);
    else if (auto pinOp = dyn_cast<PinOp>(op))
      pins.push_back(pinOp);
  });

  for (auto exprOp : exprs) {
    auto &block = exprOp.getBody().front();
    auto yieldOp = dyn_cast<YieldOp>(block.getTerminator());
    if (!yieldOp)
      return exprOp.emitOpError()
             << "cannot be dissolved: body is terminated by "
             << block.getTerminator()->getName() << " instead of a yield";
    for (auto [result, value] :
         llvm::zip(exprOp.getResults(), yieldOp.getValues()))
      result.replaceAllUsesWith(value);
    forget(yieldOp);
    yieldOp.erase();
    exprOp->getBlock()->getOperations().splice(exprOp->getIterator(),
                                               block.getOperations());
    forget(exprOp);
    exprOp.erase();
  }

  for (auto pinOp : pins) {
    for (auto [output, input] :
         llvm::zip(pinOp.getOutputs(), pinOp.getInputs()))
      output.replaceAllUsesWith(input);
    forget(pinOp);
    pinOp.erase();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// Step 2: Decompose Calls
//===----------------------------------------------------------------------===//

LogicalResult Sorter::decomposeCalls() {
  SmallVector<CallOp> calls;
  funcOp->walk([&](CallOp op) { calls.push_back(op); });
  for (auto callOp : calls)
    if (failed(decomposeCall(callOp)))
      return failure();
  return success();
}

/// Decompose a `uir.call` into per-phase `hir.call` ops using the callee's
/// `uir.split_func` witness. All pieces are created at the call site in
/// entry order; sorting later moves each piece to where its phase belongs.
/// Consecutive entries are chained through an opaque context value.
LogicalResult Sorter::decomposeCall(CallOp callOp) {
  auto calleeName = callOp.getCallee();
  auto calleeSplitFunc = symbolTable.lookup<SplitFuncOp>(calleeName);
  if (!calleeSplitFunc)
    return callOp.emitOpError()
           << "callee @" << calleeName
           << " has no uir.split_func witness; callees must be split before "
              "their callers are sorted";

  int16_t callOpPhase = analysis.getPhase(callOp);
  auto callArgPhases = callOp.getArgPhases();
  auto callResultPhases = callOp.getResultPhases();
  auto loc = callOp.getLoc();
  auto anyTy = hir::AnyType::get(callOp.getContext());

  // Gather the witness entries with their absolute caller phases and the
  // per-phase functions' result counts.
  struct CallEntry {
    int16_t phase;
    StringRef symbolName;
  };
  SmallVector<CallEntry> entries;
  for (auto [entryPhase, funcRef] :
       llvm::zip(calleeSplitFunc.getPhaseNumbers(),
                 calleeSplitFunc.getPhaseFuncs())) {
    auto name = cast<FlatSymbolRefAttr>(funcRef).getValue();
    entries.push_back(
        {static_cast<int16_t>(callOpPhase + entryPhase), name});
  }
  assert(!entries.empty() && "split_func witness has no phase entries");

  // Partition arguments, results, and their type operands by absolute phase.
  DenseMap<int16_t, SmallVector<Value>> phaseArgs, phaseTypeOfArgs;
  for (auto [arg, type, offset] : llvm::zip(
           callOp.getArguments(), callOp.getTypeOfArgs(), callArgPhases)) {
    int16_t abs = callOpPhase + static_cast<int16_t>(offset);
    phaseArgs[abs].push_back(arg);
    phaseTypeOfArgs[abs].push_back(type);
  }
  DenseMap<int16_t, SmallVector<std::pair<unsigned, Value>>> phaseResults;
  for (auto [i, offset] : llvm::enumerate(callResultPhases)) {
    int16_t abs = callOpPhase + static_cast<int16_t>(offset);
    phaseResults[abs].push_back({i, callOp.getTypeOfResults()[i]});
  }

  // Create one hir.call per entry, chaining opaque context between
  // consecutive entries.
  OpBuilder builder(callOp);
  int16_t lastPhase = entries.back().phase;
  Value prevContext;
  SmallVector<Value> replacements(callOp.getNumResults());

  for (auto &entry : entries) {
    SmallVector<Value> callArgs(phaseArgs[entry.phase]);
    SmallVector<Value> callTypeOfArgs(phaseTypeOfArgs[entry.phase]);
    if (prevContext) {
      callArgs.push_back(prevContext);
      auto opaqueType = hir::OpaqueTypeOp::create(builder, loc);
      setPhase(opaqueType, entry.phase);
      callTypeOfArgs.push_back(opaqueType.getResult());
    }

    // User-visible results at this entry's phase, plus an opaque context
    // result for all but the last entry.
    auto &entryResults = phaseResults[entry.phase];
    SmallVector<Value> callTypeOfResults;
    SmallVector<Type> resultTypes;
    for (auto &[idx, typeOfResult] : entryResults) {
      callTypeOfResults.push_back(typeOfResult);
      resultTypes.push_back(anyTy);
    }
    bool needsContext = entry.phase != lastPhase;
    if (needsContext) {
      auto opaqueType = hir::OpaqueTypeOp::create(builder, loc);
      setPhase(opaqueType, entry.phase);
      callTypeOfResults.push_back(opaqueType.getResult());
      resultTypes.push_back(anyTy);
    }

    auto call = hir::CallOp::create(
        builder, loc, resultTypes,
        FlatSymbolRefAttr::get(builder.getContext(), entry.symbolName),
        callArgs, callTypeOfArgs, callTypeOfResults);
    analysis.opPhases[call] = entry.phase;
    for (auto [i, pair] : llvm::enumerate(entryResults)) {
      analysis.actualPhase[call.getResult(i)] = entry.phase;
      replacements[pair.first] = call.getResult(i);
    }
    if (needsContext) {
      prevContext = call.getResult(entryResults.size());
      analysis.actualPhase[prevContext] = entry.phase;
    } else {
      prevContext = {};
    }
  }

  for (auto [result, replacement] :
       llvm::zip(callOp.getResults(), replacements))
    if (!replacement && !result.use_empty())
      return callOp.emitOpError()
             << "result " << result.getResultNumber()
             << " has no phase entry in the callee's uir.split_func witness";

  callOp.replaceAllUsesWith(replacements);
  forget(callOp);
  callOp.erase();
  return success();
}

//===----------------------------------------------------------------------===//
// Step 3: Sort per Boundary
//===----------------------------------------------------------------------===//

/// The range of real phases present in the function: declared arg and result
/// phases plus everything recorded in the phase map.
std::pair<int16_t, int16_t> Sorter::computePhaseRange() {
  int16_t minPhase = 0, maxPhase = 0;
  auto extend = [&](int16_t phase) {
    minPhase = std::min(minPhase, phase);
    maxPhase = std::max(maxPhase, phase);
  };
  for (auto phase : funcOp.getArgPhases())
    extend(static_cast<int16_t>(phase));
  for (auto phase : funcOp.getResultPhases())
    extend(static_cast<int16_t>(phase));
  for (auto [op, phase] : analysis.opPhases)
    if (isRealPhase(phase))
      extend(phase);
  return {minPhase, maxPhase};
}

/// Clone the defining tree of a floating pure value (constants and pure ops
/// with only floating operands) at the builder's insertion point. Returns
/// the cloned value.
static Value cloneFloatingTree(Value value, OpBuilder &builder,
                               IRMapping &mapping) {
  if (auto mapped = mapping.lookupOrNull(value))
    return mapped;
  auto *defOp = value.getDefiningOp();
  assert(defOp && "floating value must have a defining op");
  for (auto operand : defOp->getOperands())
    cloneFloatingTree(operand, builder, mapping);
  builder.clone(*defOp, mapping);
  return mapping.lookup(value);
}

/// Hoist ops at phase <= p out of regions of CF ops at phase > p, to
/// immediately before the outermost such CF op, preserving program order.
/// Floating constants used by hoisted ops but defined inside the region are
/// cloned in front of their users.
void Sorter::hoistBoundary(int16_t p) {
  // Find the maximal later-phase region ops: walk the body, but do not
  // descend into a later-phase op — it is the hoist target for everything
  // earlier inside it.
  SmallVector<Operation *> roots;
  std::function<void(Block &)> findRoots = [&](Block &block) {
    for (auto &op : block) {
      if (op.getNumRegions() > 0) {
        if (auto ph = realPhaseOf(&op); ph && *ph > p) {
          roots.push_back(&op);
          continue;
        }
      }
      for (auto &region : op.getRegions())
        for (auto &inner : region)
          findRoots(inner);
    }
  };
  findRoots(funcOp.getBody().front());

  for (auto *root : roots) {
    // Collect ops at real phase <= p transitively inside the root, in
    // program order. An op moves wholesale with its regions; its interior
    // is not scanned separately.
    SmallVector<Operation *> hoisted;
    std::function<void(Block &)> collect = [&](Block &block) {
      for (auto &op : block) {
        if (auto ph = realPhaseOf(&op); ph && *ph <= p) {
          hoisted.push_back(&op);
          continue;
        }
        for (auto &region : op.getRegions())
          for (auto &inner : region)
            collect(inner);
      }
    };
    for (auto &region : root->getRegions())
      for (auto &block : region)
        collect(block);
    if (hoisted.empty())
      continue;

    LLVM_DEBUG(llvm::dbgs() << "  hoisting " << hoisted.size() << " ops out of "
                            << root->getName() << " at boundary " << p << "\n");
    for (auto *op : hoisted)
      op->moveBefore(root);

    // Hoisted ops may still use floating constants defined inside the root;
    // clone those in front of the hoisted user. Any other remaining use of
    // an interior value is a causality violation the analysis must have
    // rejected, so it is a compiler bug here.
    for (auto *op : hoisted) {
      op->walk([&](Operation *user) {
        for (auto &use : user->getOpOperands()) {
          auto *defOp = use.get().getDefiningOp();
          if (!defOp || !root->isProperAncestor(defOp))
            continue;
          assert(!isRealPhase(analysis.getPhase(defOp)) &&
                 hir::isEffectivelyPure(defOp) &&
                 "hoisted op uses non-floating value from later-phase region");
          OpBuilder builder(user);
          IRMapping mapping;
          auto clone = cloneFloatingTree(use.get(), builder, mapping);
          clone.getDefiningOp()->walk(
              [&](Operation *cloned) { setPhase(cloned, *realPhaseOf(op)); });
          use.set(clone);
        }
      });
    }
  }
}

//===----------------------------------------------------------------------===//
// Step 4: Extract Later-Phase Content
//===----------------------------------------------------------------------===//

/// Whether any op transitively inside the given op sits at a real phase
/// later than p.
bool Sorter::hasLaterContent(Operation *root, int16_t p) {
  auto result = root->walk([&](Operation *op) {
    if (op == root)
      return WalkResult::advance();
    if (auto ph = realPhaseOf(op); ph && *ph > p)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// The top-level ops of a block at a real phase later than p, in program
/// order. After inner extraction, all later-phase content of a region sits
/// at the top level of its block.
SmallVector<Operation *> Sorter::collectLaterContent(Block &block, int16_t p) {
  SmallVector<Operation *> ops;
  for (auto &op : block)
    if (auto ph = realPhaseOf(&op); ph && *ph > p)
      ops.push_back(&op);
  return ops;
}

/// Collect the values defined inside `regionOp` (excluding the moved ops
/// themselves) that the moved ops use: these cross the phase boundary and
/// become record fields / choice payloads. Values defined outside the
/// region op remain direct uses (the receiving construct sits right after
/// the op, so they dominate it), and floating values are cloned instead.
llvm::SetVector<Value>
Sorter::collectCrossingValues(Operation *regionOp,
                              ArrayRef<Operation *> movedOps) {
  auto isMoved = [&](Operation *op) {
    return llvm::any_of(movedOps, [&](Operation *m) {
      return m == op || m->isProperAncestor(op);
    });
  };

  llvm::SetVector<Value> fields;
  for (auto *moved : movedOps) {
    moved->walk([&](Operation *user) {
      for (auto value : user->getOperands()) {
        if (auto blockArg = dyn_cast<BlockArgument>(value)) {
          auto *owner = blockArg.getOwner()->getParentOp();
          if (owner == regionOp || regionOp->isProperAncestor(owner))
            if (!isMoved(owner))
              fields.insert(value);
          continue;
        }
        auto *defOp = value.getDefiningOp();
        if (!regionOp->isProperAncestor(defOp) || isMoved(defOp))
          continue;
        // Floating values are cloned into the receiving block instead of
        // crossing through a field.
        if (!isRealPhase(analysis.getPhase(defOp)))
          continue;
        fields.insert(value);
      }
    });
  }
  return fields;
}

/// After moving content into a receiving block whose arguments correspond
/// 1:1 to `fields`: replace field uses inside the block with the block
/// arguments and clone floating trees that are still referenced from inside
/// the (former) region of `regionOp`.
void Sorter::remapIntoReceivingBlock(Operation *regionOp, Block &receivingBlock,
                                     const llvm::SetVector<Value> &fields) {
  auto *receivingRegion = receivingBlock.getParent();
  for (auto [field, arg] : llvm::zip(fields, receivingBlock.getArguments())) {
    Value fieldValue = field;
    analysis.actualPhase[arg] = analysis.getValuePhase(fieldValue);
    fieldValue.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return receivingRegion->isAncestor(use.getOwner()->getParentRegion());
    });
  }

  // Clone floating trees still referenced from inside the region op.
  receivingBlock.walk([&](Operation *user) {
    for (auto &use : user->getOpOperands()) {
      auto *defOp = use.get().getDefiningOp();
      if (!defOp || !regionOp->isProperAncestor(defOp))
        continue;
      assert(!isRealPhase(analysis.getPhase(defOp)) &&
             hir::isEffectivelyPure(defOp) &&
             "moved op uses non-floating, non-crossing value from region");
      OpBuilder builder(user);
      IRMapping mapping;
      auto clone = cloneFloatingTree(use.get(), builder, mapping);
      clone.getDefiningOp()->walk([&](Operation *cloned) {
        setPhase(cloned, PhaseAnalysis::kFloatPhase);
      });
      use.set(clone);
    }
  });
}

/// Whether an if arm (or the op nested anywhere below it) exits the
/// enclosing loop or function instead of yielding.
static Operation *findExitTerminator(Region &region) {
  Operation *found = nullptr;
  region.walk([&](Operation *op) {
    if (isa<BreakOp, ContinueOp, ReturnOp>(op) &&
        op->hasTrait<OpTrait::IsTerminator>()) {
      found = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

/// Extract the later-phase content of a branch-shaped CF op (uir.if or
/// hir.pick) at phase p: per-arm choices come out of the op as an extra
/// result, and one hir.pick after the op receives all later-phase content.
/// Later-phase results of the op are removed and reconstructed as results
/// of the receiving pick.
LogicalResult Sorter::extractFromBranch(Operation *op, int16_t p) {
  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());
  auto ifOp = dyn_cast<IfOp>(op);
  auto pickOp = dyn_cast<hir::PickOp>(op);
  assert((ifOp || pickOp) && "branch extraction expects uir.if or hir.pick");

  // Gather the arm regions. An else-less if gets its else region
  // materialized so the not-taken tag has a constructor.
  SmallVector<Region *> arms;
  if (ifOp) {
    if (ifOp.getElseRegion().empty())
      ifOp.getElseRegion().emplaceBlock();
    arms.push_back(&ifOp.getThenRegion());
    arms.push_back(&ifOp.getElseRegion());
  } else {
    for (auto &alt : pickOp.getAlternatives())
      arms.push_back(&alt);
  }

  // Arms that exit instead of yielding belong to the enclosing loop's
  // records (break/continue) or to the function-level exit chains at
  // distribution time (return). Neither is extracted here.
  for (auto *arm : arms) {
    if (auto *exit = findExitTerminator(*arm)) {
      if (isa<ReturnOp>(exit))
        return op->emitOpError()
               << "has later-phase content and an early return; "
                  "function-level exit chains are not implemented yet";
      // Break/continue: the enclosing loop's extraction owns this op.
      return success();
    }
  }

  // Materialize an empty-yield terminator in a freshly created else block.
  if (ifOp && arms[1]->front().empty()) {
    auto builder = OpBuilder::atBlockEnd(&arms[1]->front());
    auto yield = YieldOp::create(builder, loc, ValueRange{}, ValueRange{});
    analysis.opPhases[yield] = p;
  }

  // Partition the op's results into kept (at or before p, or never) and
  // later ones. Later results are reconstructed by the receiving pick.
  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : op->getResults()) {
    int16_t phase = analysis.getValuePhase(result);
    if (isRealPhase(phase) && phase > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }

  // Per arm: the moved content, the crossing values, and the operands the
  // old terminator provided for the later results (captured before the
  // terminators are rewritten).
  struct ArmInfo {
    SmallVector<Operation *> moved;
    llvm::SetVector<Value> fields;
    SmallVector<Value> laterYieldOperands;
  };
  SmallVector<ArmInfo> armInfos;
  for (auto [tag, arm] : llvm::enumerate(arms)) {
    ArmInfo info;
    info.moved = collectLaterContent(arm->front(), p);
    info.fields = collectCrossingValues(op, info.moved);

    auto *term = arm->front().getTerminator();
    ValueRange values;
    if (auto uirYield = dyn_cast<YieldOp>(term))
      values = uirYield.getValues();
    else if (auto hirYield = dyn_cast<hir::YieldOp>(term))
      values = hirYield.getOperands();
    for (auto idx : laterIdx)
      if (idx < values.size())
        info.laterYieldOperands.push_back(values[idx]);

    // The later yield operands cross the boundary too: collect their
    // crossing needs as if the yield were moved content.
    for (auto value : info.laterYieldOperands) {
      auto *defOp = value.getDefiningOp();
      if (defOp && op->isProperAncestor(defOp) &&
          !llvm::any_of(info.moved,
                        [&](Operation *m) {
                          return m == defOp || m->isProperAncestor(defOp);
                        }) &&
          isRealPhase(analysis.getPhase(defOp)))
        info.fields.insert(value);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto *owner = blockArg.getOwner()->getParentOp();
        if (owner == op || op->isProperAncestor(owner))
          info.fields.insert(value);
      }
    }
    armInfos.push_back(std::move(info));
  }

  // Construct the per-arm choice and rewrite each terminator to also carry
  // it: kept values plus the choice.
  OpBuilder builder(op);
  Value choiceType;
  if (ifOp) {
    auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
    setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
    choiceType = choiceTypeOp.getResult();
  }

  for (auto [tag, arm] : llvm::enumerate(arms)) {
    auto *term = arm->front().getTerminator();
    OpBuilder termBuilder(term);
    auto choice = hir::ChoiceOp::create(termBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        armInfos[tag].fields.getArrayRef());
    setPhase(choice, p);

    if (auto uirYield = dyn_cast<YieldOp>(term)) {
      SmallVector<Value> values, typeOfValues;
      for (auto idx : keptIdx) {
        values.push_back(uirYield.getValues()[idx]);
        typeOfValues.push_back(uirYield.getTypeOfValues()[idx]);
      }
      values.push_back(choice.getResult());
      typeOfValues.push_back(choiceType);
      auto newYield = YieldOp::create(termBuilder, loc, values, typeOfValues);
      analysis.opPhases[newYield] = p;
      forget(term);
      term->erase();
    } else {
      auto hirYield = cast<hir::YieldOp>(term);
      SmallVector<Value> values;
      for (auto idx : keptIdx)
        values.push_back(hirYield.getOperands()[idx]);
      values.push_back(choice.getResult());
      auto newYield = hir::YieldOp::create(termBuilder, loc, values);
      analysis.opPhases[newYield] = p;
      forget(term);
      term->erase();
    }
  }

  // Rebuild the op with the kept results plus the choice result.
  SmallVector<Type> newResultTypes(keptIdx.size() + 1, anyTy);
  Operation *newOp;
  Value choiceResult;
  if (ifOp) {
    SmallVector<Value> newTypeOperands;
    for (auto idx : keptIdx)
      newTypeOperands.push_back(ifOp.getResultTypes()[idx]);
    newTypeOperands.push_back(choiceType);
    auto newIf = IfOp::create(builder, loc, newResultTypes, ifOp.getCondition(),
                              newTypeOperands);
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());
    newOp = newIf;
  } else {
    auto newPick = hir::PickOp::create(builder, loc, newResultTypes,
                                       pickOp.getChoice(), arms.size());
    for (auto [oldAlt, newAlt] :
         llvm::zip(pickOp.getAlternatives(), newPick.getAlternatives()))
      newAlt.takeBody(oldAlt);
    newOp = newPick;
  }
  analysis.opPhases[newOp] = p;
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newOp->getResult(i)] =
        analysis.getValuePhase(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newOp->getResult(i));
  }
  choiceResult = newOp->getResult(keptIdx.size());
  analysis.actualPhase[choiceResult] = p;

  // Create the receiving pick after the op, one alternative per arm,
  // reconstructing the later results.
  builder.setInsertionPointAfter(newOp);
  SmallVector<Type> pickResultTypes(laterIdx.size(), anyTy);
  auto receivingPick = hir::PickOp::create(builder, loc, pickResultTypes,
                                           choiceResult, arms.size());
  analysis.opPhases[receivingPick] = p + 1;
  for (auto [i, idx] : llvm::enumerate(laterIdx)) {
    analysis.actualPhase[receivingPick.getResult(i)] =
        analysis.getValuePhase(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(receivingPick.getResult(i));
  }

  for (auto [tag, arm] : llvm::enumerate(arms)) {
    auto &info = armInfos[tag];
    auto &altBlock = receivingPick.getAlternatives()[tag].emplaceBlock();
    for (unsigned i = 0; i < info.fields.size(); ++i)
      altBlock.addArgument(anyTy, loc);
    for (auto *moved : info.moved)
      moved->moveBefore(&altBlock, altBlock.end());
    OpBuilder altBuilder(&altBlock, altBlock.end());
    auto altYield =
        hir::YieldOp::create(altBuilder, loc, info.laterYieldOperands);
    analysis.opPhases[altYield] = p + 1;
    remapIntoReceivingBlock(newOp, altBlock, info.fields);
  }

  forget(op);
  op->erase();
  return success();
}

/// Extract the later-phase content of a loop-shaped CF op (uir.loop or
/// hir.replicate) at phase p: a hit list is created before the op, each
/// activation packs a record of the crossing values, and one hir.replicate
/// after the op receives the content.
LogicalResult Sorter::extractFromLoop(Operation *op, int16_t p) {
  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());
  Block *body;
  if (auto loopOp = dyn_cast<LoopOp>(op))
    body = &loopOp.getBody().front();
  else
    body = &cast<hir::ReplicateOp>(op).getBody().front();

  // Later-phase results of a loop are break-value choices, which are not
  // implemented yet.
  for (auto result : op->getResults()) {
    int16_t phase = analysis.getValuePhase(result);
    if (isRealPhase(phase) && phase > p)
      return op->emitOpError()
             << "provides result " << result.getResultNumber() << " at phase "
             << phase << "; break-value choices are not implemented yet";
  }

  auto movedOps = collectLaterContent(*body, p);

  // Partial activations are not implemented yet: all exit-carrying ifs must
  // precede the later-phase content and must not contain any of it, so
  // exactly the full activations reach the single record push at the end of
  // the body.
  bool seenLater = false;
  for (auto &blockOp : *body) {
    bool isLater = llvm::is_contained(movedOps, &blockOp);
    if (isLater)
      seenLater = true;
    if (isLater || blockOp.getNumRegions() == 0)
      continue;
    if (hasLaterContent(&blockOp, p))
      return blockOp.emitOpError()
             << "contains later-phase content on a partial activation path; "
                "not implemented yet";
    if (findExitTerminator(*blockOp.getRegions().begin()) ||
        (blockOp.getNumRegions() > 1 &&
         findExitTerminator(*std::next(blockOp.getRegions().begin())))) {
      if (seenLater)
        return blockOp.emitOpError()
               << "loop exit after later-phase content requires partial "
                  "activation records; not implemented yet";
    }
  }

  auto fields = collectCrossingValues(op, movedOps);

  // Hit list before the op, record pack and push before the body
  // terminator: only full activations reach it.
  OpBuilder builder(op);
  auto list = hir::OpaqueListCreateOp::create(builder, loc, anyTy);
  setPhase(list, p);

  auto *term = body->getTerminator();
  OpBuilder termBuilder(term);
  auto record =
      hir::OpaquePackOp::create(termBuilder, loc, anyTy, fields.getArrayRef());
  setPhase(record, p);
  auto push = hir::OpaqueListPushOp::create(termBuilder, loc, list.getResult(),
                                            record.getResult());
  setPhase(push, p);

  // Receiving replicate after the op.
  builder.setInsertionPointAfter(op);
  auto replicate = hir::ReplicateOp::create(builder, loc, list.getResult());
  analysis.opPhases[replicate] = p + 1;
  auto &repBlock = replicate.getBody().emplaceBlock();
  for (unsigned i = 0; i < fields.size(); ++i)
    repBlock.addArgument(anyTy, loc);
  for (auto *moved : movedOps)
    moved->moveBefore(&repBlock, repBlock.end());
  OpBuilder repBuilder(&repBlock, repBlock.end());
  auto repYield = hir::YieldOp::create(repBuilder, loc, ValueRange{});
  analysis.opPhases[repYield] = p + 1;
  remapIntoReceivingBlock(op, repBlock, fields);

  return success();
}

/// Extract all later-phase content out of phase-p CF regions, innermost
/// first. Receiving constructs created here sit after their CF op and are
/// themselves CF regions for the next boundary.
LogicalResult Sorter::extractBoundary(int16_t p) {
  SmallVector<Operation *> cfOps;
  funcOp.getBody().walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<IfOp, LoopOp, hir::ReplicateOp, hir::PickOp>(op))
      if (auto ph = realPhaseOf(op); ph && *ph == p)
        cfOps.push_back(op);
  });

  for (auto *op : cfOps) {
    if (!hasLaterContent(op, p))
      continue;
    if (isa<IfOp, hir::PickOp>(op)) {
      if (failed(extractFromBranch(op, p)))
        return failure();
    } else {
      if (failed(extractFromLoop(op, p)))
        return failure();
    }
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult Sorter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Sorting phases in @" << funcOp.getSymName()
                          << "\n");
  if (failed(dissolveExprsAndPins()))
    return failure();
  if (failed(decomposeCalls()))
    return failure();

  auto [minPhase, maxPhase] = computePhaseRange();
  for (int16_t boundary = minPhase; boundary < maxPhase; ++boundary) {
    LLVM_DEBUG(llvm::dbgs() << "  boundary " << boundary << "\n");
    hoistBoundary(boundary);
    if (failed(extractBoundary(boundary)))
      return failure();
  }
  return success();
}

LogicalResult silicon::uir::sortPhases(FuncOp funcOp, PhaseAnalysis &analysis,
                                       SymbolTable &symbolTable) {
  return Sorter(funcOp, analysis, symbolTable).run();
}
