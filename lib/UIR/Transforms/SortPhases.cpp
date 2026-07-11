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
//    An expr whose body ends in `uir.return` (a later-phase early exit such
//    as `dyn { return x }`) cannot dissolve in place: its terminator has no
//    mid-block position. Such an expr stays intact as a grouping carrier,
//    travels wholesale during extraction, and dissolves once it sits at the
//    tail of a receiving block, where the return becomes the block's
//    terminator: the exit reconstructs itself in the receiving phase.
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
//      **Partial activations**: when a loop body interleaves later-phase
//      content with early exits (break/continue at the loop's phase inside
//      branch arms), activations diverge in how much of that content they
//      instantiate. The body decomposes into content segments separated by
//      exit-carrying divergence points, and the record ends in a nested
//      exit-path choice: each exit terminator packs and pushes its own
//      record in place — the shared prefix segment, then one `continued`
//      link per divergence it fell past (carrying that segment's samples),
//      ending in its own exit tag (carrying the samples of its arm's
//      content). The receiving replicate mirrors this shape with nested
//      picks: each divergence becomes a pick whose exit alternatives hold
//      the arm content and whose `continued` alternative holds the next
//      segment and the next pick.
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
/// One exit path of a divergence point: a break/continue terminator at the
/// loop's phase, together with the later-phase content on its path and the
/// crossing values that content samples; both are empty for paths that run
/// no later-phase content.
struct ExitPath {
  Operation *terminator;
  SmallVector<Operation *> content;
  llvm::SetVector<Value> fields;
};

/// An exit-carrying op in a loop body with later-phase content: activations
/// diverge here in how much of that content they instantiate, and every
/// exit path contributes one chain link to the activation's record.
struct Divergence {
  SmallVector<ExitPath> exits;
};

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

  bool hasLaterContent(Operation *root, int16_t p);
  LogicalResult extractBreakValues(Operation *&op, int16_t p);
  void collectExitTerminators(Region &region, int16_t p,
                              SmallVectorImpl<Operation *> &exits);
  Operation *findExitTerminator(Region &region, int16_t p);
  SmallVector<Operation *> collectLaterContent(Block &block, int16_t p);
  LogicalResult dissolveTailReturnExpr(Block &block);
  LogicalResult
  scanLoopBody(Operation *op, Block &body, int16_t p,
               SmallVectorImpl<SmallVector<Operation *>> &segments,
               SmallVectorImpl<Divergence> &divergences);
  llvm::SetVector<Value> collectCrossingValues(Operation *regionOp,
                                               ArrayRef<Operation *> userOps,
                                               ArrayRef<Operation *> movedOps);
  void remapFields(Block &receivingBlock, const llvm::SetVector<Value> &fields);
  void cloneFloatingUses(Operation *regionOp, Block &rootBlock);

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
    if (!yieldOp) {
      // Return-terminated exprs stay intact as grouping carriers until
      // extraction places them at the tail of a receiving block.
      if (isa<ReturnOp>(block.getTerminator())) {
        if (exprOp.getNumResults() != 0)
          return exprOp.emitOpError()
                 << "with results must not be terminated by a return";
        continue;
      }
      return exprOp.emitOpError()
             << "cannot be dissolved: body is terminated by "
             << block.getTerminator()->getName() << " instead of a yield";
    }
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
    entries.push_back({static_cast<int16_t>(callOpPhase + entryPhase), name});
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

/// Collect the values defined inside `regionOp` that the `userOps` use and
/// that are not themselves produced by any of the `movedOps`: these cross
/// the phase boundary and become record fields / choice payloads. Values
/// defined outside the region op remain direct uses (the receiving
/// construct sits right after the op, so they dominate it), and floating
/// values are cloned instead. `movedOps` lists everything that leaves the
/// region — with partial activations that is more than the `userOps` whose
/// uses are being collected here.
llvm::SetVector<Value>
Sorter::collectCrossingValues(Operation *regionOp,
                              ArrayRef<Operation *> userOps,
                              ArrayRef<Operation *> movedOps) {
  auto isMoved = [&](Operation *op) {
    return llvm::any_of(movedOps, [&](Operation *m) {
      return m == op || m->isProperAncestor(op);
    });
  };

  llvm::SetVector<Value> fields;
  for (auto *moved : userOps) {
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

/// After moving content into a receiving block whose leading arguments
/// correspond 1:1 to `fields`: replace field uses inside the block (and any
/// nested receiving blocks) with the block arguments. With nested receiving
/// structures, all levels must be remapped before floating trees are cloned
/// via `cloneFloatingUses`.
void Sorter::remapFields(Block &receivingBlock,
                         const llvm::SetVector<Value> &fields) {
  auto *receivingRegion = receivingBlock.getParent();
  for (auto [field, arg] : llvm::zip(fields, receivingBlock.getArguments())) {
    Value fieldValue = field;
    analysis.actualPhase[arg] = analysis.getValuePhase(fieldValue);
    fieldValue.replaceUsesWithIf(arg, [&](OpOperand &use) {
      return receivingRegion->isAncestor(use.getOwner()->getParentRegion());
    });
  }
}

/// Clone floating trees that moved content still references from inside the
/// (former) region of `regionOp`. Must run after every receiving block's
/// fields have been remapped; any other remaining use of a region-interior
/// value is a causality violation the analysis must have rejected.
void Sorter::cloneFloatingUses(Operation *regionOp, Block &rootBlock) {
  rootBlock.walk([&](Operation *user) {
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

/// Dissolve a return-terminated expr carrier sitting at the tail of a
/// receiving block: its ops splice into the block and its return replaces
/// the block's empty yield terminator. A carrier anywhere else in the block
/// is a hard error — content after a definite later-phase return is dead
/// and must not reach sorting.
LogicalResult Sorter::dissolveTailReturnExpr(Block &block) {
  SmallVector<ExprOp> carriers;
  for (auto exprOp : block.getOps<ExprOp>())
    if (isa<ReturnOp>(exprOp.getBody().front().getTerminator()))
      carriers.push_back(exprOp);

  for (auto exprOp : carriers) {
    auto &exprBlock = exprOp.getBody().front();
    auto *blockTerm = block.getTerminator();
    if (exprOp->getNextNode() != blockTerm)
      return exprOp.emitOpError()
             << "definitely returns but is followed by more content in the "
                "receiving block";
    auto yieldOp = dyn_cast<hir::YieldOp>(blockTerm);
    if (!yieldOp || !yieldOp.getOperands().empty())
      return exprOp.emitOpError()
             << "definitely returns but the receiving block must yield "
                "values";

    block.getOperations().splice(exprOp->getIterator(),
                                 exprBlock.getOperations());
    forget(yieldOp);
    yieldOp.erase();
    forget(exprOp);
    exprOp.erase();
  }
  return success();
}

/// Collect all terminators at phase <= p inside the region that exit out
/// of the region instead of yielding, in walk order. Break/continue inside
/// loops (including the queried region itself when it belongs to a loop)
/// target those loops and do not exit; returns exit the function from any
/// depth. Later-phase returns riding inside expr carriers are content, not
/// exits of this region.
void Sorter::collectExitTerminators(Region &region, int16_t p,
                                    SmallVectorImpl<Operation *> &exits) {
  std::function<void(Region &, bool)> visit = [&](Region &r, bool loopScoped) {
    for (auto &block : r) {
      for (auto &op : block) {
        bool isExit =
            isa<ReturnOp>(op) || (!loopScoped && isa<BreakOp, ContinueOp>(op));
        if (isExit) {
          auto it = analysis.opPhases.find(&op);
          if (it != analysis.opPhases.end() && isRealPhase(it->second) &&
              it->second <= p)
            exits.push_back(&op);
        }
        for (auto &inner : op.getRegions())
          visit(inner, loopScoped || isa<LoopOp>(op));
      }
    }
  };
  visit(region, isa<LoopOp>(region.getParentOp()));
}

/// The first exit terminator at phase <= p inside the region, or null if
/// every path through the region yields.
Operation *Sorter::findExitTerminator(Region &region, int16_t p) {
  SmallVector<Operation *> exits;
  collectExitTerminators(region, p, exits);
  return exits.empty() ? nullptr : exits.front();
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

  // Arms that exit instead of yielding belong to the enclosing loop's
  // records (break/continue) or to the function-level exit chains at
  // distribution time (return). Neither is extracted here.
  for (auto &arm : op->getRegions()) {
    if (auto *exit = findExitTerminator(arm, p)) {
      if (isa<ReturnOp>(exit))
        return op->emitOpError()
               << "has later-phase content and an early return; "
                  "function-level exit chains are not implemented yet";
      // Break/continue: the enclosing loop's extraction owns this op.
      return success();
    }
  }

  // Gather the arm regions. An else-less if gets its else region
  // materialized with an empty yield so the not-taken tag has a
  // constructor.
  SmallVector<Region *> arms;
  if (ifOp) {
    if (ifOp.getElseRegion().empty()) {
      auto &elseBlock = ifOp.getElseRegion().emplaceBlock();
      auto builder = OpBuilder::atBlockEnd(&elseBlock);
      auto yield = YieldOp::create(builder, loc, ValueRange{}, ValueRange{});
      analysis.opPhases[yield] = p;
    }
    arms.push_back(&ifOp.getThenRegion());
    arms.push_back(&ifOp.getElseRegion());
  } else {
    for (auto &alt : pickOp.getAlternatives())
      arms.push_back(&alt);
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
    info.fields = collectCrossingValues(op, info.moved, info.moved);

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
    remapFields(altBlock, info.fields);
    cloneFloatingUses(newOp, altBlock);
    if (failed(dissolveTailReturnExpr(altBlock)))
      return failure();
  }

  forget(op);
  op->erase();
  return success();
}

/// Extract later-phase results of a loop as break-value choices. Each break
/// site constructs an hir.choice (tag = site index, payload = the crossing
/// values its value computation needs); the loop is rebuilt with the kept
/// results plus the choice; and a pick after the loop reconstructs the
/// later-phase values, with each site's value slice moved into its
/// alternative. The exit content runs once, after all records, so the pick
/// sits after any replicate the loop's record extraction creates.
LogicalResult Sorter::extractBreakValues(Operation *&op, int16_t p) {
  auto loopOp = dyn_cast<LoopOp>(op);
  if (!loopOp)
    return success(); // hir.replicate has no results

  SmallVector<unsigned> keptIdx, laterIdx;
  for (auto result : op->getResults()) {
    int16_t phase = analysis.getValuePhase(result);
    if (isRealPhase(phase) && phase > p)
      laterIdx.push_back(result.getResultNumber());
    else
      keptIdx.push_back(result.getResultNumber());
  }
  if (laterIdx.empty())
    return success();

  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());

  // Collect this loop's break sites, not descending into nested loops.
  SmallVector<BreakOp> breaks;
  std::function<void(Block &)> collectBreaks = [&](Block &block) {
    for (auto &blockOp : block) {
      if (isa<LoopOp>(&blockOp))
        continue;
      if (auto breakOp = dyn_cast<BreakOp>(&blockOp)) {
        breaks.push_back(breakOp);
        continue;
      }
      for (auto &region : blockOp.getRegions())
        for (auto &inner : region)
          collectBreaks(inner);
    }
  };
  collectBreaks(loopOp.getBody().front());

  // Per break site: the later-phase values it provides, the transitive
  // later-phase ops computing them (the value slice, which moves into the
  // pick alternative), and the crossing values that become the choice
  // payload.
  struct SiteInfo {
    SmallVector<Value> laterValues;
    SmallVector<Operation *> slice;
    llvm::SetVector<Value> fields;
  };
  SmallVector<SiteInfo> sites;
  DenseSet<Operation *> allSliceOps;
  for (auto breakOp : breaks) {
    SiteInfo info;
    for (auto idx : laterIdx) {
      if (idx >= breakOp.getValues().size())
        return breakOp.emitOpError()
               << "does not provide the loop's later-phase result " << idx;
      info.laterValues.push_back(breakOp.getValues()[idx]);
    }

    // Backward closure over later-phase ops inside the loop.
    DenseSet<Operation *> sliceSet;
    SmallVector<Value> worklist(info.laterValues);
    while (!worklist.empty()) {
      auto *defOp = worklist.pop_back_val().getDefiningOp();
      if (!defOp || !op->isProperAncestor(defOp))
        continue;
      if (auto ph = realPhaseOf(defOp); !ph || *ph <= p)
        continue;
      if (!sliceSet.insert(defOp).second)
        continue;
      if (!allSliceOps.insert(defOp).second)
        return defOp->emitOpError()
               << "feeds the break values of multiple loop exits; not "
                  "implemented yet";
      defOp->walk([&](Operation *inner) {
        for (auto operand : inner->getOperands())
          worklist.push_back(operand);
      });
    }

    // Slice ops in program order, outermost units only; reject slices
    // entangled with content that stays behind.
    loopOp.getBody().walk<WalkOrder::PreOrder>([&](Operation *walked) {
      if (sliceSet.contains(walked)) {
        info.slice.push_back(walked);
        return WalkResult::skip();
      }
      return WalkResult::advance();
    });
    for (auto *sliceOp : info.slice)
      for (auto *user : sliceOp->getUsers())
        if (user != breakOp && !sliceSet.contains(user) &&
            llvm::none_of(sliceSet,
                          [&](Operation *s) { return s->isAncestor(user); }))
          return sliceOp->emitOpError()
                 << "is shared between a break value and other later-phase "
                    "content; not implemented yet";

    info.fields = collectCrossingValues(op, info.slice, info.slice);
    for (auto value : info.laterValues) {
      auto *defOp = value.getDefiningOp();
      if (defOp && op->isProperAncestor(defOp) && !sliceSet.contains(defOp) &&
          isRealPhase(analysis.getPhase(defOp)))
        info.fields.insert(value);
      if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        auto *owner = blockArg.getOwner()->getParentOp();
        if (owner == op || op->isProperAncestor(owner))
          info.fields.insert(value);
      }
    }
    sites.push_back(std::move(info));
  }

  // Construct the per-site choice and rewrite each break to carry it in
  // place of the later-phase values.
  OpBuilder builder(op);
  auto choiceTypeOp = hir::ChoiceTypeOp::create(builder, loc);
  setPhase(choiceTypeOp, PhaseAnalysis::kFloatPhase);
  Value choiceType = choiceTypeOp.getResult();

  for (auto [tag, breakOp] : llvm::enumerate(breaks)) {
    OpBuilder breakBuilder(breakOp);
    auto choice = hir::ChoiceOp::create(breakBuilder, loc, anyTy,
                                        static_cast<uint32_t>(tag),
                                        sites[tag].fields.getArrayRef());
    setPhase(choice, p);
    SmallVector<Value> values, typeOfValues;
    for (auto idx : keptIdx) {
      values.push_back(breakOp.getValues()[idx]);
      typeOfValues.push_back(breakOp.getTypeOfValues()[idx]);
    }
    values.push_back(choice.getResult());
    typeOfValues.push_back(choiceType);
    auto newBreak = BreakOp::create(breakBuilder, loc, values, typeOfValues);
    analysis.opPhases[newBreak] = p;
    forget(breakOp);
    breakOp.erase();
  }

  // Rebuild the loop with the kept results plus the choice result.
  SmallVector<Type> newResultTypes(keptIdx.size() + 1, anyTy);
  SmallVector<Value> newTypeOperands;
  for (auto idx : keptIdx)
    newTypeOperands.push_back(loopOp.getResultTypes()[idx]);
  newTypeOperands.push_back(choiceType);
  auto newLoop = LoopOp::create(builder, loc, newResultTypes, loopOp.getInits(),
                                loopOp.getInitTypes(), newTypeOperands);
  newLoop.getBody().takeBody(loopOp.getBody());
  analysis.opPhases[newLoop] = p;
  for (auto [i, idx] : llvm::enumerate(keptIdx)) {
    analysis.actualPhase[newLoop.getResult(i)] =
        analysis.getValuePhase(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(newLoop.getResult(i));
  }
  Value choiceResult = newLoop.getResult(keptIdx.size());
  analysis.actualPhase[choiceResult] = p;

  // Create the receiving pick right after the loop. Record extraction
  // below inserts its replicate between the loop and the pick, so the exit
  // content runs after all records.
  builder.setInsertionPointAfter(newLoop);
  SmallVector<Type> pickResultTypes(laterIdx.size(), anyTy);
  auto pick = hir::PickOp::create(builder, loc, pickResultTypes, choiceResult,
                                  sites.size());
  analysis.opPhases[pick] = p + 1;
  for (auto [i, idx] : llvm::enumerate(laterIdx)) {
    analysis.actualPhase[pick.getResult(i)] =
        analysis.getValuePhase(op->getResult(idx));
    op->getResult(idx).replaceAllUsesWith(pick.getResult(i));
  }

  for (auto [tag, site] : llvm::enumerate(sites)) {
    auto &altBlock = pick.getAlternatives()[tag].emplaceBlock();
    for (unsigned i = 0; i < site.fields.size(); ++i)
      altBlock.addArgument(anyTy, loc);
    for (auto *sliceOp : site.slice)
      sliceOp->moveBefore(&altBlock, altBlock.end());
    OpBuilder altBuilder(&altBlock, altBlock.end());
    auto altYield = hir::YieldOp::create(altBuilder, loc, site.laterValues);
    analysis.opPhases[altYield] = p + 1;
    remapFields(altBlock, site.fields);
    cloneFloatingUses(newLoop, altBlock);
    if (failed(dissolveTailReturnExpr(altBlock)))
      return failure();
  }

  forget(op);
  op->erase();
  op = newLoop;
  return success();
}

/// Decompose a loop body's top level into content segments separated by
/// exit-carrying divergence points. `segments` holds the later-phase ops
/// between consecutive divergences in program order (one more entry than
/// `divergences`); each divergence records its exit terminators and the
/// later-phase content on their paths. Exit paths that run no later-phase
/// content get an empty chain link, so every activation pushes exactly one
/// record and the hit list length always equals the activation count.
LogicalResult
Sorter::scanLoopBody(Operation *op, Block &body, int16_t p,
                     SmallVectorImpl<SmallVector<Operation *>> &segments,
                     SmallVectorImpl<Divergence> &divergences) {
  segments.emplace_back();
  for (auto &blockOp : body) {
    if (&blockOp == body.getTerminator())
      break;
    if (auto ph = realPhaseOf(&blockOp); ph && *ph > p) {
      segments.back().push_back(&blockOp);
      continue;
    }
    if (blockOp.getNumRegions() == 0)
      continue;

    SmallVector<Operation *> exitTerms;
    for (auto &region : blockOp.getRegions())
      collectExitTerminators(region, p, exitTerms);
    if (exitTerms.empty()) {
      // Non-exit CF at phase p was extracted by the innermost-first walk
      // already; leftover later-phase content here is a compiler bug.
      if (hasLaterContent(&blockOp, p))
        return blockOp.emitOpError()
               << "contains later-phase content that inner extraction did "
                  "not consume";
      continue;
    }

    // A return under a loop with later-phase content is a function-level
    // exit chain no matter how it is nested; those belong to distribution.
    for (auto *term : exitTerms)
      if (isa<ReturnOp>(term))
        return term->emitOpError()
               << "exits the function under a loop with later-phase "
                  "content; function-level exit chains are not implemented "
                  "yet";

    // Results whose phase was never demanded have no consumers and need no
    // reconstruction; only demanded later-phase results are unimplemented.
    for (auto result : blockOp.getResults()) {
      auto it = analysis.actualPhase.find(result);
      if (it != analysis.actualPhase.end() && isRealPhase(it->second) &&
          it->second > p)
        return blockOp.emitOpError() << "carries a loop exit and has "
                                        "later-phase results; not "
                                        "implemented yet";
    }

    Divergence divergence;
    if (!hasLaterContent(&blockOp, p)) {
      // No later-phase content anywhere in the divergence: every exit path
      // contributes an empty chain link built at its terminator, no matter
      // how deeply the terminator is nested.
      for (auto *term : exitTerms) {
        ExitPath exit;
        exit.terminator = term;
        divergence.exits.push_back(std::move(exit));
      }
    } else {
      // Later-phase content inside the divergence must be attributed to
      // the paths that run it; only the flat form (a branch whose exit
      // arms are straight-line up to their terminator) is implemented.
      auto ifOp = dyn_cast<IfOp>(&blockOp);
      if (!ifOp)
        return blockOp.emitOpError()
               << "carries a loop exit but is not a branch; cannot build "
                  "partial activation records";
      // Classify each arm as an exit path or a rejoining path.
      for (auto &region : ifOp->getRegions()) {
        if (region.empty())
          continue;
        auto &armBlock = region.front();
        auto *term = armBlock.getTerminator();
        auto termPhase = realPhaseOf(term);
        bool isExitTerm =
            termPhase && *termPhase <= p && isa<BreakOp, ContinueOp>(term);
        if (isExitTerm) {
          ExitPath exit;
          exit.terminator = term;
          exit.content = collectLaterContent(armBlock, p);
          // The arm must be straight-line up to its exit: nested divergence
          // (further exits or later-phase content inside arm-interior CF)
          // is not implemented.
          for (auto &armOp : armBlock) {
            if (&armOp == term || llvm::is_contained(exit.content, &armOp))
              continue;
            bool nestedExit = false;
            for (auto &inner : armOp.getRegions())
              nestedExit |= findExitTerminator(inner, p) != nullptr;
            if (nestedExit || hasLaterContent(&armOp, p))
              return armOp.emitOpError()
                     << "diverges inside an exit-carrying arm; nested "
                        "divergence is not implemented yet";
          }
          divergence.exits.push_back(std::move(exit));
        } else {
          // Rejoining arm: its content would need a sub-choice covering the
          // arm-local divergence, which is not implemented.
          if (findExitTerminator(region, p))
            return blockOp.emitOpError()
                   << "has a rejoining arm containing a nested exit; nested "
                      "divergence is not implemented yet";
          bool armHasContent = !collectLaterContent(armBlock, p).empty() ||
                               llvm::any_of(armBlock, [&](Operation &armOp) {
                                 return hasLaterContent(&armOp, p);
                               });
          if (armHasContent)
            return blockOp.emitOpError()
                   << "has later-phase content on a rejoining arm of an "
                      "exit-carrying branch; not implemented yet";
        }
      }
    }

    divergences.push_back(std::move(divergence));
    segments.emplace_back();
  }
  return success();
}

/// Extract the later-phase content of a loop-shaped CF op (uir.loop or
/// hir.replicate) at phase p: a hit list is created before the op, each
/// activation packs a record of the crossing values, and one hir.replicate
/// after the op receives the content. Every activation pushes exactly one
/// record, built in place at the terminator that ends it: the record chains
/// a `continued` link per divergence the activation fell past and ends in
/// the taken exit path's choice, whose payload is empty when that path ran
/// no later-phase content. The replicate reconstructs the divergences with
/// nested picks.
LogicalResult Sorter::extractFromLoop(Operation *op, int16_t p) {
  auto loc = op->getLoc();
  auto anyTy = hir::AnyType::get(op->getContext());

  // Later-phase results are break-value choices: extract them first, so
  // their value slices are out of the way of the record analysis below.
  // The loop op may be rebuilt in the process.
  if (failed(extractBreakValues(op, p)))
    return failure();
  Block *body;
  if (auto loopOp = dyn_cast<LoopOp>(op))
    body = &loopOp.getBody().front();
  else
    body = &cast<hir::ReplicateOp>(op).getBody().front();
  if (!hasLaterContent(op, p))
    return success();

  SmallVector<SmallVector<Operation *>> segments;
  SmallVector<Divergence> divergences;
  if (failed(scanLoopBody(op, *body, p, segments, divergences)))
    return failure();
  size_t numDivs = divergences.size();

  // Collect the crossing values per segment and per exit path. Everything
  // that leaves the region counts as moved for the crossing analysis. A
  // segment's fields are deduplicated against all enclosing segments (their
  // receiving-block arguments dominate the nested levels); an exit path's
  // fields against the segments enclosing its pick.
  SmallVector<Operation *> allMoved;
  for (auto &segment : segments)
    allMoved.append(segment);
  for (auto &divergence : divergences)
    for (auto &exit : divergence.exits)
      allMoved.append(exit.content);

  SmallVector<llvm::SetVector<Value>> segFields(segments.size());
  llvm::SetVector<Value> prefixFields;
  for (auto [k, segment] : llvm::enumerate(segments)) {
    segFields[k] = collectCrossingValues(op, segment, allMoved);
    segFields[k].set_subtract(prefixFields);
    prefixFields.set_union(segFields[k]);
    if (k < numDivs) {
      for (auto &exit : divergences[k].exits) {
        exit.fields = collectCrossingValues(op, exit.content, allMoved);
        exit.fields.set_subtract(prefixFields);
      }
    }
  }

  // Hit list before the op.
  OpBuilder builder(op);
  auto list = hir::OpaqueListCreateOp::create(builder, loc, anyTy);
  setPhase(list, p);

  // Build one record push per exit path, in place at its terminator. The
  // record is the shared prefix segment plus, for each divergence the path
  // fell past, a `continued` link carrying that segment's samples, ending
  // in the path's own exit tag. The `continued` tag of each divergence is
  // one past its exit tags.
  auto buildRecordPush = [&](Operation *beforeTerm, int exitLevel, uint32_t tag,
                             ArrayRef<Value> payload) {
    OpBuilder b(beforeTerm);
    Value chain;
    if (exitLevel >= 0) {
      auto choice = hir::ChoiceOp::create(b, loc, anyTy, tag, payload);
      setPhase(choice, p);
      chain = choice.getResult();
      for (int i = exitLevel - 1; i >= 0; --i) {
        SmallVector<Value> link(segFields[i + 1].getArrayRef());
        link.push_back(chain);
        auto continued = hir::ChoiceOp::create(
            b, loc, anyTy, static_cast<uint32_t>(divergences[i].exits.size()),
            link);
        setPhase(continued, p);
        chain = continued.getResult();
      }
    }
    SmallVector<Value> recordFields(segFields[0].getArrayRef());
    if (chain)
      recordFields.push_back(chain);
    auto record = hir::OpaquePackOp::create(b, loc, anyTy, recordFields);
    setPhase(record, p);
    auto push = hir::OpaqueListPushOp::create(b, loc, list.getResult(),
                                              record.getResult());
    setPhase(push, p);
  };

  for (auto [k, divergence] : llvm::enumerate(divergences))
    for (auto [tag, exit] : llvm::enumerate(divergence.exits))
      buildRecordPush(exit.terminator, k, tag, exit.fields.getArrayRef());

  // A break/continue body terminator ends a full activation just like a
  // replicate body's yield; the record with the full chain is pushed there.
  auto *bodyTerm = body->getTerminator();
  if (isa<BreakOp, ContinueOp, hir::YieldOp>(bodyTerm)) {
    if (numDivs == 0)
      buildRecordPush(bodyTerm, -1, 0, {});
    else
      buildRecordPush(bodyTerm, numDivs - 1,
                      divergences[numDivs - 1].exits.size(),
                      segFields[numDivs].getArrayRef());
  } else if (isa<ReturnOp>(bodyTerm)) {
    return bodyTerm->emitOpError()
           << "terminates a loop body with later-phase content; "
              "function-level exit chains are not implemented yet";
  } else if (!isa<UnreachableOp>(bodyTerm)) {
    return bodyTerm->emitOpError()
           << "unsupported loop body terminator for record extraction";
  }

  // Receiving replicate after the op, mirroring the record shape: one level
  // per segment, connected by picks over the chain links. Level k holds
  // segment k's content; its pick's exit alternatives hold the arm content,
  // and the `continued` alternative is level k+1.
  builder.setInsertionPointAfter(op);
  auto replicate = hir::ReplicateOp::create(builder, loc, list.getResult());
  analysis.opPhases[replicate] = p + 1;

  SmallVector<std::pair<Block *, const llvm::SetVector<Value> *>> remapBlocks;
  Block *levelBlock = &replicate.getBody().emplaceBlock();
  for (auto [k, segment] : llvm::enumerate(segments)) {
    bool hasChain = k < numDivs;
    for (unsigned i = 0; i < segFields[k].size(); ++i)
      levelBlock->addArgument(anyTy, loc);
    Value chainArg;
    if (hasChain) {
      chainArg = levelBlock->addArgument(anyTy, loc);
      analysis.actualPhase[chainArg] = p;
    }
    remapBlocks.push_back({levelBlock, &segFields[k]});
    for (auto *moved : segment)
      moved->moveBefore(levelBlock, levelBlock->end());

    Block *nextLevel = nullptr;
    if (hasChain) {
      auto &divergence = divergences[k];
      OpBuilder levelBuilder(levelBlock, levelBlock->end());
      auto pick = hir::PickOp::create(levelBuilder, loc, TypeRange{}, chainArg,
                                      divergence.exits.size() + 1);
      analysis.opPhases[pick] = p + 1;
      for (auto [tag, exit] : llvm::enumerate(divergence.exits)) {
        auto &altBlock = pick.getAlternatives()[tag].emplaceBlock();
        for (unsigned i = 0; i < exit.fields.size(); ++i)
          altBlock.addArgument(anyTy, loc);
        remapBlocks.push_back({&altBlock, &exit.fields});
        for (auto *moved : exit.content)
          moved->moveBefore(&altBlock, altBlock.end());
        OpBuilder altBuilder(&altBlock, altBlock.end());
        auto altYield = hir::YieldOp::create(altBuilder, loc, ValueRange{});
        analysis.opPhases[altYield] = p + 1;
      }
      nextLevel =
          &pick.getAlternatives()[divergence.exits.size()].emplaceBlock();
    }
    OpBuilder levelBuilder(levelBlock, levelBlock->end());
    auto levelYield = hir::YieldOp::create(levelBuilder, loc, ValueRange{});
    analysis.opPhases[levelYield] = p + 1;
    levelBlock = nextLevel;
  }

  // Remap fields outermost-first so that deduplicated inner uses resolve to
  // the enclosing level's arguments, then clone floating trees once over
  // the whole structure and dissolve return carriers at block tails.
  for (auto [block, fields] : remapBlocks)
    remapFields(*block, *fields);
  cloneFloatingUses(op, replicate.getBody().front());
  for (auto [block, fields] : remapBlocks)
    if (failed(dissolveTailReturnExpr(*block)))
      return failure();

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
