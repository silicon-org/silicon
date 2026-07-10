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
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/SortPhases.h"
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
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult Sorter::run() {
  LLVM_DEBUG(llvm::dbgs() << "Sorting phases in @" << funcOp.getSymName()
                          << "\n");
  if (failed(dissolveExprsAndPins()))
    return failure();
  if (failed(decomposeCalls()))
    return failure();
  return success();
}

LogicalResult silicon::uir::sortPhases(FuncOp funcOp, PhaseAnalysis &analysis,
                                       SymbolTable &symbolTable) {
  return Sorter(funcOp, analysis, symbolTable).run();
}
