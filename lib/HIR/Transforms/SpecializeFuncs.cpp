//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Base/Attributes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "specialize-funcs"

namespace silicon {
namespace hir {
#define GEN_PASS_DEF_SPECIALIZEFUNCSPASS
#include "silicon/HIR/Passes.h.inc"
} // namespace hir
} // namespace silicon

//===----------------------------------------------------------------------===//
// SpecializeFuncsPass
//
// Performs two kinds of specialization:
//
// 1. Multiphase chaining: when the first sub-function of a multiphase_func
//    has been evaluated (mir.evaluated_func exists), chain its results into
//    the next sub-function by replacing the opaque context arg with
//    hir.mir_constant ops. Dissolve the multiphase_func when only one
//    sub-function remains.
//
// 2. Transitive call-site specialization: after chaining, walk the
//    specialized function's body for hir.call ops that pass hir.mir_constant
//    opaque values as arguments. Clone the callee, inline the constant,
//    expand the opaque_unpack, and update the call.
//===----------------------------------------------------------------------===//

namespace {
struct SpecializeFuncsPass
    : public silicon::hir::impl::SpecializeFuncsPassBase<SpecializeFuncsPass> {
  void runOnOperation() override;

  /// Expand an opaque context arg in a function: replace the last block arg
  /// with individual hir.mir_constant ops derived from the opaque attribute.
  void expandOpaqueContext(hir::FuncOp func, base::OpaqueAttr opaqueAttr);

  /// Transitively specialize callees that receive hir.mir_constant opaque
  /// arguments.
  void transitiveSpecialize(hir::FuncOp func, SymbolTable &symbolTable);

  unsigned specCounter = 0;

  /// Cache of specializations keyed on (original callee symbol, OpaqueAttr).
  /// Avoids creating redundant clones when multiple call sites pass the same
  /// opaque constant to the same callee.
  DenseMap<std::pair<StringAttr, Attribute>, StringAttr> specCache;
};
} // namespace

/// Replace the last block arg (the opaque context) with individual
/// hir.mir_constant ops for each element of the opaque attribute. Erases the
/// opaque_unpack consuming the context arg.
void SpecializeFuncsPass::expandOpaqueContext(hir::FuncOp func,
                                              base::OpaqueAttr opaqueAttr) {
  auto &block = func.getBody().front();
  auto ctxArg = block.getArgument(block.getNumArguments() - 1);

  // Find the opaque_unpack consuming the context arg directly.
  hir::OpaqueUnpackOp unpackOp;
  for (auto *user : ctxArg.getUsers()) {
    if (auto u = dyn_cast<hir::OpaqueUnpackOp>(user)) {
      unpackOp = u;
      break;
    }
  }

  if (!unpackOp) {
    LLVM_DEBUG(llvm::dbgs() << "  No opaque_unpack found, inserting "
                               "hir.mir_constant and relying on "
                               "canonicalization\n");
    // Insert a mir_constant for the whole opaque and let canonicalization
    // handle the unpack expansion.
    OpBuilder builder(&block, block.begin());
    auto constOp = hir::MIRConstantOp::create(builder, ctxArg.getLoc(),
                                              cast<TypedAttr>(opaqueAttr));
    ctxArg.replaceAllUsesWith(constOp.getResult());
  } else {
    // Replace each unpack result with an individual mir_constant.
    OpBuilder builder(unpackOp);
    for (auto [result, elem] :
         llvm::zip(unpackOp.getResults(), opaqueAttr.getElements())) {
      auto constOp = hir::MIRConstantOp::create(builder, unpackOp.getLoc(),
                                                cast<TypedAttr>(elem));
      result.replaceAllUsesWith(constOp.getResult());
    }
    unpackOp.erase();
  }

  // Erase the context block arg.
  block.eraseArgument(block.getNumArguments() - 1);

  // Drop the last typeOfArgs entry on the return op to match.
  if (auto returnOp = dyn_cast<hir::ReturnOp>(block.getTerminator())) {
    auto args = returnOp.getTypeOfArgs();
    if (!args.empty()) {
      SmallVector<Value> newArgTypes(args.drop_back());
      returnOp.getTypeOfArgsMutable().assign(newArgTypes);
    }
  }

  // Update argNames to match.
  SmallVector<Attribute> newArgNames(func.getArgNames().begin(),
                                     func.getArgNames().end());
  if (!newArgNames.empty())
    newArgNames.pop_back();
  func.setArgNamesAttr(ArrayAttr::get(func.getContext(), newArgNames));
}

/// Walk the function body for hir.call ops that pass hir.mir_constant opaque
/// values as the last argument (the context arg of the callee). Clone the
/// callee, expand the opaque context, and update the call.
void SpecializeFuncsPass::transitiveSpecialize(hir::FuncOp func,
                                               SymbolTable &symbolTable) {
  SmallVector<hir::CallOp> callsToSpecialize;
  func.walk([&](hir::CallOp callOp) {
    if (callOp.getArguments().empty())
      return;
    auto lastArg = callOp.getArguments().back();
    if (lastArg.getDefiningOp<hir::MIRConstantOp>())
      callsToSpecialize.push_back(callOp);
  });

  SmallVector<hir::FuncOp, 4> templateFuncs;
  for (auto callOp : callsToSpecialize) {
    auto lastArg = callOp.getArguments().back();
    auto mirConst = lastArg.getDefiningOp<hir::MIRConstantOp>();
    auto opaqueAttr = dyn_cast<base::OpaqueAttr>(mirConst.getValue());
    if (!opaqueAttr)
      continue;

    // Handle MultiphaseFuncOp callees: expand the opaque in the first
    // sub-function and update the multiphase_func's args. This enables
    // the first sub-function to be lowered and interpreted in the next
    // PhaseEvalLoop iteration.
    if (auto calleeMpFunc =
            symbolTable.lookup<hir::MultiphaseFuncOp>(callOp.getCallee())) {
      LLVM_DEBUG(llvm::dbgs() << "Transitive specializing multiphase @"
                              << calleeMpFunc.getSymName() << "\n");

      // Check specialization cache.
      auto key =
          std::make_pair(calleeMpFunc.getSymNameAttr(), Attribute(opaqueAttr));
      auto cacheIt = specCache.find(key);
      if (cacheIt != specCache.end()) {
        auto specName = cacheIt->second;
        LLVM_DEBUG(llvm::dbgs()
                   << "  Reusing cached specialization @" << specName << "\n");
        OpBuilder builder(callOp);
        SmallVector<Value> newArgs(callOp.getArguments().drop_back());
        SmallVector<Value> newTypeOfArgs(callOp.getTypeOfArgs().drop_back());
        auto newCallOp = hir::CallOp::create(
            builder, callOp.getLoc(), callOp.getResultTypes(), specName,
            newArgs, newTypeOfArgs, callOp.getTypeOfResults());
        callOp.replaceAllUsesWith(newCallOp.getResults());
        callOp.erase();
        continue;
      }

      // Clone the MultiphaseFuncOp if it has other users (e.g., from a
      // split_func reference or another call site). Only mutate in place
      // for the last remaining user.
      auto mpFuncToSpecialize = calleeMpFunc;
      unsigned useCount = 0;
      if (auto uses =
              SymbolTable::getSymbolUses(calleeMpFunc, getOperation())) {
        for (auto &use : *uses) {
          (void)use;
          if (++useCount > 1)
            break;
        }
      }
      if (useCount > 1) {
        // Clone the MultiphaseFuncOp and its sub-functions.
        auto specName =
            (calleeMpFunc.getSymName() + "_" + Twine(specCounter++)).str();
        OpBuilder builder(calleeMpFunc);
        builder.setInsertionPointAfter(calleeMpFunc);

        // Clone sub-functions first.
        SmallVector<Attribute> newSubFuncAttrs;
        for (auto subRef : calleeMpFunc.getPhaseFuncs()) {
          auto subName = cast<FlatSymbolRefAttr>(subRef).getValue();
          if (auto subFunc = symbolTable.lookup<hir::FuncOp>(subName)) {
            auto clonedSub = cast<hir::FuncOp>(builder.clone(*subFunc));
            auto subSpecName = (subName + "_" + Twine(specCounter++)).str();
            clonedSub.setSymName(subSpecName);
            symbolTable.insert(clonedSub);
            newSubFuncAttrs.push_back(
                FlatSymbolRefAttr::get(&getContext(), clonedSub.getSymName()));
          } else {
            newSubFuncAttrs.push_back(subRef);
          }
        }

        // Create the cloned MultiphaseFuncOp.
        auto clonedMp = hir::MultiphaseFuncOp::create(
            builder, calleeMpFunc.getLoc(), builder.getStringAttr(specName),
            /*sym_visibility=*/StringAttr{}, calleeMpFunc.getArgNamesAttr(),
            calleeMpFunc.getArgIsFirstAttr(), calleeMpFunc.getResultNamesAttr(),
            builder.getArrayAttr(newSubFuncAttrs));
        symbolTable.insert(clonedMp);
        specCache[key] = clonedMp.getSymNameAttr();
        mpFuncToSpecialize = clonedMp;
      } else {
        specCache[key] = calleeMpFunc.getSymNameAttr();
      }

      // Expand the opaque in the first sub-function.
      auto firstSubSym =
          cast<FlatSymbolRefAttr>(mpFuncToSpecialize.getPhaseFuncs()[0]);
      if (auto firstSubFunc =
              symbolTable.lookup<hir::FuncOp>(firstSubSym.getValue())) {
        expandOpaqueContext(firstSubFunc, opaqueAttr);

        // Remove the "first" args from the MultiphaseFuncOp.
        auto argIsFirst = mpFuncToSpecialize.getArgIsFirst();
        SmallVector<Attribute> newArgNames;
        SmallVector<bool> newArgIsFirst;
        for (unsigned k = 0; k < mpFuncToSpecialize.getArgNames().size(); ++k) {
          if (!argIsFirst[k]) {
            newArgNames.push_back(mpFuncToSpecialize.getArgNames()[k]);
            newArgIsFirst.push_back(false);
          }
        }
        mpFuncToSpecialize.setArgNamesAttr(
            ArrayAttr::get(&getContext(), newArgNames));
        mpFuncToSpecialize.setArgIsFirstAttr(
            DenseBoolArrayAttr::get(&getContext(), newArgIsFirst));
      }

      // Update the call: drop the last argument and reference the
      // specialized MultiphaseFuncOp.
      OpBuilder builder(callOp);
      SmallVector<Value> newArgs(callOp.getArguments().drop_back());
      SmallVector<Value> newTypeOfArgs(callOp.getTypeOfArgs().drop_back());
      auto newCallOp =
          hir::CallOp::create(builder, callOp.getLoc(), callOp.getResultTypes(),
                              mpFuncToSpecialize.getSymNameAttr(), newArgs,
                              newTypeOfArgs, callOp.getTypeOfResults());
      callOp.replaceAllUsesWith(newCallOp.getResults());
      callOp.erase();
      continue;
    }

    auto calleeFunc = symbolTable.lookup<hir::FuncOp>(callOp.getCallee());
    if (!calleeFunc)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Transitive specializing @"
                            << calleeFunc.getSymName() << "\n");

    if (!llvm::is_contained(templateFuncs, calleeFunc))
      templateFuncs.push_back(calleeFunc);

    // Check the specialization cache to avoid redundant clones.
    auto key =
        std::make_pair(calleeFunc.getSymNameAttr(), Attribute(opaqueAttr));
    auto cacheIt = specCache.find(key);
    if (cacheIt != specCache.end()) {
      // Reuse existing specialization.
      auto specName = cacheIt->second;
      LLVM_DEBUG(llvm::dbgs()
                 << "  Reusing cached specialization @" << specName << "\n");
      OpBuilder builder(callOp);
      SmallVector<Value> newArgs(callOp.getArguments().drop_back());
      SmallVector<Value> newTypeOfArgs(callOp.getTypeOfArgs().drop_back());
      auto newCallOp = hir::CallOp::create(
          builder, callOp.getLoc(), callOp.getResultTypes(), specName, newArgs,
          newTypeOfArgs, callOp.getTypeOfResults());
      callOp.replaceAllUsesWith(newCallOp.getResults());
      callOp.erase();
      continue;
    }

    // Clone the callee.
    OpBuilder builder(calleeFunc);
    builder.setInsertionPointAfter(calleeFunc);
    auto cloned = cast<hir::FuncOp>(builder.clone(*calleeFunc));
    auto specName =
        (calleeFunc.getSymName() + "_" + Twine(specCounter++)).str();
    cloned.setSymName(specName);
    symbolTable.insert(cloned);
    specCache[key] = cloned.getSymNameAttr();

    // Expand the opaque context in the clone.
    expandOpaqueContext(cloned, opaqueAttr);

    // Update the call: drop the last argument and reference the specialized
    // callee.
    builder.setInsertionPoint(callOp);
    SmallVector<Value> newArgs(callOp.getArguments().drop_back());
    SmallVector<Value> newTypeOfArgs(callOp.getTypeOfArgs().drop_back());
    auto newCallOp =
        hir::CallOp::create(builder, callOp.getLoc(), callOp.getResultTypes(),
                            builder.getStringAttr(specName), newArgs,
                            newTypeOfArgs, callOp.getTypeOfResults());
    callOp.replaceAllUsesWith(newCallOp.getResults());
    callOp.erase();

    // Recursively specialize the clone.
    transitiveSpecialize(cloned, symbolTable);
  }

  // Erase template functions that have no remaining symbol uses.
  for (auto templateFunc : templateFuncs) {
    if (templateFunc.symbolKnownUseEmpty(func->getParentOfType<ModuleOp>()))
      symbolTable.erase(templateFunc);
  }
}

void SpecializeFuncsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  // Multiphase chaining: chain evaluated results into the next sub-function.
  bool changed = true;
  while (changed) {
    changed = false;

    for (auto mpFunc : llvm::make_early_inc_range(
             getOperation().getOps<hir::MultiphaseFuncOp>())) {
      auto phaseFuncs = mpFunc.getPhaseFuncs();
      if (phaseFuncs.empty())
        continue;

      auto firstSym = cast<FlatSymbolRefAttr>(phaseFuncs[0]);
      auto evalFunc =
          symbolTable.lookup<mir::EvaluatedFuncOp>(firstSym.getValue());
      if (!evalFunc)
        continue;

      LLVM_DEBUG(llvm::dbgs() << "Chaining " << firstSym << " in multiphase @"
                              << mpFunc.getSymName() << "\n");

      // Extract the evaluated result attributes.
      auto resultAttrs = evalFunc.getResults();

      // If only one sub-function remains after removing the evaluated one,
      // or there's no next function, dissolve.
      if (phaseFuncs.size() < 2) {
        symbolTable.erase(evalFunc);
        mpFunc.erase();
        changed = true;
        continue;
      }

      // SplitPhases always emits opaque pack/unpack pairs, so
      // evaluated_func should always have exactly one result (the opaque).
      if (resultAttrs.size() != 1) {
        emitBug(evalFunc.getLoc())
            << "evaluated_func has " << resultAttrs.size()
            << " results, expected 1 (opaque pack)";
        return signalPassFailure();
      }
      auto opaqueAttr = dyn_cast<base::OpaqueAttr>(resultAttrs[0]);
      if (!opaqueAttr) {
        emitBug(evalFunc.getLoc())
            << "evaluated_func result is not an opaque attribute";
        return signalPassFailure();
      }

      // Find the next sub-function.
      auto nextSym = cast<FlatSymbolRefAttr>(phaseFuncs[1]);
      auto nextFunc = symbolTable.lookup<hir::FuncOp>(nextSym.getValue());
      if (!nextFunc) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Next sub-function " << nextSym << " not found\n");
        continue;
      }

      // Expand the opaque context in the next function.
      expandOpaqueContext(nextFunc, opaqueAttr);

      // Transitively specialize callees that now receive constant opaques.
      transitiveSpecialize(nextFunc, symbolTable);

      // Remove the evaluated first sub-function and update the multiphase_func.
      symbolTable.erase(evalFunc);

      SmallVector<Attribute> newPhaseFuncs(phaseFuncs.begin() + 1,
                                           phaseFuncs.end());

      // Remove "first" args from the multiphase_func.
      auto argIsFirst = mpFunc.getArgIsFirst();
      SmallVector<Attribute> newArgNames;
      SmallVector<bool> newArgIsFirst;
      for (unsigned k = 0; k < mpFunc.getArgNames().size(); ++k) {
        if (!argIsFirst[k]) {
          newArgNames.push_back(mpFunc.getArgNames()[k]);
          newArgIsFirst.push_back(false);
        }
      }
      // Remaining args keep their original first/last designation. Don't
      // relabel last args as first — they should stay targeted at the last
      // sub-function.

      if (newPhaseFuncs.size() <= 1) {
        // Dissolve: redirect all symbol uses of the multiphase_func to the
        // remaining sub-function, then erase the multiphase_func.
        if (!newPhaseFuncs.empty()) {
          auto remainingSym = cast<FlatSymbolRefAttr>(newPhaseFuncs[0]);
          if (failed(SymbolTable::replaceAllSymbolUses(
                  mpFunc, remainingSym.getAttr(), getOperation()))) {
            emitBug(mpFunc.getLoc())
                << "failed to replace symbol uses of @" << mpFunc.getSymName();
            return signalPassFailure();
          }
        }
        symbolTable.erase(mpFunc);
      } else {
        mpFunc.setPhaseFuncsAttr(ArrayAttr::get(&getContext(), newPhaseFuncs));
        mpFunc.setArgNamesAttr(ArrayAttr::get(&getContext(), newArgNames));
        mpFunc.setArgIsFirstAttr(
            DenseBoolArrayAttr::get(&getContext(), newArgIsFirst));
      }

      changed = true;
    }
  }

  // Split-func chaining: when a split_func entry is an evaluated_func and the
  // next entry is an HIR func, chain the evaluated result's context (the last
  // result, an opaque) into the next entry. This handles cross-phase chaining
  // within split_funcs, complementing the within-MultiphaseFuncOp chaining
  // above.
  for (auto splitFunc : getOperation().getOps<hir::SplitFuncOp>()) {
    auto entries = splitFunc.getPhaseFuncs();
    for (unsigned i = 0; i + 1 < entries.size(); ++i) {
      auto entrySym = cast<FlatSymbolRefAttr>(entries[i]);
      auto evalFunc =
          symbolTable.lookup<mir::EvaluatedFuncOp>(entrySym.getValue());
      if (!evalFunc)
        continue;

      auto nextSym = cast<FlatSymbolRefAttr>(entries[i + 1]);
      auto nextFunc = symbolTable.lookup<hir::FuncOp>(nextSym.getValue());
      if (!nextFunc)
        continue;

      // The last result of the evaluated_func is the opaque context.
      auto resultAttrs = evalFunc.getResults();
      if (resultAttrs.empty())
        continue;
      auto opaqueAttr =
          dyn_cast<base::OpaqueAttr>(resultAttrs[resultAttrs.size() - 1]);
      if (!opaqueAttr)
        continue;

      LLVM_DEBUG(llvm::dbgs()
                 << "Split-func chaining " << entrySym << " into " << nextSym
                 << " in split_func @" << splitFunc.getSymName() << "\n");

      expandOpaqueContext(nextFunc, opaqueAttr);
      transitiveSpecialize(nextFunc, symbolTable);
    }
  }

  // Strip unused opaque context args from HIR functions. When SplitPhases
  // emits an empty opaque pair (zero cross-phase values), the zero-result
  // opaque_unpack gets removed by canonicalization, leaving an unused block
  // arg with opaque_type. Remove it so the function can be lowered to MIR.
  // Skip multiphase sub-functions — their opaque args are resolved through
  // SpecializeFuncs chaining, not this cleanup.
  // Collect non-first sub-functions of multiphase_funcs (their opaque args
  // are part of the within-group protocol) and map first sub-functions to
  // their parent multiphase_func (so we can update it when stripping).
  DenseSet<StringRef> mpNonFirstSubs;
  DenseMap<StringRef, hir::MultiphaseFuncOp> mpFirstSubToParent;
  for (auto mpFunc : getOperation().getOps<hir::MultiphaseFuncOp>()) {
    auto refs = mpFunc.getPhaseFuncs();
    if (!refs.empty())
      mpFirstSubToParent[cast<FlatSymbolRefAttr>(refs[0]).getValue()] = mpFunc;
    for (unsigned i = 1; i < refs.size(); ++i)
      mpNonFirstSubs.insert(cast<FlatSymbolRefAttr>(refs[i]).getValue());
  }

  for (auto func : getOperation().getOps<hir::FuncOp>()) {
    if (mpNonFirstSubs.contains(func.getSymName()))
      continue;
    auto &block = func.getBody().front();
    if (block.getNumArguments() == 0)
      continue;
    auto lastArg = block.getArgument(block.getNumArguments() - 1);
    if (!lastArg.use_empty())
      continue;

    // Check if the corresponding typeOfArgs entry is opaque_type.
    auto retOp = dyn_cast<hir::ReturnOp>(block.getTerminator());
    if (!retOp)
      continue;
    auto typeOfArgs = retOp.getTypeOfArgs();
    if (typeOfArgs.size() != block.getNumArguments())
      continue;
    auto lastTypeOfArg = typeOfArgs[block.getNumArguments() - 1];
    if (!lastTypeOfArg.getDefiningOp<hir::OpaqueTypeOp>())
      continue;

    // Skip stripping if the function (or its parent MultiphaseFuncOp) has
    // callers beyond split_func/multiphase_func references. Direct callers
    // still pass the arg; transitiveSpecialize will handle it when the
    // caller is processed.
    auto checkExternalCallers = [&](Operation *symbolOp) -> bool {
      if (auto uses = SymbolTable::getSymbolUses(symbolOp, getOperation())) {
        for (auto &use : *uses) {
          if (!isa<hir::SplitFuncOp, hir::MultiphaseFuncOp>(use.getUser()))
            return true;
        }
      }
      return false;
    };

    bool hasExternalCallers = checkExternalCallers(func);
    if (!hasExternalCallers) {
      auto it = mpFirstSubToParent.find(func.getSymName());
      if (it != mpFirstSubToParent.end())
        hasExternalCallers = checkExternalCallers(it->second);
    }
    if (hasExternalCallers) {
      LLVM_DEBUG(llvm::dbgs() << "  Skipping strip of @" << func.getSymName()
                              << " — has external callers\n");
      continue;
    }

    LLVM_DEBUG(llvm::dbgs() << "Stripping unused opaque arg from @"
                            << func.getSymName() << "\n");

    // Erase the block arg.
    block.eraseArgument(block.getNumArguments() - 1);

    // Drop the last typeOfArgs entry.
    SmallVector<Value> newArgTypes(typeOfArgs.drop_back());
    retOp.getTypeOfArgsMutable().assign(newArgTypes);

    // Update argNames on the function.
    SmallVector<Attribute> newArgNames(func.getArgNames().begin(),
                                       func.getArgNames().end());
    if (!newArgNames.empty())
      newArgNames.pop_back();
    func.setArgNamesAttr(ArrayAttr::get(&getContext(), newArgNames));

    // If this is the first sub-function of a multiphase_func, remove the
    // corresponding "first" arg from the multiphase declaration.
    auto mpIt = mpFirstSubToParent.find(func.getSymName());
    if (mpIt != mpFirstSubToParent.end()) {
      auto mpFunc = mpIt->second;
      auto argIsFirst = mpFunc.getArgIsFirst();
      SmallVector<Attribute> mpArgNames;
      SmallVector<bool> mpArgIsFirst;
      for (unsigned k = 0; k < mpFunc.getArgNames().size(); ++k) {
        if (!argIsFirst[k]) {
          mpArgNames.push_back(mpFunc.getArgNames()[k]);
          mpArgIsFirst.push_back(false);
        }
      }
      mpFunc.setArgNamesAttr(ArrayAttr::get(&getContext(), mpArgNames));
      mpFunc.setArgIsFirstAttr(
          DenseBoolArrayAttr::get(&getContext(), mpArgIsFirst));
    }
  }

  // Clean up multiphase_func ops that are no longer referenced by any
  // split_func. These are templates whose call sites have all been specialized.
  for (auto mpFunc : llvm::make_early_inc_range(
           getOperation().getOps<hir::MultiphaseFuncOp>())) {
    if (mpFunc.symbolKnownUseEmpty(getOperation())) {
      // Erase unreferenced sub-functions.
      for (auto phaseFuncAttr : mpFunc.getPhaseFuncs()) {
        auto sym = cast<FlatSymbolRefAttr>(phaseFuncAttr);
        if (auto *op = symbolTable.lookup(sym.getValue()))
          if (SymbolTable::symbolKnownUseEmpty(op, getOperation()))
            symbolTable.erase(op);
      }
      mpFunc.erase();
    }
  }
}
