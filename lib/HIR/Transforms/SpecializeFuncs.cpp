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

/// Erase all void calls to a given symbol throughout the module. This is used
/// when an evaluated function returned no results: any calls to it are no-ops
/// and must be removed before the evaluated func itself can be erased.
static void eraseVoidCalls(ModuleOp module, StringRef calleeName) {
  SmallVector<Operation *> toErase;
  module.walk([&](Operation *op) {
    if (auto callOp = dyn_cast<hir::CallOp>(op)) {
      if (callOp.getCallee() == calleeName && callOp.getNumResults() == 0)
        toErase.push_back(callOp);
    } else if (auto callOp = dyn_cast<mir::CallOp>(op)) {
      if (callOp.getCallee() == calleeName && callOp.getNumResults() == 0)
        toErase.push_back(callOp);
    }
  });
  for (auto *op : toErase)
    op->erase();
}

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
        eraseVoidCalls(getOperation(), evalFunc.getSymName());
        symbolTable.erase(evalFunc);
        mpFunc.erase();
        changed = true;
        continue;
      }

      // If the evaluated function has no results, there's no context to
      // chain into the next sub-function. Just remove it and continue.
      if (resultAttrs.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  No results to chain, removing evaluated func\n");
        eraseVoidCalls(getOperation(), evalFunc.getSymName());
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
          for (auto splitFunc : getOperation().getOps<hir::SplitFuncOp>()) {
            auto pfAttrs = splitFunc.getPhaseFuncs();
            bool updated = false;
            SmallVector<Attribute> updatedPhaseFuncs(pfAttrs.begin(),
                                                     pfAttrs.end());
            for (unsigned i = 0; i < pfAttrs.size(); ++i) {
              if (cast<FlatSymbolRefAttr>(pfAttrs[i]).getValue() ==
                  mpFunc.getSymName()) {
                updatedPhaseFuncs[i] = newPhaseFuncs.empty()
                                           ? FlatSymbolRefAttr{}
                                           : newPhaseFuncs[0];
                updated = true;
                break;
              }
            }
            if (updated)
              splitFunc.setPhaseFuncsAttr(
                  ArrayAttr::get(&getContext(), updatedPhaseFuncs));
          }
          mpFunc.erase();
        } else {
          mpFunc.setPhaseFuncsAttr(
              ArrayAttr::get(&getContext(), newPhaseFuncs));
          mpFunc.setArgNamesAttr(ArrayAttr::get(&getContext(), newArgNames));
          mpFunc.setArgIsFirstAttr(
              DenseBoolArrayAttr::get(&getContext(), newArgIsFirst));
        }

        changed = true;
        continue;
      }

      // The evaluated_func should have exactly one result (the opaque pack).
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
      eraseVoidCalls(getOperation(), evalFunc.getSymName());
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
        // Dissolve: update split_func to reference the remaining sub-function
        // directly.
        for (auto splitFunc : getOperation().getOps<hir::SplitFuncOp>()) {
          auto pfAttrs = splitFunc.getPhaseFuncs();
          bool updated = false;
          SmallVector<Attribute> updatedPhaseFuncs(pfAttrs.begin(),
                                                   pfAttrs.end());
          for (unsigned i = 0; i < pfAttrs.size(); ++i) {
            if (cast<FlatSymbolRefAttr>(pfAttrs[i]).getValue() ==
                mpFunc.getSymName()) {
              updatedPhaseFuncs[i] = newPhaseFuncs.empty() ? FlatSymbolRefAttr{}
                                                           : newPhaseFuncs[0];
              updated = true;
              break;
            }
          }
          if (updated)
            splitFunc.setPhaseFuncsAttr(
                ArrayAttr::get(&getContext(), updatedPhaseFuncs));
        }
        mpFunc.erase();
      } else {
        mpFunc.setPhaseFuncsAttr(ArrayAttr::get(&getContext(), newPhaseFuncs));
        mpFunc.setArgNamesAttr(ArrayAttr::get(&getContext(), newArgNames));
        mpFunc.setArgIsFirstAttr(
            DenseBoolArrayAttr::get(&getContext(), newArgIsFirst));
      }

      changed = true;
    }
  }

  // Clean up void calls to evaluated funcs with empty results. These are
  // no-op calls to phase sub-functions that have already been fully evaluated
  // and returned nothing. This happens with multi-phase functions like
  // `const const fn` where inner phases produce no results but other functions
  // still reference them.
  for (auto evalFunc : getOperation().getOps<mir::EvaluatedFuncOp>()) {
    if (!evalFunc.getResults().empty())
      continue;
    eraseVoidCalls(getOperation(), evalFunc.getSymName());
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
    auto it = mpFirstSubToParent.find(func.getSymName());
    if (it != mpFirstSubToParent.end()) {
      auto mpFunc = it->second;
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
