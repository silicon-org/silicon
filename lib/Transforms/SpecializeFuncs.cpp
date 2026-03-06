//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/MIR/Attributes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "specialize-funcs"

namespace silicon {
#define GEN_PASS_DEF_SPECIALIZEFUNCSPASS
#include "silicon/Transforms/Passes.h.inc"
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
    : public silicon::impl::SpecializeFuncsPassBase<SpecializeFuncsPass> {
  void runOnOperation() override;

  /// Expand an opaque context arg in a function: replace the last block arg
  /// with individual hir.mir_constant ops derived from the opaque attribute.
  void expandOpaqueContext(hir::FuncOp func, mir::OpaqueAttr opaqueAttr);

  /// Transitively specialize callees that receive hir.mir_constant opaque
  /// arguments.
  void transitiveSpecialize(hir::FuncOp func, SymbolTable &symbolTable);

  /// Legacy: specialize a mir::FuncOp by materializing constants from a
  /// SpecializedFuncAttr.
  mir::FuncOp specialize(mir::FuncOp originalFunc,
                         mir::SpecializedFuncAttr spec);

  unsigned specCounter = 0;
};
} // namespace

/// Replace the last block arg (the opaque context) with individual
/// hir.mir_constant ops for each element of the opaque attribute. Erases the
/// coerce_type + opaque_unpack chain consuming the context arg.
void SpecializeFuncsPass::expandOpaqueContext(hir::FuncOp func,
                                              mir::OpaqueAttr opaqueAttr) {
  auto &block = func.getBody().front();
  auto ctxArg = block.getArgument(block.getNumArguments() - 1);

  // Find the coerce_type consuming the context arg.
  hir::CoerceTypeOp coerceOp;
  for (auto *user : ctxArg.getUsers()) {
    if (auto c = dyn_cast<hir::CoerceTypeOp>(user)) {
      coerceOp = c;
      break;
    }
  }

  // Find the opaque_unpack consuming the coerced value.
  hir::OpaqueUnpackOp unpackOp;
  if (coerceOp) {
    for (auto *user : coerceOp.getResult().getUsers()) {
      if (auto u = dyn_cast<hir::OpaqueUnpackOp>(user)) {
        unpackOp = u;
        break;
      }
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

    // Erase the unpack, coerce_type, and opaque_type ops.
    unpackOp.erase();
    if (coerceOp && coerceOp.use_empty()) {
      auto typeOperand = coerceOp.getTypeOperand();
      coerceOp.erase();
      if (auto *defOp = typeOperand.getDefiningOp())
        if (defOp->use_empty())
          defOp->erase();
    }
  }

  // Erase the context block arg.
  block.eraseArgument(block.getNumArguments() - 1);

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
    auto opaqueAttr = dyn_cast<mir::OpaqueAttr>(mirConst.getValue());
    if (!opaqueAttr)
      continue;

    auto calleeFunc = symbolTable.lookup<hir::FuncOp>(callOp.getCallee());
    if (!calleeFunc)
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Transitive specializing @"
                            << calleeFunc.getSymName() << "\n");

    if (!llvm::is_contained(templateFuncs, calleeFunc))
      templateFuncs.push_back(calleeFunc);

    // Clone the callee.
    OpBuilder builder(calleeFunc);
    builder.setInsertionPointAfter(calleeFunc);
    auto cloned = cast<hir::FuncOp>(builder.clone(*calleeFunc));
    auto specName =
        (calleeFunc.getSymName() + "_" + Twine(specCounter++)).str();
    cloned.setSymName(specName);
    symbolTable.insert(cloned);

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

      // Find the next sub-function.
      auto nextSym = cast<FlatSymbolRefAttr>(phaseFuncs[1]);
      auto nextFunc = symbolTable.lookup<hir::FuncOp>(nextSym.getValue());
      if (!nextFunc) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  Next sub-function " << nextSym << " not found\n");
        continue;
      }

      // The evaluated_func should have exactly one result (the opaque pack).
      if (resultAttrs.size() != 1) {
        emitBug(evalFunc.getLoc())
            << "evaluated_func has " << resultAttrs.size()
            << " results, expected 1 (opaque pack)";
        return signalPassFailure();
      }
      auto opaqueAttr = dyn_cast<mir::OpaqueAttr>(resultAttrs[0]);
      if (!opaqueAttr) {
        emitBug(evalFunc.getLoc())
            << "evaluated_func result is not an opaque attribute";
        return signalPassFailure();
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
      if (!newArgIsFirst.empty())
        newArgIsFirst[0] = true;

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

  // Legacy specialization: handle SpecializedFuncAttr constants from the old
  // mechanism. Walk mir.constant ops with SpecializedFuncAttr values and
  // create specialized function copies.
  DenseMap<mir::SpecializedFuncAttr, mir::FuncOp> funcs;
  SmallVector<Operation *> worklist;
  worklist.push_back(getOperation());
  while (!worklist.empty()) {
    auto *op = worklist.pop_back_val();
    auto result = op->walk([&](mir::ConstantOp op) {
      auto attr = dyn_cast<mir::SpecializedFuncAttr>(op.getValue());
      if (!attr)
        return WalkResult::advance();
      auto &func = funcs[attr];
      if (!func) {
        LLVM_DEBUG(llvm::dbgs() << "Specializing " << attr << "\n");
        auto originalFunc =
            symbolTable.lookup<mir::FuncOp>(attr.getFunc().getAttr());
        if (!originalFunc) {
          op.emitError() << "callee not found: " << attr.getFunc();
          return WalkResult::interrupt();
        }
        func = specialize(originalFunc, attr);
        if (!func)
          return WalkResult::interrupt();
        symbolTable.insert(func);
        worklist.push_back(func);
      }
      OpBuilder builder(op);
      auto newAttr = mir::FuncAttr::get(
          &getContext(),
          FunctionType::get(&getContext(), attr.getArgs(), attr.getResults()),
          FlatSymbolRefAttr::get(func.getSymNameAttr()));
      auto newOp = mir::ConstantOp::create(builder, op.getLoc(), newAttr);
      LLVM_DEBUG(llvm::dbgs()
                 << "Replacing " << op << " with " << newOp << "\n");
      op.getResult().replaceAllUsesWith(newOp);
      op.erase();
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      return signalPassFailure();
  }
}

//===----------------------------------------------------------------------===//
// Legacy: Specialize a mir::FuncOp via SpecializedFuncAttr
//
// Clones the original function, materializes constant arguments at the start
// of the body, erases the corresponding block arguments, and updates the
// function_type to reflect the remaining arguments.
//===----------------------------------------------------------------------===//

mir::FuncOp SpecializeFuncsPass::specialize(mir::FuncOp originalFunc,
                                            mir::SpecializedFuncAttr spec) {
  // Make sure we have the correct number of arguments.
  auto argsExpected = spec.getArgs().size() + spec.getConsts().size();
  auto argsActual = originalFunc.getBody().getNumArguments();
  if (argsExpected != argsActual) {
    emitBug(originalFunc.getLoc())
        << "function has " << argsActual
        << " arguments, but specialization expects " << argsExpected << " ("
        << spec.getArgs().size() << " args and " << spec.getConsts().size()
        << " consts)";
    return {};
  }

  // Create a clone of the function.
  OpBuilder builder(originalFunc);
  builder.setInsertionPointAfter(originalFunc);
  auto func = cast<mir::FuncOp>(builder.clone(*originalFunc));

  // Skip past the arg-typed arguments to the const arguments.
  auto &block = func.getBody().front();
  builder.setInsertionPointToStart(&block);
  unsigned argIdx = spec.getArgs().size();

  // Materialize constant arguments. Since block args now have MIR types, the
  // constant's type should match the block arg's type directly.
  auto firstConstIdx = argIdx;
  for (auto attr : spec.getConsts()) {
    auto arg = block.getArgument(argIdx++);
    auto constOp =
        mir::ConstantOp::create(builder, arg.getLoc(), cast<TypedAttr>(attr));
    arg.replaceAllUsesWith(constOp);
  }
  block.eraseArguments(firstConstIdx, argIdx - firstConstIdx);

  // Update argNames to match the new argument count after erasing consts.
  SmallVector<Attribute> newArgNames(func.getArgNames().begin(),
                                     func.getArgNames().end());
  newArgNames.erase(newArgNames.begin() + firstConstIdx,
                    newArgNames.begin() + argIdx);
  func.setArgNamesAttr(builder.getArrayAttr(newArgNames));

  // Update function_type to reflect the remaining arg types and result types.
  SmallVector<Type> newArgTypes;
  for (auto arg : block.getArguments())
    newArgTypes.push_back(arg.getType());
  func.setFunctionTypeAttr(mlir::TypeAttr::get(FunctionType::get(
      &getContext(), newArgTypes, func.getFunctionType().getResults())));

  return func;
}
