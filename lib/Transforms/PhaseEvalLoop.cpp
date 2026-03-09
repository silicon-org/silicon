//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "silicon/MIR/Ops.h"
#include "silicon/MIR/Passes.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "phase-eval-loop"

namespace silicon {
#define GEN_PASS_DEF_PHASEEVALLOOPPASS
#include "silicon/Transforms/Passes.h.inc"
} // namespace silicon

//===----------------------------------------------------------------------===//
// PhaseEvalLoopPass
//
// Iteratively runs HIRToMIR + canonicalize/CSE + Interpret + SpecializeFuncs +
// canonicalize/CSE until no multiphase_func ops remain and no HIR functions
// need lowering. Each iteration peels off one more const phase.
//===----------------------------------------------------------------------===//

namespace {
struct PhaseEvalLoopPass
    : public silicon::impl::PhaseEvalLoopPassBase<PhaseEvalLoopPass> {
  void runOnOperation() override;
};
} // namespace

void PhaseEvalLoopPass::runOnOperation() {
  constexpr unsigned maxIterations = 100;

  // Collect actionable ops: multiphase_func ops whose first sub-function has
  // been evaluated (ready for chaining), and HIR funcs that don't contain
  // opaque_unpack ops (ready for lowering). Ops that require further
  // specialization before they can be processed are not collected. Returns
  // sets of symbol names so the progress check can detect when different ops
  // are actionable even if the total count stays the same.
  using NameSet = DenseSet<StringAttr>;
  struct ActionableOps {
    NameSet multiphase;
    NameSet hirFuncs;
  };
  auto collectActionableOps = [&]() -> ActionableOps {
    SymbolTable symTable(getOperation());
    ActionableOps result;
    getOperation()->walk([&](Operation *op) {
      if (auto mpFunc = dyn_cast<hir::MultiphaseFuncOp>(op)) {
        // Only count if the first sub-function is actionable: either already
        // evaluated, or a zero-arg function that can be evaluated. Template
        // multiphase_func ops whose sub-functions take args are processed
        // through transitive specialization, not direct evaluation.
        if (!mpFunc.getPhaseFuncs().empty()) {
          auto firstSym = cast<FlatSymbolRefAttr>(mpFunc.getPhaseFuncs()[0]);
          auto name = firstSym.getValue();
          if (symTable.lookup<mir::EvaluatedFuncOp>(name)) {
            result.multiphase.insert(mpFunc.getSymNameAttr());
          } else if (auto hirFunc = symTable.lookup<hir::FuncOp>(name)) {
            if (hirFunc.getBody().front().getNumArguments() == 0)
              result.multiphase.insert(mpFunc.getSymNameAttr());
          } else if (auto mirFunc = symTable.lookup<mir::FuncOp>(name)) {
            if (mirFunc.getBody().front().getNumArguments() == 0)
              result.multiphase.insert(mpFunc.getSymNameAttr());
          }
        }
      } else if (auto func = dyn_cast<hir::FuncOp>(op)) {
        // Only count HIR funcs that are actually lowerable: no
        // opaque_unpack ops, no opaque_type in typeOfArgs, and no calls
        // to MultiphaseFuncOp symbols (all indicate unresolved state).
        bool blocked = false;
        func.walk([&](Operation *inner) {
          if (isa<hir::OpaqueUnpackOp>(inner))
            blocked = true;
          if (auto ret = dyn_cast<hir::ReturnOp>(inner))
            for (auto val : ret.getTypeOfArgs())
              if (val.getDefiningOp<hir::OpaqueTypeOp>())
                blocked = true;
          if (auto call = dyn_cast<hir::CallOp>(inner)) {
            auto *callee = symTable.lookup(call.getCallee());
            if (isa_and_nonnull<hir::MultiphaseFuncOp>(callee))
              blocked = true;
          }
        });
        if (!blocked)
          result.hirFuncs.insert(func.getSymNameAttr());
      }
    });
    return result;
  };

  for (unsigned i = 0; i < maxIterations; ++i) {
    auto before = collectActionableOps();

    if (before.multiphase.empty() && before.hirFuncs.empty())
      break;

    LLVM_DEBUG(llvm::dbgs()
               << "Phase eval loop iteration " << i
               << " (multiphase=" << before.multiphase.size()
               << ", hirFuncs=" << before.hirFuncs.size() << ")\n");

    // Build and run the sub-pipeline.
    OpPassManager subPipeline("builtin.module");
    subPipeline.addPass(createHIRToMIRPass());
    auto &anyPM = subPipeline.nestAny();
    anyPM.addPass(mlir::createCanonicalizerPass());
    anyPM.addPass(mlir::createCSEPass());
    subPipeline.addPass(mir::createInterpretPass());
    subPipeline.addPass(hir::createSpecializeFuncsPass());
    auto &anyPM2 = subPipeline.nestAny();
    anyPM2.addPass(mlir::createCanonicalizerPass());
    anyPM2.addPass(mlir::createCSEPass());
    subPipeline.addPass(mlir::createSymbolDCEPass());
    if (failed(runPipeline(subPipeline, getOperation())))
      return signalPassFailure();

    // Progress check: verify that the set of actionable ops changed.
    auto after = collectActionableOps();

    if (after.multiphase == before.multiphase &&
        after.hirFuncs == before.hirFuncs && i > 0) {
      auto diag = emitBug(getOperation().getLoc())
                  << "phase eval loop made no progress in iteration " << i;

      // List remaining HIR funcs that couldn't be lowered.
      SymbolTable diagSymTable(getOperation());
      getOperation()->walk([&](hir::FuncOp func) {
        bool blocked = false;
        func.walk([&](Operation *inner) {
          if (isa<hir::OpaqueUnpackOp>(inner))
            blocked = true;
          if (auto ret = dyn_cast<hir::ReturnOp>(inner))
            for (auto val : ret.getTypeOfArgs())
              if (val.getDefiningOp<hir::OpaqueTypeOp>())
                blocked = true;
          if (auto call = dyn_cast<hir::CallOp>(inner)) {
            auto *callee = diagSymTable.lookup(call.getCallee());
            if (isa_and_nonnull<hir::MultiphaseFuncOp>(callee))
              blocked = true;
          }
        });
        if (blocked)
          return;
        diag.attachNote(func.getLoc()) << "hir.func @" << func.getSymName()
                                       << " could not be lowered to MIR";
      });

      getOperation()->walk([&](hir::MultiphaseFuncOp mpFunc) {
        diag.attachNote(mpFunc.getLoc())
            << "hir.multiphase_func @" << mpFunc.getSymName()
            << " still pending";
      });

      return signalPassFailure();
    }
  }
}
