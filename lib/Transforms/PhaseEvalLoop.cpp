//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Ops.h"
#include "silicon/MIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
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

  // Count actionable ops: multiphase_func ops whose first sub-function has
  // been evaluated (ready for chaining), and HIR funcs that don't contain
  // opaque_unpack ops (ready for lowering). Ops that require further
  // specialization before they can be processed are not counted.
  auto countOps = [&]() -> std::pair<unsigned, unsigned> {
    SymbolTable symTable(getOperation());
    unsigned numMultiphase = 0;
    unsigned numHIRFuncs = 0;
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
            ++numMultiphase;
          } else if (auto hirFunc = symTable.lookup<hir::FuncOp>(name)) {
            if (hirFunc.getBody().front().getNumArguments() == 0)
              ++numMultiphase;
          } else if (auto mirFunc = symTable.lookup<mir::FuncOp>(name)) {
            if (mirFunc.getBody().front().getNumArguments() == 0)
              ++numMultiphase;
          }
        }
      } else if (auto func = dyn_cast<hir::FuncOp>(op)) {
        bool hasOpaqueUnpack = false;
        func.walk([&](hir::OpaqueUnpackOp) { hasOpaqueUnpack = true; });
        if (!hasOpaqueUnpack)
          ++numHIRFuncs;
      }
    });
    return {numMultiphase, numHIRFuncs};
  };

  for (unsigned i = 0; i < maxIterations; ++i) {
    auto [numMultiphase, numHIRFuncs] = countOps();

    if (numMultiphase == 0 && numHIRFuncs == 0)
      break;

    LLVM_DEBUG(llvm::dbgs() << "Phase eval loop iteration " << i
                            << " (multiphase=" << numMultiphase
                            << ", hirFuncs=" << numHIRFuncs << ")\n");

    // Build and run the sub-pipeline.
    OpPassManager subPipeline("builtin.module");
    subPipeline.addPass(createHIRToMIRPass());
    auto &anyPM = subPipeline.nestAny();
    anyPM.addPass(mlir::createCanonicalizerPass());
    anyPM.addPass(mlir::createCSEPass());
    subPipeline.addPass(createInterpretPass());
    subPipeline.addPass(createSpecializeFuncsPass());
    auto &anyPM2 = subPipeline.nestAny();
    anyPM2.addPass(mlir::createCanonicalizerPass());
    anyPM2.addPass(mlir::createCSEPass());

    if (failed(runPipeline(subPipeline, getOperation())))
      return signalPassFailure();

    // Progress check: verify that something changed by comparing counts.
    auto [newNumMultiphase, newNumHIRFuncs] = countOps();

    if (newNumMultiphase == numMultiphase && newNumHIRFuncs == numHIRFuncs &&
        i > 0) {
      emitBug(getOperation().getLoc())
          << "phase eval loop made no progress in iteration " << i;
      return signalPassFailure();
    }
  }
}
