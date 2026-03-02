//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/Conversion/Passes.h"
#include "silicon/HIR/Ops.h"
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
// Iteratively runs HIRToMIR + Interpret + SpecializeFuncs until no
// multiphase_func ops remain, indicating all compile-time phases have been
// evaluated. Each iteration peels off one more const phase.
//===----------------------------------------------------------------------===//

namespace {
struct PhaseEvalLoopPass
    : public silicon::impl::PhaseEvalLoopPassBase<PhaseEvalLoopPass> {
  void runOnOperation() override;
};
} // namespace

void PhaseEvalLoopPass::runOnOperation() {
  constexpr unsigned maxIterations = 100;

  for (unsigned i = 0; i < maxIterations; ++i) {
    // Check if any multiphase_func ops remain.
    bool hasMultiphase = false;
    getOperation()->walk([&](hir::MultiphaseFuncOp) {
      hasMultiphase = true;
      return WalkResult::interrupt();
    });

    // Also check if any hir::FuncOp has HIR ops that need lowering (for the
    // first iteration and after SpecializeFuncs creates new funcs).
    bool hasHIROps = false;
    for (auto func : getOperation().getOps<hir::FuncOp>()) {
      for (auto &op : func.getBody().front()) {
        if (isa<hir::HIRDialect>(op.getDialect())) {
          hasHIROps = true;
          break;
        }
      }
      if (hasHIROps)
        break;
    }

    if (!hasMultiphase && !hasHIROps)
      break;

    LLVM_DEBUG(llvm::dbgs() << "Phase eval loop iteration " << i << "\n");

    // Build and run the sub-pipeline.
    OpPassManager subPipeline("builtin.module");
    subPipeline.addNestedPass<hir::FuncOp>(createHIRToMIRPass());
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
  }
}
