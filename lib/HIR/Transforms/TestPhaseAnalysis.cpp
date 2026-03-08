//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Analysis/PhaseAnalysis.h"
#include "silicon/HIR/Ops.h"
#include "silicon/HIR/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace silicon;
using namespace hir;

namespace {
/// Test pass that runs PhaseAnalysis on each `unified_func` and annotates every
/// op with a `{phase = N}` attribute showing the computed phase. Floating
/// constants (INT16_MIN) are annotated as `{phase = "float"}`.
struct TestPhaseAnalysisPass
    : public PassWrapper<TestPhaseAnalysisPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPhaseAnalysisPass)

  StringRef getArgument() const override { return "test-phase-analysis"; }
  StringRef getDescription() const override {
    return "Annotate ops with computed phase assignments (test-only)";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto funcOp : module.getOps<UnifiedFuncOp>()) {
      PhaseAnalysis analysis(funcOp);
      analysis.analyze();
      analysis.pullPhases();
      analysis.refreshPhases();

      // Annotate each op in the body with its phase.
      funcOp.getBody().walk([&](Operation *op) {
        auto it = analysis.opPhases.find(op);
        if (it == analysis.opPhases.end())
          return;
        int16_t phase = it->second;
        if (phase == INT16_MIN) {
          op->setAttr("phase", StringAttr::get(op->getContext(), "float"));
        } else {
          op->setAttr("phase",
                      IntegerAttr::get(IntegerType::get(op->getContext(), 16,
                                                        IntegerType::Signed),
                                       phase));
        }
      });
    }
  }
};
} // namespace

void silicon::hir::registerTestPhaseAnalysisPass() {
  PassRegistration<TestPhaseAnalysisPass>();
}
