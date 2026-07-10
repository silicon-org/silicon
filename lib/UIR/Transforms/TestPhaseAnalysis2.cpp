//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/UIR/Analysis/PhaseAnalysis.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace silicon;
using namespace uir;

namespace {
/// Test pass that runs PhaseAnalysis on each `uir.func` and annotates every op
/// with `pa.phase`/`pa.operands`/`pa.results` attributes showing the computed
/// phases. Floating constants are annotated as "float", never-produced results
/// as "never".
struct TestPhaseAnalysis2Pass
    : public PassWrapper<TestPhaseAnalysis2Pass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestPhaseAnalysis2Pass)

  StringRef getArgument() const override { return "test-phase-analysis2"; }
  StringRef getDescription() const override {
    return "Annotate ops with computed phase assignments for UIR (test-only)";
  }

  void runOnOperation() override {
    auto module = getOperation();
    for (auto funcOp : module.getOps<FuncOp>()) {
      PhaseAnalysis analysis(funcOp);
      if (failed(analysis.run())) {
        signalPassFailure();
        return;
      }
      analysis.annotate();
    }
  }
};
} // namespace

void silicon::uir::registerTestPhaseAnalysis2Pass() {
  PassRegistration<TestPhaseAnalysis2Pass>();
}
