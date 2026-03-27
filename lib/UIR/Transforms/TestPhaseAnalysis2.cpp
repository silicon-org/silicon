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
/// with a `{phase = N}` attribute showing the computed phase. Floating
/// constants (INT16_MIN) are annotated as `{phase = "float"}`.
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

      auto *ctx = funcOp.getContext();

      // Turn a phase value into a string attribute: "float" or "-1", "0", etc.
      auto phaseAttr = [&](int16_t phase) -> StringAttr {
        if (phase == INT16_MIN)
          return StringAttr::get(ctx, "float");
        return StringAttr::get(ctx, std::to_string(phase));
      };

      // Build an array of phase strings for a range of values.
      auto valuePhases = [&](auto values) -> ArrayAttr {
        SmallVector<Attribute> attrs;
        for (Value v : values) {
          auto it = analysis.actualPhase.find(v);
          if (it != analysis.actualPhase.end())
            attrs.push_back(phaseAttr(it->second));
          else
            attrs.push_back(StringAttr::get(ctx, "?"));
        }
        return ArrayAttr::get(ctx, attrs);
      };

      // Annotate each op with its phase, plus operand and result phases.
      auto annotate = [&](Region &region) {
        region.walk([&](Operation *op) {
          auto it = analysis.opPhases.find(op);
          if (it == analysis.opPhases.end())
            return;
          op->setAttr("pa.phase", phaseAttr(it->second));

          if (op->getNumOperands() > 0)
            op->setAttr("pa.operands", valuePhases(op->getOperands()));

          if (op->getNumResults() > 0)
            op->setAttr("pa.results", valuePhases(op->getResults()));
        });
      };
      annotate(funcOp.getSignature());
      annotate(funcOp.getBody());
    }
  }
};
} // namespace

void silicon::uir::registerTestPhaseAnalysis2Pass() {
  PassRegistration<TestPhaseAnalysis2Pass>();
}
