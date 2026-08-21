//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Ops.h"
#include "silicon/Support/MLIR.h"
#include "silicon/Transforms/Passes.h"
#include "silicon/UIR/Ops.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace silicon;

#define DEBUG_TYPE "check-calls"

namespace silicon {
#define GEN_PASS_DEF_CHECKCALLSPASS
#include "silicon/Transforms/Passes.h.inc"
} // namespace silicon

//===----------------------------------------------------------------------===//
// Inlining a Declared Signature
//===----------------------------------------------------------------------===//
//
// A function's declared types are not attributes but a computation: the
// signature region computes one type value per argument and result, possibly
// depending on the function's own arguments. Every consumer of a declaration --
// a call site and the function body itself -- needs those types expressed in
// its *own* context, so the checker copies the computation there and connects
// the copy to the local types with `hir.unify` and `hir.coerce_type`.
//
// Copying, rather than referring, is what makes the phase split work: a caller
// that recomputes a dependent callee type from arguments it already holds
// threads that type through its own phase boundaries, and never has to read the
// callee's private context. Agreement between the two computations is then
// structural, since both sides are copies of the same region.

namespace {
/// The type values a copied signature computes, in declaration order.
struct DeclaredTypes {
  SmallVector<Value> args;
  SmallVector<Value> results;
};
} // namespace

/// Copy a function's signature computation to a builder's insertion point.
///
/// The signature's block arguments stand for the function's arguments and are
/// substituted by `args`, so the copy computes the declared types in terms of
/// the values the consumer holds. Returns the copied type values.
static FailureOr<DeclaredTypes>
inlineSignature(uir::DeclaredFuncOpInterface funcOp, ValueRange args,
                OpBuilder &builder) {
  auto &signature = funcOp.getSignature();

  // The declared types are the operands of the `uir.signature` terminating the
  // signature region. Call checking runs on structured, exit-normalized input:
  // a signature spread across multiple blocks, exiting through more than one
  // `uir.signature`, or not exiting at all has no single computation to copy.
  auto &entry = signature.front();
  auto signatureOp = dyn_cast<uir::SignatureOp>(entry.getTerminator());
  bool singleExit = signature.hasOneBlock() && signatureOp;
  if (singleExit)
    signature.walk([&](uir::SignatureOp op) {
      if (op != signatureOp)
        singleExit = false;
    });
  if (!singleExit)
    return funcOp->emitOpError()
           << "signature region must be a single block exiting through one "
              "'uir.signature' for its declared types to be inlined at a use; "
              "call checking expects structured, exit-normalized input";

  // Snapshot the ops before cloning any: a recursive call in a signature clones
  // the very block it inserts into.
  auto ops =
      llvm::to_vector(llvm::make_pointer_range(entry.without_terminator()));
  IRMapping mapping;
  mapping.map(entry.getArguments(), args);
  for (auto *op : ops)
    builder.clone(*op, mapping);

  DeclaredTypes types;
  for (auto type : signatureOp.getTypeOfArgs())
    types.args.push_back(mapping.lookupOrDefault(type));
  for (auto type : signatureOp.getTypeOfResults())
    types.results.push_back(mapping.lookupOrDefault(type));
  return types;
}

//===----------------------------------------------------------------------===//
// Checking Calls and Bodies
//===----------------------------------------------------------------------===//

/// Constrain a call's type operands with the callee's declared types.
///
/// The callee's signature computation is copied in front of the call, applied
/// to the call's arguments, and unified with the types the call carries. The
/// unify results become the call's type operands, so consumers of the call see
/// a type value that stands for both sides.
static LogicalResult checkCall(uir::CallOp callOp,
                               const SymbolTable &symbolTable) {
  LLVM_DEBUG(llvm::dbgs() << "- " << callOp << "\n");
  auto calleeOp =
      symbolTable.lookup<uir::DeclaredFuncOpInterface>(callOp.getCallee());
  assert(calleeOp && "callee resolved by uir.call verifier");

  OpBuilder builder(callOp);
  auto declared = inlineSignature(calleeOp, callOp.getArguments(), builder);
  if (failed(declared))
    return failure();

  SmallVector<Value> argTypes, resultTypes;
  for (auto [declaredType, callType] :
       llvm::zip_equal(declared->args, callOp.getTypeOfArgs()))
    argTypes.push_back(
        hir::UnifyOp::create(builder, callOp.getLoc(), declaredType, callType));
  for (auto [declaredType, callType] :
       llvm::zip_equal(declared->results, callOp.getTypeOfResults()))
    resultTypes.push_back(
        hir::UnifyOp::create(builder, callOp.getLoc(), declaredType, callType));
  callOp.getTypeOfArgsMutable().assign(argTypes);
  callOp.getTypeOfResultsMutable().assign(resultTypes);
  return success();
}

/// Check all calls in a function's signature region.
static LogicalResult checkSignature(uir::DeclaredFuncOpInterface funcOp,
                                    const SymbolTable &symbolTable) {
  LLVM_DEBUG(llvm::dbgs() << "Checking signature of "
                          << SymbolTable::getSymbolName(funcOp) << "\n");
  SmallVector<uir::CallOp> calls;
  funcOp.getSignature().walk([&](uir::CallOp op) { calls.push_back(op); });
  for (auto callOp : calls)
    if (failed(checkCall(callOp, symbolTable)))
      return failure();
  return success();
}

/// Constrain a function body with its own declared types and check its calls.
///
/// The function's signature computation is copied to the top of the body, where
/// it applies to the body's own arguments. Each argument is then constrained by
/// its declared type through a `hir.coerce_type` that takes over the argument's
/// uses, and every return's type operands are unified with the declared result
/// types.
static LogicalResult checkBody(uir::FuncOp funcOp,
                               const SymbolTable &symbolTable) {
  LLVM_DEBUG(llvm::dbgs() << "Checking body of " << funcOp.getSymNameAttr()
                          << "\n");
  auto &bodyBlock = funcOp.getBody().front();

  // Collect the calls the user wrote before copying anything in: the signature
  // copy brings the signature's own calls along, and those have already been
  // checked in the signature.
  SmallVector<uir::CallOp> calls;
  funcOp.getBody().walk([&](uir::CallOp op) { calls.push_back(op); });

  // Collect the argument uses the coercions take over, likewise before copying
  // the signature in. The declared types are computed from the raw arguments,
  // so the copy's uses must keep referring to them.
  SmallVector<SmallVector<OpOperand *>> argUses;
  for (auto arg : bodyBlock.getArguments()) {
    auto &uses = argUses.emplace_back();
    for (auto &use : arg.getUses())
      uses.push_back(&use);
  }

  OpBuilder builder(&bodyBlock, bodyBlock.begin());
  auto declared = inlineSignature(funcOp, bodyBlock.getArguments(), builder);
  if (failed(declared))
    return failure();

  for (auto [arg, declaredType, uses] :
       llvm::zip_equal(bodyBlock.getArguments(), declared->args, argUses)) {
    auto coerced =
        hir::CoerceTypeOp::create(builder, arg.getLoc(), arg, declaredType);
    for (auto *use : uses)
      use->set(coerced);
  }

  funcOp.getBody().walk([&](uir::ReturnOp returnOp) {
    OpBuilder returnBuilder(returnOp);
    SmallVector<Value> valueTypes;
    for (auto [valueType, declaredType] :
         llvm::zip_equal(returnOp.getTypeOfValues(), declared->results))
      valueTypes.push_back(hir::UnifyOp::create(
          returnBuilder, returnOp.getLoc(), valueType, declaredType));
    returnOp.getTypeOfValuesMutable().assign(valueTypes);
  });

  for (auto callOp : calls)
    if (failed(checkCall(callOp, symbolTable)))
      return failure();
  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//
//
// Signatures are checked before bodies, and a signature that calls another
// function must be checked after the callee's: inlining a callee's signature
// copies whatever it holds, so the callee's own calls must already be resolved
// against their declarations. That ordering is a post-order traversal of the
// call graph restricted to signature calls, and a cycle in it is a user error:
// a signature that transitively calls itself has no order that satisfies it.
//
// Every signature region is checked the same way, whether it belongs to a
// `uir.func` or to a `uir.split_func`: a declaration's signature is the
// interface every caller copies, so an unchecked call in one lands unchecked in
// user code. Nothing marks a signature as already checked, so a declaration
// checked by an earlier run is checked again; the second copy of its inlined
// types is pure computation and a duplicate constraint, which CSE and
// unification absorb.
//
// Bodies belong to `uir.func` alone. A `uir.split_func` has no body: it is the
// interface of a function whose implementation is compiled elsewhere, already
// split into per-phase functions, or not present at all.

namespace {
struct CheckCallsPass
    : public silicon::impl::CheckCallsPassBase<CheckCallsPass> {
  void runOnOperation() override;
};
} // namespace

void CheckCallsPass::runOnOperation() {
  auto &symbolTable = getAnalysis<SymbolTable>();

  /// A function on the traversal stack, together with the signature calls that
  /// still have to be visited before it can be checked itself.
  struct WorklistItem {
    uir::DeclaredFuncOpInterface funcOp;
    Location callLoc;
    SmallVector<uir::CallOp, 1> calls;
  };
  SmallVector<WorklistItem> worklist;
  SmallPtrSet<Operation *, 4> callStack;
  DenseSet<Operation *> checkedFuncs;

  auto lookupCallee = [&](uir::CallOp callOp) {
    auto callee =
        symbolTable.lookup<uir::DeclaredFuncOpInterface>(callOp.getCallee());
    assert(callee && "callee resolved by uir.call verifier");
    return callee;
  };

  auto addToWorklist = [&](uir::DeclaredFuncOpInterface funcOp,
                           Location callLoc) {
    SmallPtrSet<Operation *, 4> seenCallees;
    SmallVector<uir::CallOp, 1> calls;
    funcOp.getSignature().walk([&](uir::CallOp op) {
      auto callee = lookupCallee(op);
      if (!checkedFuncs.contains(callee) && seenCallees.insert(callee).second)
        calls.push_back(op);
    });
    worklist.push_back({funcOp, callLoc, std::move(calls)});
  };

  bool anyErrors = false;
  for (auto funcOp : getOperation().getOps<uir::DeclaredFuncOpInterface>()) {
    if (checkedFuncs.contains(funcOp))
      continue;
    addToWorklist(funcOp, funcOp.getLoc());
    callStack.clear();
    callStack.insert(funcOp);
    while (!worklist.empty()) {
      auto &item = worklist.back();

      // If all calls in the function's signature have been visited, check the
      // signature itself and pop the function off the worklist.
      if (item.calls.empty()) {
        if (failed(checkSignature(item.funcOp, symbolTable)))
          anyErrors = true;
        checkedFuncs.insert(item.funcOp);
        callStack.erase(item.funcOp);
        worklist.pop_back();
        continue;
      }

      // Descend into the next call in the function's signature, using the call
      // stack to detect recursion.
      auto callOp = item.calls.pop_back_val();
      auto callee = lookupCallee(callOp);
      if (checkedFuncs.contains(callee))
        continue;
      if (callStack.insert(callee).second) {
        addToWorklist(callee, callOp.getLoc());
        continue;
      }

      // The call stack already contains the callee, so the function is on a
      // recursive call chain through function signatures. Report the cycle
      // along the frames that lead back to the callee.
      anyErrors = true;
      auto calleeName = SymbolTable::getSymbolName(callee).getValue();
      emitError(callee.getLoc())
          << "signature of `" << calleeName << "` cannot call itself";
      bool selfSeen = false;
      for (auto &frame : worklist) {
        if (!selfSeen) {
          selfSeen = frame.funcOp == callee;
          continue;
        }
        emitRemark(frame.callLoc)
            << "called through `"
            << SymbolTable::getSymbolName(frame.funcOp).getValue() << "`";
      }
      emitRemark(callOp.getLoc())
          << "calling `" << calleeName << "` itself here";
    }
  }

  // Bodies are only checked once every signature has been, since a body copies
  // its own signature and inlines the signatures it calls.
  if (anyErrors)
    return signalPassFailure();

  // Bodies are independent of each other, but checking them in module order
  // keeps the diagnostics a malformed declaration produces in a fixed order.
  bool anyBodyErrors = false;
  for (auto funcOp : getOperation().getOps<uir::FuncOp>())
    if (failed(checkBody(funcOp, symbolTable)))
      anyBodyErrors = true;
  if (anyBodyErrors)
    signalPassFailure();
}
