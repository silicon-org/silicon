//===----------------------------------------------------------------------===//
//
// Part of Silicon, licensed under the Apache License v2.0 with LLVM Exceptions.
// See the LICENSE file for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// # Split Threaded UIR Functions into Per-Phase HIR Functions
//
// This pass is the phase-blind function splitter. It consumes the fully
// threaded phase form produced by `thread-phases` and never runs phase
// analysis: the only phase information it reads is the `uir.phase` op `phase`
// attribute and the function's declared `argPhases`/`resultPhases`.
//
// Its input is a `uir.split_func` in the *sorted* state — the declaration
// snapshot phase sorting took custody of, holding the original flat signature
// and an `impl` reference to the `uir.func` with the threaded implementation.
// For each such pair it:
//
// - **Diagnoses** content later than the declared span (runtime work past the
//   last declared phase, which has no executor) as a user error, and rejects
//   any region content that violates the threaded input contract.
// - **Splits** every `uir.phase` op in the body region into a private
//   `hir.func`. Values crossing a phase boundary ride an opaque context: each
//   non-last func packs its outgoing crossings with `hir.opaque_pack` into one
//   trailing context result, and each non-first func unpacks its incoming
//   context. Declared arguments enter at their phase's func; declared results
//   ride the `hir.return` of their phase's func. Each `uir.phase_call` becomes
//   an `hir.call` to the callee's concrete per-phase func, resolved through the
//   callee `uir.split_func`'s phase list; its context threads like any other
//   crossing.
// - **Reconstructs signatures** for each per-phase func. A per-phase signature
//   names the types of that func's arguments and results. A dependent type
//   computed one phase earlier crosses in the opaque context and is unpacked in
//   the signature region — relocated (recomputed) into the producing phase's
//   body when the body does not already thread it, so the body and signature
//   share one context. The function's *flat* signature is not reconstructed:
//   it already sits on the `uir.split_func`, preserved from before sorting.
// - **Fills the split_func**: the phase list maps each span phase to its entry
//   func, and the `impl` reference and the implementation function go away.
//   When the function has pre-span (compile-time) phases, an `hir.eager_func`
//   chains them into the earliest span entry and takes that slot.
//
// Because callers resolve their `uir.phase_call`s through the callee's phase
// list rather than by name convention, the pass runs in two stages over the
// whole module: every split_func's phase list is filled before any phase call
// is lowered.
//
// ## Crossings: a shared, deduplicated context
//
// The body and the signature are threaded independently, so each phase boundary
// carries the union of two crossing sets: the body values the next phase
// consumes, and the signature type values a later phase's signature depends on.
// Both are packed into one opaque context per boundary and deduplicated by a
// structural key `canonKey`: a declared argument keys by its index (so the body
// and signature copies of an argument coincide), a phase-op result sees through
// to the value it yields, and a pure computation keys by op name, attributes,
// and its operands' keys. Impure or opaque values (calls, context) key by
// identity and never coincide. A signature-only crossing whose computation the
// body lacks is cloned into the producing phase's body (its argument and
// earlier-crossing leaves resolved there), so it rides the shared context like
// any body crossing.
//
//===----------------------------------------------------------------------===//

#include "silicon/HIR/Dialect.h"
#include "silicon/HIR/Ops.h"
#include "silicon/UIR/Ops.h"
#include "silicon/UIR/Passes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/raw_ostream.h"
#include <map>

using namespace mlir;
using namespace silicon;
using namespace uir;

namespace silicon {
namespace uir {
#define GEN_PASS_DEF_SPLITFUNCSPASS
#include "silicon/UIR/Passes.h.inc"
} // namespace uir
} // namespace silicon

//===----------------------------------------------------------------------===//
// FuncSplitter
//===----------------------------------------------------------------------===//

namespace {

/// Splits a single threaded `uir.func` into per-phase `hir.func` ops and fills
/// in the `uir.split_func` that took custody of its declaration.
///
/// The work runs in two stages: `declare()` checks the input and creates the
/// per-phase functions and the split_func's phase list, and `implement()` fills
/// their bodies and signatures. Splitting the module's functions stage by stage
/// makes every callee's phase list available before any `uir.phase_call` is
/// lowered against it.
struct FuncSplitter {
  FuncSplitter(SplitFuncOp splitFunc, FuncOp funcOp, SymbolTable &symbolTable)
      : splitFunc(splitFunc), funcOp(funcOp), symbolTable(symbolTable),
        ctx(funcOp.getContext()), anyTy(hir::AnyType::get(funcOp.getContext())),
        argPhases(funcOp.getArgPhases()),
        resultPhases(funcOp.getResultPhases()) {}

  LogicalResult declare();
  LogicalResult implement();
  void retire();

private:
  SplitFuncOp splitFunc;
  FuncOp funcOp;
  SymbolTable &symbolTable;
  MLIRContext *ctx;
  Type anyTy;
  ArrayRef<int32_t> argPhases;
  ArrayRef<int32_t> resultPhases;

  int spanMin = 0, spanMax = 0;
  int bodyMin = 0, bodyMax = 0;

  /// The body/signature `uir.phase` op for each phase.
  std::map<int, PhaseOp> bodyPhase;
  std::map<int, PhaseOp> sigPhase;
  /// The `uir.signature` terminator of the signature region.
  SignatureOp sigTerminator;

  /// The generated per-phase `hir.func` for each body phase.
  std::map<int, hir::FuncOp> phaseFunc;

  /// One value crossing a phase boundary in the opaque context. A crossing is a
  /// body value the next phase consumes or a signature type value a later phase
  /// depends on; both regions share one deduplicated slot list per boundary.
  struct Slot {
    std::string key;
    Value source;
  };
  /// The ordered context slots entering each phase, keyed by the receiving
  /// phase P (boundary P-1 -> P), plus a key-to-index lookup.
  std::map<int, SmallVector<Slot>> slots;
  std::map<int, llvm::StringMap<unsigned>> slotIndex;

  DenseMap<Value, std::string> keyMemo;

  /// A per-phase-func signature-region unpack of the incoming context, created
  /// lazily when a dependent type needs a body crossing slot.
  struct SigCtx {
    hir::OpaqueUnpackOp unpack;
  };

  std::string phaseName(int phase);
  std::string canonKey(Value value);
  static Value underlying(Value value);

  LogicalResult diagnose();
  void computeRanges();
  void computeSlots();

  hir::FuncOp buildPhaseFunc(int phase);
  LogicalResult buildPhaseBody(int phase, hir::FuncOp func);
  LogicalResult buildPhaseSignature(int phase, hir::FuncOp func);
  Value realizeSigType(int phase, Value sigValue, hir::FuncOp func,
                       OpBuilder &builder, IRMapping &memo, SigCtx &sigCtx);
  FailureOr<FlatSymbolRefAttr> resolvePhaseEntry(PhaseCallOp call);

  void buildEagerFuncAndLinks();
};
} // namespace

//===----------------------------------------------------------------------===//
// Naming and Structural Keys
//===----------------------------------------------------------------------===//

/// The per-phase function name `@<f>.<phase>`, derived from the split_func's
/// symbol (the phase is a plain suffix; a negative phase makes the whole symbol
/// require quoting by the printer). Names are cosmetic — the split_func's phase
/// list and the eager chain's links are the authoritative connections — so the
/// symbol table is free to uniquify them.
std::string FuncSplitter::phaseName(int phase) {
  return (splitFunc.getSymName() + "." + Twine(phase)).str();
}

/// See through phase-op results to the value actually computed: a `uir.phase`
/// result resolves to the operand its region's `uir.phase_return` yields.
Value FuncSplitter::underlying(Value value) {
  while (auto phaseOp = dyn_cast_or_null<PhaseOp>(value.getDefiningOp())) {
    auto ret = cast<PhaseReturnOp>(phaseOp.getBody().front().getTerminator());
    value = ret.getYields()[cast<OpResult>(value).getResultNumber()];
  }
  return value;
}

/// A structural key identifying a value across the body and signature regions.
/// Declared arguments key by index; pure computations key by op name,
/// attributes, and operand keys; everything else keys by identity.
std::string FuncSplitter::canonKey(Value value) {
  if (auto it = keyMemo.find(value); it != keyMemo.end())
    return it->second;
  std::string key;
  value = underlying(value);
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getOwner()->getParentOp() == funcOp)
      key = ("arg#" + Twine(blockArg.getArgNumber())).str();
    else
      key = ("blk#" + Twine((intptr_t)value.getAsOpaquePointer())).str();
  } else {
    Operation *defOp = value.getDefiningOp();
    if (defOp->getNumRegions() == 0 && isMemoryEffectFree(defOp)) {
      llvm::raw_string_ostream os(key);
      os << defOp->getName().getStringRef() << "[";
      llvm::interleaveComma(defOp->getAttrs(), os, [&](NamedAttribute attr) {
        os << attr.getName() << "=" << attr.getValue();
      });
      os << "](";
      llvm::interleaveComma(defOp->getOperands(), os,
                            [&](Value v) { os << canonKey(v); });
      os << ")#" << cast<OpResult>(value).getResultNumber();
    } else {
      key = ("id#" + Twine((intptr_t)value.getAsOpaquePointer())).str();
    }
  }
  keyMemo[value] = key;
  return key;
}

//===----------------------------------------------------------------------===//
// Diagnostics
//===----------------------------------------------------------------------===//

/// Reject inputs that violate the threaded contract, and diagnose content
/// later than the declared span as a user error. Runs before any rewriting.
LogicalResult FuncSplitter::diagnose() {
  // A region holds only `uir.phase` ops and its terminator; each phase op is
  // straight-line (terminated directly by `uir.phase_return`). thread-phases
  // guarantees this shape; anything else means the pipeline ran out of order.
  auto checkRegion = [&](Region &region, bool isBody) -> LogicalResult {
    Block &block = region.front();
    Operation *term = block.getTerminator();
    if (isBody && !isa<UnreachableOp>(term))
      return term->emitOpError()
             << "cannot terminate a threaded body region; expected "
                "'uir.unreachable'";
    if (!isBody && !isa<SignatureOp>(term))
      return term->emitOpError()
             << "cannot terminate a threaded signature region; expected "
                "'uir.signature'";
    for (auto &op : block.without_terminator()) {
      auto phaseOp = dyn_cast<PhaseOp>(&op);
      if (!phaseOp)
        return op.emitOpError()
               << "is not a 'uir.phase' op or the region terminator; "
                  "split-funcs expects fully threaded phase form";
      if (!isa<PhaseReturnOp>(phaseOp.getBody().front().getTerminator()))
        return phaseOp.emitOpError()
               << "phase body is not terminated by 'uir.phase_return'; "
                  "structured control flow inside a phase is not supported by "
                  "split-funcs";
    }
    return success();
  };
  if (failed(checkRegion(funcOp.getSignature(), /*isBody=*/false)) ||
      failed(checkRegion(funcOp.getBody(), /*isBody=*/true)))
    return failure();

  // Content later than the declared span has no executor. The span maximum is
  // the range over the declared argument and result phases, always including
  // phase 0; this uniformly catches call-implied late phases, since calls are
  // already decomposed into `uir.phase_call` entries living in phase ops.
  int spanHi = 0;
  for (auto p : argPhases)
    spanHi = std::max<int>(spanHi, p);
  for (auto p : resultPhases)
    spanHi = std::max<int>(spanHi, p);
  for (auto &op : funcOp.getBody().front().without_terminator()) {
    auto phaseOp = cast<PhaseOp>(&op);
    if (phaseOp.getPhase() > spanHi)
      return phaseOp.emitError()
             << "phase " << phaseOp.getPhase()
             << " runs past the function's last declared phase " << spanHi
             << ", which has no executor; declare a result at phase "
             << phaseOp.getPhase() << " (e.g. a dyn result) to extend the span";
  }
  return success();
}

/// Compute the declared span and the contiguous body/signature phase ranges,
/// and collect the phase ops of both regions.
void FuncSplitter::computeRanges() {
  spanMin = spanMax = 0;
  for (auto p : argPhases) {
    spanMin = std::min<int>(spanMin, p);
    spanMax = std::max<int>(spanMax, p);
  }
  for (auto p : resultPhases) {
    spanMin = std::min<int>(spanMin, p);
    spanMax = std::max<int>(spanMax, p);
  }
  for (auto &op : funcOp.getBody().front().without_terminator()) {
    auto phaseOp = cast<PhaseOp>(&op);
    bodyPhase[phaseOp.getPhase()] = phaseOp;
  }
  for (auto &op : funcOp.getSignature().front().without_terminator()) {
    auto phaseOp = cast<PhaseOp>(&op);
    sigPhase[phaseOp.getPhase()] = phaseOp;
  }
  sigTerminator =
      cast<SignatureOp>(funcOp.getSignature().front().getTerminator());
  bodyMin = bodyPhase.begin()->first;
  bodyMax = bodyPhase.rbegin()->first;
}

/// Compute the ordered context slots entering each phase. A boundary P-1 -> P
/// carries the deduplicated union of the body crossings the next phase consumes
/// (the yields of body phase P-1) and the signature crossings a later phase's
/// signature depends on (the results of signature phase P-1 that signature
/// phase P references). Both regions are threaded independently, so a dependent
/// signature type computed one phase earlier than it is used is delivered here
/// even when the body never threads that type itself.
void FuncSplitter::computeSlots() {
  for (int phase = bodyMin + 1; phase <= bodyMax; ++phase) {
    auto &list = slots[phase];
    auto &index = slotIndex[phase];
    auto addSlot = [&](Value source) {
      auto key = canonKey(source);
      if (index.count(key))
        return;
      index[key] = list.size();
      list.push_back({key, source});
    };
    auto bodyIt = bodyPhase.find(phase - 1);
    if (bodyIt != bodyPhase.end()) {
      auto ret =
          cast<PhaseReturnOp>(bodyIt->second.getBody().front().getTerminator());
      for (auto yield : ret.getYields())
        addSlot(yield);
    }
    auto prevSig = sigPhase.find(phase - 1);
    auto curSig = sigPhase.find(phase);
    if (prevSig != sigPhase.end() && curSig != sigPhase.end()) {
      // The signature crossings of this boundary are the previous signature
      // phase's results that the current signature phase reads.
      SetVector<Value> referenced;
      curSig->second.getBody().walk([&](Operation *op) {
        for (auto operand : op->getOperands())
          if (operand.getDefiningOp() == prevSig->second.getOperation())
            referenced.insert(operand);
      });
      for (auto v : referenced)
        addSlot(v);
    }
  }
}

//===----------------------------------------------------------------------===//
// Per-Phase Function Bodies
//===----------------------------------------------------------------------===//

/// Create the empty per-phase `hir.func` shell for one body phase, with its
/// declared arguments, a trailing context argument for every phase but the
/// first, its declared results, and a trailing context result for every phase
/// but the last.
hir::FuncOp FuncSplitter::buildPhaseFunc(int phase) {
  SmallVector<Attribute> argNames, resultNames;
  for (auto [i, p] : llvm::enumerate(argPhases))
    if ((int)p == phase)
      argNames.push_back(funcOp.getArgNames()[i]);
  if (phase != bodyMin)
    argNames.push_back(StringAttr::get(ctx, "ctx"));
  for (auto [i, p] : llvm::enumerate(resultPhases))
    if ((int)p == phase)
      resultNames.push_back(funcOp.getResultNames()[i]);
  if (phase != bodyMax)
    resultNames.push_back(StringAttr::get(ctx, "ctx"));

  OpBuilder builder(splitFunc);
  auto func = hir::FuncOp::create(
      builder, funcOp.getLoc(), StringAttr::get(ctx, phaseName(phase)),
      StringAttr::get(ctx, "private"), ArrayAttr::get(ctx, argNames),
      ArrayAttr::get(ctx, resultNames), UnitAttr{});
  Block &sigBlock = func.getSignature().emplaceBlock();
  Block &bodyBlock = func.getBody().emplaceBlock();
  for (unsigned i = 0, e = argNames.size(); i < e; ++i) {
    sigBlock.addArgument(anyTy, funcOp.getLoc());
    bodyBlock.addArgument(anyTy, funcOp.getLoc());
  }
  symbolTable.insert(func);
  return func;
}

/// Resolve a `uir.phase_call` to the concrete function implementing that phase
/// of the callee, through the callee `uir.split_func`'s phase list. Nothing
/// resolves by name convention: the list is the authoritative mapping, and it
/// is filled for every function being split before any call is lowered.
FailureOr<FlatSymbolRefAttr> FuncSplitter::resolvePhaseEntry(PhaseCallOp call) {
  auto *calleeOp = symbolTable.lookup(call.getCallee());
  assert(calleeOp && "callee of a verified 'uir.phase_call' must resolve");
  auto calleeSplit = dyn_cast<SplitFuncOp>(calleeOp);
  if (!calleeSplit)
    return call.emitOpError()
           << "callee @" << call.getCallee()
           << " is not a 'uir.split_func'; a phase call must name the "
              "function's late-binding interface, not its implementation";
  for (auto [phase, entry] :
       llvm::zip(calleeSplit.getPhaseNumbers(), calleeSplit.getPhaseFuncs()))
    if (phase == call.getCalleePhase())
      return cast<FlatSymbolRefAttr>(entry);
  return call.emitOpError()
         << "callee @" << call.getCallee() << " has no function for phase "
         << call.getCalleePhase() << "; its implementation is not available";
}

/// Fill in a per-phase func's body: unpack the incoming context, clone the
/// phase's straight-line content (rewriting `uir.phase_call` to `hir.call`),
/// and terminate with an `hir.return` carrying the declared results plus the
/// packed outgoing context.
LogicalResult FuncSplitter::buildPhaseBody(int phase, hir::FuncOp func) {
  PhaseOp phaseOp = bodyPhase.at(phase);
  Block &block = func.getBody().front();
  auto loc = func.getLoc();
  IRMapping map;

  // Map declared arguments at this phase to the func's leading block args. Both
  // the body and signature copies of an argument map to the same func arg, so a
  // relocated signature type computation resolves its argument leaves here too.
  unsigned argIdx = 0;
  for (auto [i, p] : llvm::enumerate(argPhases))
    if ((int)p == phase) {
      Value funcArg = block.getArgument(argIdx++);
      map.map(funcOp.getBody().getArgument(i), funcArg);
      map.map(funcOp.getSignature().getArgument(i), funcArg);
    }

  // Cache one opaque type value at the block front for context plumbing.
  hir::OpaqueTypeOp opaqueType;
  auto getOpaque = [&]() -> Value {
    if (!opaqueType) {
      OpBuilder b(&block, block.begin());
      opaqueType = hir::OpaqueTypeOp::create(b, loc);
    }
    return opaqueType.getResult();
  };

  // Unpack the incoming context lazily: the unpack exposes every slot entering
  // this phase, and `getIncoming` returns the one carrying a given key.
  hir::OpaqueUnpackOp inUnpack;
  auto getIncoming = [&](const std::string &key) -> Value {
    if (!inUnpack) {
      OpBuilder b(&block, block.begin());
      inUnpack = hir::OpaqueUnpackOp::create(
          b, loc, SmallVector<Type>(slots.at(phase).size(), anyTy),
          block.getArgument(argIdx));
    }
    return inUnpack.getResult(slotIndex.at(phase).at(key));
  };

  OpBuilder builder(&block, block.end());

  // Produce a value from either region inside this func's body: a value already
  // built maps directly, a crossing from an earlier phase comes from the
  // incoming context, and any other pure computation is cloned in place. Used
  // both to seed the incoming body crossings and to relocate the outgoing
  // signature crossings so they ride the same context as the body crossings.
  // Returns a null value after emitting a diagnostic for a signature
  // computation that cannot be relocated.
  std::function<Value(Value)> resolveBody = [&](Value v) -> Value {
    if (Value mapped = map.lookupOrNull(v))
      return mapped;
    Value result;
    if (auto phaseResult = dyn_cast<OpResult>(v);
        phaseResult && isa<PhaseOp>(phaseResult.getOwner())) {
      auto phaseOp = cast<PhaseOp>(phaseResult.getOwner());
      if (phaseOp.getPhase() == phase) {
        auto ret =
            cast<PhaseReturnOp>(phaseOp.getBody().front().getTerminator());
        result = resolveBody(ret.getYields()[phaseResult.getResultNumber()]);
      } else {
        result = getIncoming(canonKey(v));
      }
    } else {
      // Relocating a signature computation into the body re-evaluates it
      // there, which only a pure, region-free computation supports. A
      // dependent type computed by a call crosses this line: its
      // `uir.phase_call` would have to be re-created in the body with its own
      // context chain, which this pass does not build.
      Operation *defOp = v.getDefiningOp();
      assert(defOp && "block argument outside the function's own arguments");
      if (defOp->getNumRegions() != 0 || !isMemoryEffectFree(defOp)) {
        defOp->emitOpError()
            << "cannot be relocated into the phase " << phase
            << " body; a signature computation crossing a phase boundary must "
               "be pure and region-free";
        return {};
      }
      IRMapping cloneMap;
      for (auto operand : defOp->getOperands()) {
        Value resolved = resolveBody(operand);
        if (!resolved)
          return {};
        cloneMap.map(operand, resolved);
      }
      Operation *clone = builder.clone(*defOp, cloneMap);
      result = clone->getResult(cast<OpResult>(v).getResultNumber());
    }
    map.map(v, result);
    return result;
  };

  // Seed the incoming body crossings this phase consumes, so cloning the phase
  // content resolves them to the unpacked context.
  if (phase != bodyMin)
    for (auto result : bodyPhase.at(phase - 1).getResults())
      map.map(result, getIncoming(canonKey(result)));

  // Clone the phase content in order. A `uir.phase_call` becomes an `hir.call`
  // to the callee's concrete per-phase func, its context operand and result
  // threaded like any other value.
  for (auto &op : phaseOp.getBody().front().without_terminator()) {
    if (auto call = dyn_cast<PhaseCallOp>(&op)) {
      SmallVector<Value> args, typeOfArgs, typeOfResults;
      for (auto v : call.getArguments())
        args.push_back(map.lookup(v));
      for (auto v : call.getTypeOfArgs())
        typeOfArgs.push_back(map.lookup(v));
      for (auto v : call.getTypeOfResults())
        typeOfResults.push_back(map.lookup(v));
      if (call.getContextIn()) {
        args.push_back(map.lookup(call.getContextIn()));
        typeOfArgs.push_back(getOpaque());
      }
      if (call.getContextOut())
        typeOfResults.push_back(getOpaque());
      auto calleeSym = resolvePhaseEntry(call);
      if (failed(calleeSym))
        return failure();
      auto hirCall =
          hir::CallOp::create(builder, call.getLoc(),
                              SmallVector<Type>(typeOfResults.size(), anyTy),
                              *calleeSym, args, typeOfArgs, typeOfResults);
      // The callee's declared results map to the leading `hir.call` results;
      // the callee context result (if any) maps to the trailing one.
      for (auto [orig, repl] :
           llvm::zip(call.getResults(), hirCall.getResults()))
        map.map(orig, repl);
      if (call.getContextOut())
        map.map(call.getContextOut(), hirCall.getResults().back());
    } else {
      // Every operand is either an interior value already cloned into the func
      // (hence in `map`), an argument at this phase, or an incoming crossing
      // seeded from the context. An operand missing from `map` is a value that
      // reached this phase without being threaded to it — a violation of the
      // thread-phases crossing contract; assert loudly rather than clone a
      // dangling cross-region reference.
      for (auto operand : op.getOperands())
        assert(map.contains(operand) &&
               "phase content references a value not threaded to its phase");
      builder.clone(op, map);
    }
  }

  // Build the terminator. Declared results and their types ride the return;
  // the yielded crossings are packed into the trailing context result.
  auto ret = cast<PhaseReturnOp>(phaseOp.getBody().front().getTerminator());
  SmallVector<Value> values, typeOfValues;
  for (auto v : ret.getValues())
    values.push_back(map.lookup(v));
  for (auto v : ret.getTypeOfValues())
    typeOfValues.push_back(map.lookup(v));
  if (phase != bodyMax) {
    // Pack every slot crossing to the next phase: the body crossings this phase
    // yields and the signature crossings a later signature depends on, the
    // latter relocated (cloned) into this body so they ride the same context.
    SmallVector<Value> outgoing;
    for (auto &slot : slots.at(phase + 1)) {
      Value value = resolveBody(slot.source);
      if (!value)
        return failure();
      outgoing.push_back(value);
    }
    auto pack = hir::OpaquePackOp::create(builder, loc, anyTy, outgoing);
    values.push_back(pack.getResult());
    typeOfValues.push_back(getOpaque());
  }
  hir::ReturnOp::create(builder, loc, values, typeOfValues);
  return success();
}

//===----------------------------------------------------------------------===//
// Per-Phase Signatures
//===----------------------------------------------------------------------===//

/// Realize the type value `sigValue` (an operand of the threaded
/// `uir.signature`, delivered at `phase`) inside a per-phase func's signature
/// region. Interior computations are cloned; a value crossing from an earlier
/// phase is looked up by structural key in this boundary's context slots and
/// unpacked from the incoming context (the producing body packs that slot).
/// Returns a null value after emitting a diagnostic for a computation that
/// cannot be cloned into a signature region.
Value FuncSplitter::realizeSigType(int phase, Value sigValue, hir::FuncOp func,
                                   OpBuilder &builder, IRMapping &memo,
                                   SigCtx &sigCtx) {
  if (Value cached = memo.lookupOrNull(sigValue))
    return cached;

  Value result;
  if (auto blockArg = dyn_cast<BlockArgument>(sigValue)) {
    // A declared argument at this phase: the func's leading signature args.
    assert(blockArg.getOwner()->getParentOp() == funcOp);
    unsigned argIdx = 0;
    for (auto [i, p] : llvm::enumerate(argPhases)) {
      if ((int)p != phase)
        continue;
      if (i == blockArg.getArgNumber()) {
        result = func.getSignature().front().getArgument(argIdx);
        break;
      }
      ++argIdx;
    }
    assert(result && "signature references an argument of another phase");
  } else if (auto phaseOp = dyn_cast<PhaseOp>(sigValue.getDefiningOp())) {
    if (phaseOp.getPhase() < phase) {
      // A crossing from an earlier phase: look up its context slot and unpack
      // it from the incoming context. computeSlots guarantees a slot exists.
      auto &index = slotIndex.at(phase);
      auto it = index.find(canonKey(sigValue));
      assert(it != index.end() &&
             "dependent signature type has no context slot");
      if (!sigCtx.unpack) {
        Value ctxArg = func.getSignature().front().getArguments().back();
        OpBuilder b(&func.getSignature().front(),
                    func.getSignature().front().begin());
        sigCtx.unpack = hir::OpaqueUnpackOp::create(
            b, func.getLoc(), SmallVector<Type>(slots.at(phase).size(), anyTy),
            ctxArg);
      }
      result = sigCtx.unpack.getResult(it->second);
    } else {
      // Delivered at this phase: realize the value it yields.
      auto ret = cast<PhaseReturnOp>(phaseOp.getBody().front().getTerminator());
      Value yielded =
          ret.getYields()[cast<OpResult>(sigValue).getResultNumber()];
      result = realizeSigType(phase, yielded, func, builder, memo, sigCtx);
    }
  } else {
    // A per-phase signature region holds the type computation of its own
    // phase's arguments and results, cloned from the threaded signature. Only
    // a pure, region-free computation can be re-evaluated there; a type
    // computed by a call would need its own context chain in the signature.
    Operation *defOp = sigValue.getDefiningOp();
    if (defOp->getNumRegions() != 0 || !isMemoryEffectFree(defOp)) {
      defOp->emitOpError() << "cannot be cloned into the phase " << phase
                           << " signature; a signature type computation must "
                              "be pure and region-free";
      return {};
    }
    IRMapping cloneMap;
    for (auto operand : defOp->getOperands()) {
      Value resolved =
          realizeSigType(phase, operand, func, builder, memo, sigCtx);
      if (!resolved)
        return {};
      cloneMap.map(operand, resolved);
    }
    Operation *clone = builder.clone(*defOp, cloneMap);
    result = clone->getResult(cast<OpResult>(sigValue).getResultNumber());
  }
  if (!result)
    return {};

  memo.map(sigValue, result);
  return result;
}

/// Build a per-phase func's signature region: the type of each declared
/// argument and result at this phase, plus an opaque type for each context
/// argument and result.
LogicalResult FuncSplitter::buildPhaseSignature(int phase, hir::FuncOp func) {
  Block &block = func.getSignature().front();
  OpBuilder builder(&block, block.end());
  IRMapping memo;
  SigCtx sigCtx;
  hir::OpaqueTypeOp opaqueType;
  auto getOpaque = [&]() -> Value {
    if (!opaqueType) {
      OpBuilder b(&block, block.begin());
      opaqueType = hir::OpaqueTypeOp::create(b, func.getLoc());
    }
    return opaqueType.getResult();
  };

  SmallVector<Value> typeOfArgs, typeOfResults;
  for (auto [i, p] : llvm::enumerate(argPhases)) {
    if ((int)p != phase)
      continue;
    Value type = realizeSigType(phase, sigTerminator.getTypeOfArgs()[i], func,
                                builder, memo, sigCtx);
    if (!type)
      return failure();
    typeOfArgs.push_back(type);
  }
  if (phase != bodyMin)
    typeOfArgs.push_back(getOpaque());
  for (auto [i, p] : llvm::enumerate(resultPhases)) {
    if ((int)p != phase)
      continue;
    Value type = realizeSigType(phase, sigTerminator.getTypeOfResults()[i],
                                func, builder, memo, sigCtx);
    if (!type)
      return failure();
    typeOfResults.push_back(type);
  }
  if (phase != bodyMax)
    typeOfResults.push_back(getOpaque());

  hir::SignatureOp::create(builder, func.getLoc(), typeOfArgs, typeOfResults);
  return success();
}

//===----------------------------------------------------------------------===//
// Structural Ops
//===----------------------------------------------------------------------===//

/// Fold any pre-span phases into an `hir.eager_func` and fill the
/// `uir.split_func`'s phase list, mapping each span phase to its entry.
void FuncSplitter::buildEagerFuncAndLinks() {
  OpBuilder builder(splitFunc);
  auto loc = funcOp.getLoc();

  // The span entry for phase `spanMin` is normally its per-phase func. When
  // pre-span phases precede it, they and the (renamed) span-entry func become
  // the links of an `hir.eager_func` that takes over the span-entry name.
  SmallVector<int32_t> phaseNumbers;
  SmallVector<Attribute> phaseFuncs;
  for (int phase = spanMin; phase <= spanMax; ++phase) {
    phaseNumbers.push_back(phase);
    phaseFuncs.push_back(FlatSymbolRefAttr::get(phaseFunc.at(phase)));
  }

  if (bodyMin < spanMin) {
    hir::FuncOp entry = phaseFunc.at(spanMin);
    SmallVector<Attribute> links;
    for (int phase = bodyMin; phase < spanMin; ++phase)
      links.push_back(FlatSymbolRefAttr::get(phaseFunc.at(phase)));
    // The earliest span func becomes the eager chain's tail; rename it so the
    // span-entry name is free for the eager func itself. The eager func's
    // external interface is the tail's declared arguments (all but the trailing
    // pre-span context argument) and its results.
    symbolTable.remove(entry);
    SymbolTable::setSymbolName(entry, phaseName(spanMin) + ".tail");
    symbolTable.insert(entry);
    links.push_back(FlatSymbolRefAttr::get(entry));
    SmallVector<Attribute> eagerArgNames(entry.getArgNames().begin(),
                                         entry.getArgNames().end() - 1);
    auto eager = hir::EagerFuncOp::create(
        builder, loc, StringAttr::get(ctx, phaseName(spanMin)), StringAttr{},
        ArrayAttr::get(ctx, eagerArgNames), entry.getResultNamesAttr(),
        ArrayAttr::get(ctx, links));
    symbolTable.insert(eager);
    phaseFuncs.front() = FlatSymbolRefAttr::get(eager);
  }

  splitFunc.setPhaseNumbers(phaseNumbers);
  splitFunc.setPhaseFuncsAttr(ArrayAttr::get(ctx, phaseFuncs));
}

//===----------------------------------------------------------------------===//
// Driver
//===----------------------------------------------------------------------===//

/// Check the threaded input and create everything callers bind against: the
/// per-phase function shells, the eager chain, and the split_func's phase list.
LogicalResult FuncSplitter::declare() {
  if (failed(diagnose()))
    return failure();
  computeRanges();
  computeSlots();
  for (auto &[phase, op] : bodyPhase)
    phaseFunc[phase] = buildPhaseFunc(phase);
  buildEagerFuncAndLinks();
  return success();
}

/// Fill the per-phase functions. Runs only after every function's phase list
/// exists, since lowering a `uir.phase_call` resolves the callee's entry
/// through its list.
LogicalResult FuncSplitter::implement() {
  for (auto &[phase, func] : phaseFunc) {
    if (failed(buildPhaseBody(phase, func)) ||
        failed(buildPhaseSignature(phase, func)))
      return failure();
  }
  return success();
}

/// Drop the implementation once every function has been implemented, so that
/// symbol resolution during implementation is independent of the order the
/// module's functions are processed in.
void FuncSplitter::retire() {
  splitFunc.removeImplAttr();
  symbolTable.erase(funcOp);
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct SplitFuncsPass : public uir::impl::SplitFuncsPassBase<SplitFuncsPass> {
  void runOnOperation() override {
    SymbolTable symbolTable(getOperation());

    // The pass is driven by the `uir.split_func` ops phase sorting left in the
    // sorted state; a split_func without an implementation reference is either
    // an uncompiled declaration or already split, and is left alone.
    SmallVector<std::unique_ptr<FuncSplitter>> splitters;
    DenseSet<Operation *> implementations;
    for (auto splitFunc :
         SmallVector<SplitFuncOp>(getOperation().getOps<SplitFuncOp>())) {
      auto impl = splitFunc.getImplAttr();
      if (!impl)
        continue;
      auto funcOp = symbolTable.lookup<FuncOp>(impl.getValue());
      assert(funcOp && "implementation of a verified 'uir.split_func' must "
                       "resolve to a 'uir.func'");
      implementations.insert(funcOp);
      splitters.push_back(
          std::make_unique<FuncSplitter>(splitFunc, funcOp, symbolTable));
    }

    // Every remaining function must be one of those implementations. A
    // `uir.func` on its own has no declaration snapshot to fill in and no
    // caller can bind to its phases, which means the pipeline skipped sorting.
    for (auto funcOp : getOperation().getOps<FuncOp>()) {
      if (implementations.contains(funcOp))
        continue;
      funcOp.emitOpError()
          << "is not the implementation of a 'uir.split_func'; run "
             "'sort-phases' to take custody of the declaration first";
      return signalPassFailure();
    }

    // Declare every function before implementing any: a `uir.phase_call` is
    // lowered against the callee split_func's phase list, so all lists must
    // exist first. Recursive and mutually recursive callees fall out of this.
    for (auto &splitter : splitters)
      if (failed(splitter->declare()))
        return signalPassFailure();
    for (auto &splitter : splitters)
      if (failed(splitter->implement()))
        return signalPassFailure();
    for (auto &splitter : splitters)
      splitter->retire();
  }
};
} // namespace
