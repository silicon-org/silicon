set(LLVM_LINK_COMPONENTS native)
add_silicon_dialect_library(SiliconHIRTransforms
  EvalConstRegions.cpp
  EvalConsts.cpp
  InferTypes.cpp

  LINK_LIBS PUBLIC
  MLIRArithToLLVM
  MLIRBuiltinToLLVMIRTranslation
  MLIRControlFlowToLLVM
  MLIRExecutionEngine
  MLIRFuncToLLVM
  MLIRIndexToLLVM
  MLIRIR
  MLIRLLVMCommonConversion
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  SiliconHIR
)
