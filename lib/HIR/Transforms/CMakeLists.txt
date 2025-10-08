add_silicon_dialect_library(SiliconHIRTransforms
  CheckCalls.cpp
  EvalConsts.cpp
  InferTypes.cpp
  SplitPhases.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
