add_silicon_dialect_library(SiliconHIRTransforms
  CheckCalls.cpp
  EvalConsts.cpp
  InferTypes.cpp
  SpecializeFuncs.cpp
  SplitPhases.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconMIR
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
