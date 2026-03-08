add_silicon_dialect_library(SiliconHIRTransforms
  CheckCalls.cpp
  EvalConsts.cpp
  InferTypes.cpp
  SpecializeFuncs.cpp
  SplitPhases.cpp
  TestPhaseAnalysis.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconHIRAnalysis
  SiliconMIR
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
