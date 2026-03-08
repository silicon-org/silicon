add_silicon_dialect_library(SiliconHIRTransforms
  CheckCalls.cpp
  CheckTypes.cpp
  InferTypes.cpp
  SpecializeFuncs.cpp
  SplitPhases.cpp
  TestPhaseAnalysis.cpp

  LINK_LIBS PUBLIC
  SiliconHIR
  SiliconHIRAnalysis
  SiliconMIR
  MLIRControlFlowDialect
  MLIRIR
  MLIRPass
  MLIRTransformUtils
)
